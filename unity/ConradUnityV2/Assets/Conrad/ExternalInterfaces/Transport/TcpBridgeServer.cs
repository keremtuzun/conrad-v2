// Dependency-free transport for the Conrad bridge: length-prefixed JSON frames over TCP (System.Net.Sockets).
// Frame = 4-byte unsigned big-endian payload length + UTF-8 JSON envelope (conrad/adapters/unity/transport.py,
// TcpBridgeTransport). Listeners:
//   control  request/reply - RobotHardwareInterface surface (handshake, reset, step, poll, command, fault, metrics,
//                            configure_scene, frame_probe); one client at a time, a newer client replaces an older one
//   stream   push          - SENSOR frames in free-running mode (optional)
//   truth    request/reply - ground truth, ONLY when the experiment sets truth_endpoint_enabled (training/evaluation)
// Everything is serviced on the Unity main thread (Unity/PhysX APIs are main-thread only). Endpoints must be literal
// loopback/private IPs; wildcard binds are refused, and a connecting peer that is not loopback/private is dropped
// (ch34 Network).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using Conrad.UnityV2.Core;
using Conrad.UnityV2.MissionExperimentRuntime;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Conrad.UnityV2.ExternalInterfaces.Transport
{
    public static class BridgeEndpointPolicy
    {
        public static bool IsLoopbackOrPrivate(IPAddress ip)
        {
            if (ip == null) return false;
            if (ip.IsIPv4MappedToIPv6) ip = ip.MapToIPv4();
            if (IPAddress.IsLoopback(ip)) return true;
            byte[] b = ip.GetAddressBytes();
            if (b.Length == 4)
                return b[0] == 10 || (b[0] == 172 && b[1] >= 16 && b[1] <= 31) || (b[0] == 192 && b[1] == 168)
                       || (b[0] == 169 && b[1] == 254);
            return ip.IsIPv6LinkLocal || (b[0] & 0xFE) == 0xFC; // fe80::/10, fc00::/7
        }

        /// <summary>tcp://&lt;literal loopback/private IP&gt;:&lt;port&gt; -> IPEndPoint, else ArgumentException.</summary>
        public static IPEndPoint Parse(string endpoint)
        {
            if (string.IsNullOrEmpty(endpoint) || !endpoint.StartsWith("tcp://", StringComparison.Ordinal))
                throw new ArgumentException("only tcp://<ip>:<port> endpoints are allowed: " + endpoint);
            string hostPort = endpoint.Substring("tcp://".Length);
            int colon = hostPort.LastIndexOf(':');
            if (colon <= 0 || !int.TryParse(hostPort.Substring(colon + 1), out int port) || port <= 0 || port > 65535)
                throw new ArgumentException("endpoint needs an explicit port: " + endpoint);
            string host = hostPort.Substring(0, colon).Trim('[', ']');
            if (!IPAddress.TryParse(host, out IPAddress ip))
                throw new ArgumentException("endpoint host must be a literal IP address: " + endpoint);
            if (ip.Equals(IPAddress.Any) || ip.Equals(IPAddress.IPv6Any))
                throw new ArgumentException("wildcard binds are forbidden for bridge endpoints");
            if (!IsLoopbackOrPrivate(ip)) throw new ArgumentException("endpoint must be loopback or private: " + endpoint);
            return new IPEndPoint(ip, port);
        }
    }

    /// <summary>4-byte big-endian length prefix + payload. Identical to the Python framing.</summary>
    public static class LengthPrefixedFraming
    {
        public const int MaxFrameBytes = 64 * 1024 * 1024;

        public static void Write(Stream s, byte[] payload)
        {
            if (payload.Length > MaxFrameBytes) throw new IOException("frame exceeds " + MaxFrameBytes + " bytes");
            int n = payload.Length;
            var header = new byte[] { (byte)(n >> 24), (byte)(n >> 16), (byte)(n >> 8), (byte)n };
            s.Write(header, 0, 4);
            s.Write(payload, 0, n);
            s.Flush();
        }

        private static void ReadExactly(Stream s, byte[] buffer, int count)
        {
            int got = 0;
            while (got < count)
            {
                int r = s.Read(buffer, got, count - got);
                if (r <= 0) throw new EndOfStreamException("peer closed the connection mid-frame");
                got += r;
            }
        }

        /// <summary>Reads one complete frame (blocking up to the stream's ReadTimeout).</summary>
        public static byte[] Read(Stream s)
        {
            var header = new byte[4];
            ReadExactly(s, header, 4);
            long n = ((long)header[0] << 24) | ((long)header[1] << 16) | ((long)header[2] << 8) | header[3];
            if (n > MaxFrameBytes) throw new IOException("announced frame of " + n + " bytes exceeds the limit");
            var payload = new byte[n];
            ReadExactly(s, payload, (int)n);
            return payload;
        }
    }

    internal sealed class FramedConnection : IDisposable
    {
        public readonly TcpClient Client;
        public readonly NetworkStream Stream;
        public readonly string Peer;

        public FramedConnection(TcpClient client, int ioTimeoutMs)
        {
            Client = client;
            Client.NoDelay = true;
            Client.ReceiveTimeout = ioTimeoutMs;
            Client.SendTimeout = ioTimeoutMs;
            Stream = client.GetStream();
            Stream.ReadTimeout = ioTimeoutMs;
            Stream.WriteTimeout = ioTimeoutMs;
            Peer = client.Client.RemoteEndPoint?.ToString() ?? "?";
        }

        /// <summary>True when at least one byte is waiting; throws EndOfStreamException when the peer closed.</summary>
        public bool HasData(int waitMicroseconds)
        {
            Socket s = Client.Client;
            if (!s.Poll(waitMicroseconds, SelectMode.SelectRead)) return false;
            if (s.Available == 0) throw new EndOfStreamException("peer " + Peer + " disconnected");
            return true;
        }

        public void Dispose()
        {
            try { Stream.Dispose(); } catch (Exception) { /* already closed */ }
            try { Client.Close(); } catch (Exception) { /* already closed */ }
        }
    }

    [RequireComponent(typeof(ExperimentRuntime))]
    public sealed class TcpBridgeServer : MonoBehaviour
    {
        [Tooltip("Longest time one Update may spend serving requests in lock-step mode (ms).")]
        public int lockStepServiceBudgetMs = 200;
        [Tooltip("Socket read/write timeout once a frame has started (ms).")]
        public int ioTimeoutMs = 10000;

        private ExperimentRuntime _rt;
        private TcpListener _controlListener, _truthListener, _streamListener;
        private FramedConnection _control, _truth;
        private readonly List<FramedConnection> _streamClients = new List<FramedConnection>();
        private BridgeProtocolHandler _controlHandler, _truthHandler;
        private bool _faulted;
        private long _streamSeq;

        public bool Listening => _controlListener != null;
        public bool HasControlClient => _control != null;
        public string ControlEndpoint { get; private set; }

        private static TcpListener Listen(string endpoint)
        {
            IPEndPoint ep = BridgeEndpointPolicy.Parse(endpoint);
            var l = new TcpListener(ep);
            l.Server.ExclusiveAddressUse = true;
            l.Start(4);
            return l;
        }

        private void Start()
        {
            _rt = GetComponent<ExperimentRuntime>();
            if (!_rt.Ready)
            {
                Debug.LogError("[Conrad] bridge not started: " + _rt.FailureReason);
                enabled = false;
                return;
            }
            var bridge = J.Obj(_rt.Experiment, "bridge");
            ControlEndpoint = J.Str(bridge, "control_endpoint");
            _controlHandler = new BridgeProtocolHandler(_rt, EndpointRole.CONTROL);
            _controlListener = Listen(ControlEndpoint);
            string streamEp = J.StrOrNull(bridge, "stream_endpoint");
            if (streamEp != null) _streamListener = Listen(streamEp);
            string truthEp = J.StrOrNull(bridge, "truth_endpoint");
            if (truthEp != null && _rt.TruthEndpointEnabled)
            {
                _truthHandler = new BridgeProtocolHandler(_rt, EndpointRole.TRUTH);
                _truthListener = Listen(truthEp);
            }
            Debug.Log("[Conrad] bridge up: transport=tcp_length_prefixed_json control=" + ControlEndpoint
                      + " truth=" + (_truthListener != null ? truthEp : "off") + " validity=" + _rt.ValidityLevel
                      + " lock_step=" + _rt.LockStep + " digest=" + _rt.Robot.Digest);
        }

        private FramedConnection AcceptOne(TcpListener listener, FramedConnection current, string what)
        {
            while (listener != null && listener.Pending())
            {
                TcpClient c = listener.AcceptTcpClient();
                var remote = c.Client.RemoteEndPoint as IPEndPoint;
                if (remote == null || !BridgeEndpointPolicy.IsLoopbackOrPrivate(remote.Address))
                {
                    Debug.LogWarning("[Conrad] refused " + what + " peer " + remote);
                    c.Close();
                    continue;
                }
                current?.Dispose(); // the newest client owns the endpoint; it must handshake again
                current = new FramedConnection(c, ioTimeoutMs);
            }
            return current;
        }

        private byte[] Reply(BridgeProtocolHandler handler, byte[] request)
        {
            if (_faulted)
                return BridgeProtocolHandler.Encode("ERROR", new Dictionary<string, object>
                {
                    ["code"] = "SIMULATOR_FAULTED", ["detail"] = "the runtime raised; restart the experiment",
                }, handler.SessionId, 0);
            try { return handler.Handle(request); }
            catch (Exception ex)
            {
                // Internal failure (e.g. non-finite dynamics): stop serving the world, never guess.
                _faulted = true;
                Debug.LogException(ex);
                return BridgeProtocolHandler.Encode("ERROR", new Dictionary<string, object>
                {
                    ["code"] = "SIMULATOR_FAULTED", ["detail"] = ex.GetType().Name + ": " + ex.Message,
                }, handler.SessionId, 0);
            }
        }

        /// <summary>Serve at most one request on <paramref name="conn"/>. Returns false when nothing was pending.</summary>
        private bool ServeOne(ref FramedConnection conn, BridgeProtocolHandler handler, int waitMicroseconds)
        {
            if (conn == null) return false;
            try
            {
                if (!conn.HasData(waitMicroseconds)) return false;
                byte[] request = LengthPrefixedFraming.Read(conn.Stream);
                LengthPrefixedFraming.Write(conn.Stream, Reply(handler, request)); // exactly one reply per request
                return true;
            }
            catch (Exception ex) when (ex is IOException || ex is SocketException || ex is ObjectDisposedException)
            {
                Debug.Log("[Conrad] client " + conn.Peer + " dropped: " + ex.Message);
                conn.Dispose();
                conn = null;
                return false;
            }
        }

        private void Update()
        {
            if (_controlListener == null) return;
            var clock = Stopwatch.StartNew();
            int budgetMs = _rt.LockStep ? lockStepServiceBudgetMs : 1;
            do
            {
                _control = AcceptOne(_controlListener, _control, "control");
                _truth = AcceptOne(_truthListener, _truth, "truth");
                bool served = ServeOne(ref _control, _controlHandler, 0);
                served |= ServeOne(ref _truth, _truthHandler, 0);
                if (!served)
                {
                    if (_control == null && _truth == null) break;
                    // Idle: wait briefly for the next request instead of spinning or ending the frame.
                    var read = new List<Socket>();
                    if (_control != null) read.Add(_control.Client.Client);
                    if (_truth != null) read.Add(_truth.Client.Client);
                    Socket.Select(read, null, null, 1000);
                    if (read.Count == 0 && !_rt.LockStep) break;
                }
            } while (clock.ElapsedMilliseconds < budgetMs);
            PumpStream();
        }

        private void PumpStream()
        {
            if (_streamListener == null) return;
            while (_streamListener.Pending())
            {
                TcpClient c = _streamListener.AcceptTcpClient();
                var remote = c.Client.RemoteEndPoint as IPEndPoint;
                if (remote == null || !BridgeEndpointPolicy.IsLoopbackOrPrivate(remote.Address)) { c.Close(); continue; }
                _streamClients.Add(new FramedConnection(c, ioTimeoutMs));
            }
            if (_rt.LockStep || _faulted || !_controlHandler.HasSession || _rt.Server.CommLoss) return;
            var packets = _rt.Server.TakeDeliveredPackets();
            foreach (var pkt in packets)
            {
                byte[] frame = BridgeProtocolHandler.Encode("SENSOR", pkt.ToWire(_rt.Clock.Domain), _controlHandler.SessionId, ++_streamSeq);
                for (int i = _streamClients.Count - 1; i >= 0; i--)
                {
                    try { LengthPrefixedFraming.Write(_streamClients[i].Stream, frame); }
                    catch (Exception ex) when (ex is IOException || ex is SocketException || ex is ObjectDisposedException)
                    {
                        _streamClients[i].Dispose();
                        _streamClients.RemoveAt(i);
                    }
                }
            }
        }

        private void OnDestroy()
        {
            _control?.Dispose();
            _truth?.Dispose();
            foreach (var c in _streamClients) c.Dispose();
            _controlListener?.Stop();
            _truthListener?.Stop();
            _streamListener?.Stop();
        }
    }
}
