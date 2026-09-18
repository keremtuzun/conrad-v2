// ZeroMQ transport for the Conrad bridge (NetMQ, pure C#). Separate assembly so the simulator core compiles
// without NetMQ. Sockets:
//   control  REP  - RobotHardwareInterface surface (handshake, reset, step, poll, command, fault, metrics)
//   stream   PUB  - SENSOR packets in free-running mode
//   truth    REP  - ground truth, ONLY when the experiment sets truth_endpoint_enabled (training/evaluation)
// All sockets are serviced on the Unity main thread (Unity/PhysX APIs are main-thread only). Endpoints must be
// literal loopback/private addresses; wildcard binds are refused (ch34 Network).
using System;
using System.Collections.Generic;
using System.Net;
using Conrad.UnityV2.Core;
using Conrad.UnityV2.MissionExperimentRuntime;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;

namespace Conrad.UnityV2.ExternalInterfaces.Transport
{
    [RequireComponent(typeof(ExperimentRuntime))]
    public sealed class ZmqBridgeServer : MonoBehaviour
    {
        [Tooltip("Maximum requests serviced per rendered frame (lock-step throughput).")]
        public int maxRequestsPerFrame = 256;

        private ExperimentRuntime _rt;
        private ResponseSocket _control, _truth;
        private PublisherSocket _stream;
        private BridgeProtocolHandler _controlHandler, _truthHandler;
        private bool _faulted;
        private long _streamSeq;

        public static string ValidateEndpoint(string endpoint)
        {
            if (string.IsNullOrEmpty(endpoint) || !endpoint.StartsWith("tcp://", StringComparison.Ordinal))
                throw new ArgumentException("only tcp://<ip>:<port> endpoints are allowed: " + endpoint);
            string hostPort = endpoint.Substring("tcp://".Length);
            int colon = hostPort.LastIndexOf(':');
            if (colon <= 0 || !int.TryParse(hostPort.Substring(colon + 1), out int port) || port <= 0 || port > 65535)
                throw new ArgumentException("endpoint needs an explicit port: " + endpoint);
            if (!IPAddress.TryParse(hostPort.Substring(0, colon), out IPAddress ip))
                throw new ArgumentException("endpoint host must be a literal IP address: " + endpoint);
            byte[] b = ip.GetAddressBytes();
            bool loopback = IPAddress.IsLoopback(ip);
            bool priv = b.Length == 4 && (b[0] == 10 || (b[0] == 172 && b[1] >= 16 && b[1] <= 31) || (b[0] == 192 && b[1] == 168));
            if (ip.Equals(IPAddress.Any) || ip.Equals(IPAddress.IPv6Any))
                throw new ArgumentException("wildcard binds are forbidden for control endpoints");
            if (!(loopback || priv)) throw new ArgumentException("endpoint must be loopback or private: " + endpoint);
            return endpoint;
        }

        private void Start()
        {
            _rt = GetComponent<ExperimentRuntime>();
            var bridge = J.Obj(_rt.Experiment, "bridge");
            AsyncIO.ForceDotNet.Force(); // required by NetMQ inside the Unity (Mono) runtime
            _controlHandler = new BridgeProtocolHandler(_rt, EndpointRole.CONTROL);
            _control = new ResponseSocket();
            _control.Options.Linger = TimeSpan.Zero;
            _control.Bind(ValidateEndpoint(J.Str(bridge, "control_endpoint")));
            string streamEp = J.StrOrNull(bridge, "stream_endpoint");
            if (streamEp != null)
            {
                _stream = new PublisherSocket();
                _stream.Options.Linger = TimeSpan.Zero;
                _stream.Options.SendHighWatermark = 256;
                _stream.Bind(ValidateEndpoint(streamEp));
            }
            string truthEp = J.StrOrNull(bridge, "truth_endpoint");
            if (truthEp != null && _rt.TruthEndpointEnabled)
            {
                _truthHandler = new BridgeProtocolHandler(_rt, EndpointRole.TRUTH);
                _truth = new ResponseSocket();
                _truth.Options.Linger = TimeSpan.Zero;
                _truth.Bind(ValidateEndpoint(truthEp));
            }
            Debug.Log("[Conrad] bridge up: validity=" + _rt.ValidityLevel + " lock_step=" + _rt.LockStep + " digest=" + _rt.Robot.Digest);
        }

        private void Serve(ResponseSocket socket, BridgeProtocolHandler handler)
        {
            for (int i = 0; i < maxRequestsPerFrame; i++)
            {
                if (!socket.TryReceiveFrameBytes(TimeSpan.Zero, out byte[] request)) return;
                byte[] reply;
                if (_faulted)
                {
                    reply = BridgeProtocolHandler.Encode("ERROR", new Dictionary<string, object>
                    {
                        ["code"] = "SIMULATOR_FAULTED", ["detail"] = "the runtime raised; restart the experiment",
                    }, handler.SessionId, 0);
                }
                else
                {
                    try { reply = handler.Handle(request); }
                    catch (Exception ex)
                    {
                        // Internal failure (e.g. non-finite dynamics): stop serving the world, never guess.
                        _faulted = true;
                        Debug.LogException(ex);
                        reply = BridgeProtocolHandler.Encode("ERROR", new Dictionary<string, object>
                        {
                            ["code"] = "SIMULATOR_FAULTED", ["detail"] = ex.GetType().Name + ": " + ex.Message,
                        }, handler.SessionId, 0);
                    }
                }
                socket.SendFrame(reply); // REP must answer every request exactly once
            }
        }

        private void Update()
        {
            if (_control == null) return;
            Serve(_control, _controlHandler);
            if (_truth != null) Serve(_truth, _truthHandler);
            if (_stream != null && !_rt.LockStep && !_faulted && _controlHandler.HasSession && !_rt.Server.CommLoss)
            {
                foreach (var pkt in _rt.Server.TakeDeliveredPackets())
                    _stream.SendFrame(BridgeProtocolHandler.Encode("SENSOR", pkt.ToWire(_rt.Clock.Domain),
                        _controlHandler.SessionId, ++_streamSeq));
            }
        }

        private void OnDestroy()
        {
            _control?.Dispose();
            _stream?.Dispose();
            _truth?.Dispose();
            NetMQConfig.Cleanup(false);
        }
    }
}
