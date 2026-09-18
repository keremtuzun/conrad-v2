// Sensor framework (ch20 Sensor base class, ch21 Sensor framework / Sensor timestamps / Sensor degradation).
// Every sample carries (t_measurement, t_delivery): acquisition time is stamped when the physics state is
// sampled; the packet is released only after the configured latency. Noise/bias/drift/dropout come from a
// seeded per-sensor stream so replays are exact.
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using Conrad.UnityV2.Core;
using Conrad.UnityV2.VehicleDynamics;

namespace Conrad.UnityV2.SensorSimulation
{
    public enum SensorHealth { OK, DEGRADED, FAULT, UNKNOWN }

    public enum PayloadEncoding { f64le, f32le, u8 }

    public sealed class SensorPacketData
    {
        public string SensorName, Modality, FrameId, Layout, Units, CalibrationRef;
        public PayloadEncoding Encoding;
        public SensorHealth Health;
        public long AcquisitionNs, DeliveryNs, Sequence;
        public int[] Shape;
        public byte[] Payload;
        public Dictionary<string, object> Context = new Dictionary<string, object>();

        public static string Sha256Hex(byte[] data)
        {
            using (var sha = SHA256.Create())
            {
                byte[] h = sha.ComputeHash(data);
                var c = new char[h.Length * 2];
                const string hex = "0123456789abcdef";
                for (int i = 0; i < h.Length; i++) { c[2 * i] = hex[h[i] >> 4]; c[2 * i + 1] = hex[h[i] & 0xF]; }
                return new string(c);
            }
        }

        /// <summary>Wire body of a SensorPacket (conrad/adapters/unity/sensor_packet.py). Keys must match exactly.</summary>
        public Dictionary<string, object> ToWire(string clockDomain)
        {
            var shape = new List<object>();
            foreach (int s in Shape) shape.Add((long)s);
            return new Dictionary<string, object>
            {
                ["sensor_name"] = SensorName, ["modality"] = Modality, ["frame_id"] = FrameId,
                ["clock_domain"] = clockDomain, ["acquisition_time_ns"] = AcquisitionNs, ["delivery_time_ns"] = DeliveryNs,
                ["sequence_index"] = Sequence, ["health"] = Health.ToString(), ["layout"] = Layout,
                ["encoding"] = Encoding.ToString(), ["shape"] = shape, ["units"] = Units,
                ["payload_b64"] = Convert.ToBase64String(Payload), ["payload_digest"] = Sha256Hex(Payload),
                ["calibration_ref"] = CalibrationRef, ["context"] = Context,
            };
        }
    }

    /// <summary>Per-sensor fault/degradation state; recomputed from the fault schedule every step.</summary>
    public sealed class SensorFaultState
    {
        public bool Dropout, Failed;
        public double ExtraBias, NoiseScale = 1.0, Fouling; // fouling 0..1 (camera)
        public void Clear() { Dropout = false; Failed = false; ExtraBias = 0; NoiseScale = 1.0; Fouling = 0; }
    }

    public interface ISimSensor
    {
        string Name { get; }
        string Modality { get; }
        double FrequencyHz { get; }
        SensorFaultState Fault { get; }
        /// <summary>Called every physics step; samples when due and queues the packet for delayed delivery.</summary>
        void Tick(TruthSnapshot truth, SimClock clock);
        /// <summary>Packets whose delivery time has been reached, oldest first.</summary>
        List<SensorPacketData> Deliver(long nowNs);
        int Backlog { get; }
        long DroppedFrames { get; }
        void Reset();
    }

    public abstract class SimSensorBase : ISimSensor
    {
        protected readonly SensorParameters P;
        protected readonly SeededRandom Rng;
        private readonly Queue<SensorPacketData> _inFlight = new Queue<SensorPacketData>();
        private readonly int _maxInFlight;
        private long _nextSampleNs;
        private long _seq;
        protected long PeriodNs { get; }
        protected long LatencyNs { get; }

        public string Name => P.Name;
        public abstract string Modality { get; }
        public double FrequencyHz => P.RateHz;
        public SensorFaultState Fault { get; } = new SensorFaultState();
        public int Backlog => _inFlight.Count;
        public long DroppedFrames { get; private set; }

        protected SimSensorBase(SensorParameters p, ulong seed, int maxInFlight = 64)
        {
            if (!(p.RateHz > 0)) throw new ArgumentException("sensor " + p.Name + " needs a positive rate");
            P = p;
            Rng = new SeededRandom(seed, "sensor:" + p.Name);
            PeriodNs = (long)Math.Round(SimClock.NsPerSecond / p.RateHz);
            LatencyNs = (long)Math.Round(Math.Max(0, p.LatencyS) * SimClock.NsPerSecond);
            _maxInFlight = maxInFlight;
        }

        /// <summary>Bias + drift + (scaled) white noise for one scalar channel.</summary>
        protected double Corrupt(double truth, double tS)
        {
            return truth + P.Bias + Fault.ExtraBias + P.DriftPerS * tS + Rng.Gaussian(P.NoiseStd * Fault.NoiseScale);
        }

        protected SensorHealth CurrentHealth =>
            Fault.Failed ? SensorHealth.FAULT : (Fault.NoiseScale > 1 || Fault.ExtraBias != 0 || Fault.Fouling > 0)
                ? SensorHealth.DEGRADED : SensorHealth.OK;

        /// <summary>Produce the payload for the current truth; null = no measurement this time.</summary>
        protected abstract SensorPacketData Measure(TruthSnapshot truth, double tS);

        public void Tick(TruthSnapshot truth, SimClock clock)
        {
            if (clock.NowNs < _nextSampleNs) return;
            _nextSampleNs += PeriodNs;
            if (_nextSampleNs <= clock.NowNs) _nextSampleNs = clock.NowNs + PeriodNs; // never burst after a pause
            if (Fault.Failed || Fault.Dropout) { DroppedFrames++; return; }
            SensorPacketData pkt = Measure(truth, clock.Seconds);
            if (pkt == null) { DroppedFrames++; return; }
            pkt.SensorName = P.Name;
            pkt.Modality = Modality;
            pkt.FrameId = P.FrameId;
            pkt.CalibrationRef = P.CalibrationRef;
            pkt.AcquisitionNs = clock.NowNs;
            pkt.DeliveryNs = clock.NowNs + LatencyNs;
            pkt.Sequence = ++_seq;
            pkt.Health = CurrentHealth;
            if (_inFlight.Count >= _maxInFlight) { _inFlight.Dequeue(); DroppedFrames++; }
            _inFlight.Enqueue(pkt);
        }

        public List<SensorPacketData> Deliver(long nowNs)
        {
            var ready = new List<SensorPacketData>();
            while (_inFlight.Count > 0 && _inFlight.Peek().DeliveryNs <= nowNs) ready.Add(_inFlight.Dequeue());
            return ready;
        }

        public virtual void Reset()
        {
            _inFlight.Clear();
            _nextSampleNs = 0;
            _seq = 0;
            DroppedFrames = 0;
            Fault.Clear();
        }

        protected static byte[] Pack(double[] values)
        {
            if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("wire payloads are little-endian");
            var b = new byte[values.Length * 8];
            Buffer.BlockCopy(values, 0, b, 0, b.Length);
            return b;
        }

        protected static byte[] Pack(float[] values)
        {
            if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("wire payloads are little-endian");
            var b = new byte[values.Length * 4];
            Buffer.BlockCopy(values, 0, b, 0, b.Length);
            return b;
        }
    }
}
