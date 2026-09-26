// Fault injection (ch20 Fault injection list, ch21 Thruster failures / Sensor degradation).
// Faults are a timed schedule; every step the effect state is rebuilt from scratch from the faults active at
// that simulation instant, so start/duration/magnitude are exact and replays are deterministic.
// Timing and strength of every fault are recorded by the replay logger.
using System;
using System.Collections.Generic;
using Conrad.UnityV2.Core;
using Conrad.UnityV2.Propulsion;
using Conrad.UnityV2.RobotHardwareSimulation;
using Conrad.UnityV2.SensorSimulation;

namespace Conrad.UnityV2.MissionExperimentRuntime
{
    public sealed class FaultSpec
    {
        public string FaultId, FaultType, Target;
        public long StartNs;
        public long? DurationNs;
        public double Magnitude = 1.0;
        public Dictionary<string, object> Parameters = new Dictionary<string, object>();

        public bool ActiveAt(long nowNs) => nowNs >= StartNs && (!DurationNs.HasValue || nowNs < StartNs + DurationNs.Value);

        public static FaultSpec FromWire(Dictionary<string, object> b) => new FaultSpec
        {
            FaultId = J.Str(b, "fault_id"),
            FaultType = J.Str(b, "fault_type"),
            Target = J.StrOrNull(b, "target"),
            StartNs = J.Long(b, "start_time_ns"),
            DurationNs = J.Has(b, "duration_ns") ? J.Long(b, "duration_ns") : (long?)null,
            Magnitude = J.Has(b, "magnitude") ? J.Num(b, "magnitude") : 1.0,
            Parameters = b.TryGetValue("parameters", out object p) && p is Dictionary<string, object> d ? d : new Dictionary<string, object>(),
        };
    }

    public sealed class FaultInjector
    {
        private static readonly HashSet<string> Supported = new HashSet<string>
        {
            "THRUSTER_FAILURE", "THRUSTER_DEGRADATION", "THRUSTER_STUCK", "THRUSTER_LATENCY", "THRUSTER_INTERMITTENT",
            "IMU_BIAS", "IMU_DROPOUT", "DEPTH_BIAS", "CAMERA_FAILURE", "CAMERA_FOULING", "SONAR_NOISE", "SONAR_FAILURE",
            "COMM_LOSS", "LOW_POWER", "BATTERY_DEGRADATION", "LEAK_SIGNAL",
        };

        private readonly List<FaultSpec> _schedule = new List<FaultSpec>();
        private readonly ThrusterBank _bank;
        private readonly Dictionary<string, ISimSensor> _sensorsByName = new Dictionary<string, ISimSensor>();
        private readonly List<ISimSensor> _sensors;
        private readonly PowerModel _power;
        private readonly RobotHardwareServer _server;
        public IReadOnlyList<FaultSpec> Schedule => _schedule;

        public FaultInjector(ThrusterBank bank, List<ISimSensor> sensors, PowerModel power, RobotHardwareServer server)
        {
            _bank = bank; _sensors = sensors; _power = power; _server = server;
            foreach (var s in sensors) _sensorsByName[s.Name] = s;
        }

        /// <summary>Returns reason codes; empty = accepted. LOCALIZATION_DEGRADATION is refused: Unity has no localization sensor.</summary>
        public List<object> TrySchedule(FaultSpec f, long nowNs)
        {
            var reasons = new List<object>();
            if (!Supported.Contains(f.FaultType)) reasons.Add("UNSUPPORTED_BY_SIMULATOR:" + f.FaultType);
            if (f.FaultType.StartsWith("THRUSTER_", StringComparison.Ordinal) && (f.Target == null || !_bank.TryGet(f.Target, out _)))
                reasons.Add("UNKNOWN_THRUSTER");
            if (f.StartNs < nowNs) f.StartNs = nowNs; // the past cannot be changed
            foreach (var existing in _schedule) if (existing.FaultId == f.FaultId) reasons.Add("DUPLICATE_FAULT_ID");
            if (reasons.Count == 0) _schedule.Add(f);
            return reasons;
        }

        private IEnumerable<ISimSensor> SensorsFor(FaultSpec f, string modality)
        {
            if (f.Target != null)
            {
                if (_sensorsByName.TryGetValue(f.Target, out ISimSensor s)) yield return s;
                yield break;
            }
            foreach (var s in _sensors) if (s.Modality == modality) yield return s;
        }

        public void Apply(long nowNs)
        {
            foreach (var t in _bank.Thrusters)
            {
                t.Fault.Effectiveness = 1.0; t.Fault.StuckCommand = null; t.Fault.ExtraLatencyS = 0; t.Fault.IntermittentDropProbability = 0;
            }
            foreach (var s in _sensors) s.Fault.Clear();
            _power.CapacityFactor = 1.0;
            _server.CommLoss = false;
            _server.LeakDetected = false;
            _server.ActiveFaults.Clear();
            foreach (var f in _schedule)
            {
                if (!f.ActiveAt(nowNs)) continue;
                _server.ActiveFaults.Add(f.FaultType + (f.Target != null ? ":" + f.Target : ""));
                Thruster th = null;
                if (f.Target != null) _bank.TryGet(f.Target, out th);
                switch (f.FaultType)
                {
                    case "THRUSTER_FAILURE": th.Fault.Effectiveness = 0.0; break;
                    case "THRUSTER_DEGRADATION": th.Fault.Effectiveness = Math.Max(0, Math.Min(1, f.Magnitude)); break;
                    case "THRUSTER_STUCK": th.Fault.StuckCommand = Math.Max(-1, Math.Min(1, f.Magnitude)); break;
                    case "THRUSTER_LATENCY": th.Fault.ExtraLatencyS = Math.Max(0, f.Magnitude); break;
                    case "THRUSTER_INTERMITTENT": th.Fault.IntermittentDropProbability = Math.Max(0, Math.Min(1, f.Magnitude)); break;
                    case "IMU_BIAS": foreach (var s in SensorsFor(f, "IMU")) s.Fault.ExtraBias += f.Magnitude; break;
                    case "IMU_DROPOUT": foreach (var s in SensorsFor(f, "IMU")) s.Fault.Dropout = true; break;
                    case "DEPTH_BIAS": foreach (var s in SensorsFor(f, "PRESSURE_DEPTH")) s.Fault.ExtraBias += f.Magnitude; break;
                    case "CAMERA_FAILURE": foreach (var s in SensorsFor(f, "RGB")) s.Fault.Failed = true; break;
                    case "CAMERA_FOULING": foreach (var s in SensorsFor(f, "RGB")) s.Fault.Fouling = Math.Max(0, Math.Min(1, f.Magnitude)); break;
                    case "SONAR_NOISE": foreach (var s in SensorsFor(f, "SONAR")) s.Fault.NoiseScale = Math.Max(1, f.Magnitude); break;
                    case "SONAR_FAILURE": foreach (var s in SensorsFor(f, "SONAR")) s.Fault.Failed = true; break;
                    case "COMM_LOSS": _server.CommLoss = true; break;
                    case "LOW_POWER": _power.SetRemainingFraction(f.Magnitude); break;
                    case "BATTERY_DEGRADATION": _power.CapacityFactor = Math.Max(0.01, Math.Min(1, f.Magnitude)); break;
                    case "LEAK_SIGNAL": _server.LeakDetected = true; break;
                }
            }
        }

        public void Clear() { _schedule.Clear(); }
    }
}
