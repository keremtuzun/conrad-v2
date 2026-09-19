// Robot Hardware Simulation: the SERVER side of RobotHardwareInterface (ch20 Physical software swap).
// Python's UnityRobotHardware talks to this through the bridge. It validates every command (config digest,
// clock domain, deadline, exact thruster set, finite values in [-1, 1], safety authorization) exactly like a
// physical driver would, and it never exposes truth: state packets carry only sensor packets, thruster state,
// power and health. Power: P_total = P_thrusters + P_compute + P_sensors + P_comms;  E(t+dt) = E(t) - P dt.
using System;
using System.Collections.Generic;
using Conrad.UnityV2.Core;
using Conrad.UnityV2.Propulsion;
using Conrad.UnityV2.SensorSimulation;

namespace Conrad.UnityV2.RobotHardwareSimulation
{
    public sealed class PowerModel
    {
        public double CapacityJ { get; }
        public double NominalVoltageV { get; }
        public double EnergyUsedJ { get; private set; }
        public double LastPowerW { get; private set; }
        public double CapacityFactor = 1.0; // < 1 under BATTERY_DEGRADATION
        private readonly double _computeW, _sensorsW, _commsW, _thrusterCoefficient;

        /// <summary>Power terms come from the experiment config and are labelled SYNTHETIC_ONLY until bench curves exist.</summary>
        public PowerModel(double capacityJ, double nominalVoltageV, double computeW, double sensorsW, double commsW,
                          double thrusterCoefficient)
        {
            if (!(capacityJ > 0)) throw new ArgumentException("battery capacity must be positive");
            CapacityJ = capacityJ; NominalVoltageV = nominalVoltageV;
            _computeW = computeW; _sensorsW = sensorsW; _commsW = commsW; _thrusterCoefficient = thrusterCoefficient;
        }

        public void Step(ThrusterBank bank, double dtS)
        {
            LastPowerW = bank.ElectricalPowerW(_thrusterCoefficient) + _computeW + _sensorsW + _commsW;
            EnergyUsedJ += LastPowerW * dtS;
        }

        public double RemainingFraction => Math.Max(0.0, Math.Min(1.0, 1.0 - EnergyUsedJ / (CapacityJ * CapacityFactor)));
        public void Reset() { EnergyUsedJ = 0; LastPowerW = 0; CapacityFactor = 1.0; }
    }

    public sealed class RobotHardwareServer
    {
        public static readonly HashSet<string> MotionPermittedSafetyStates = new HashSet<string> { "NORMAL", "DEGRADED", "HOLD" };
        // Explicit stop commands (every thruster exactly 0) are accepted in these states, as by the Command Gateway.
        public static readonly HashSet<string> ZeroOnlySafetyStates = new HashSet<string> { "EMERGENCY_STOP", "RECOVER", "RETURN" };
        private long? _commandDeadlineNs;   // deadline of the last accepted command (command-timeout watchdog)
        public long WatchdogTrips { get; private set; }

        private readonly RobotParameters _p;
        private readonly ThrusterBank _bank;
        private readonly List<ISimSensor> _sensors;
        private readonly PowerModel _power;
        private readonly SimClock _clock;
        private readonly HashSet<string> _seenCommands = new HashSet<string>();
        private readonly List<SensorPacketData> _undelivered = new List<SensorPacketData>();
        public bool CommLoss;          // COMM_LOSS fault: link drops commands and telemetry
        public bool LeakDetected;      // LEAK_SIGNAL fault
        public readonly List<string> ActiveFaults = new List<string>();
        public long DroppedFrames { get; private set; }

        public RobotHardwareServer(RobotParameters p, ThrusterBank bank, List<ISimSensor> sensors, PowerModel power, SimClock clock)
        {
            _p = p; _bank = bank; _sensors = sensors; _power = power; _clock = clock;
        }

        public Dictionary<string, object> Capabilities()
        {
            var cams = new List<object>(); var sonars = new List<object>(); var ranges = new List<object>();
            bool imu = false, depth = false;
            foreach (var s in _sensors)
            {
                if (s.Modality == "RGB") cams.Add(s.Name);
                else if (s.Modality == "SONAR") sonars.Add(s.Name);
                else if (s.Modality == "DEPTH_RANGE") ranges.Add(s.Name);
                else if (s.Modality == "IMU") imu = true;
                else if (s.Modality == "PRESSURE_DEPTH") depth = true;
            }
            var ids = new List<object>();
            foreach (var t in _bank.Thrusters) ids.Add(t.Parameters.Id);
            return new Dictionary<string, object>
            {
                ["capability_version"] = "unity-v2:" + _p.ConfigVersion, ["cameras"] = cams, ["sonars"] = sonars,
                ["range_imagers"] = ranges,
                ["imu"] = imu, ["depth"] = depth, ["environmental_sensors"] = new List<object>(),
                ["thruster_ids"] = ids, ["controllable_dof"] = new List<object> { "surge", "sway", "heave", "roll", "pitch", "yaw" },
                ["communication_links"] = new List<object> { "unity_bridge" }, ["onboard_compute"] = null,
            };
        }

        /// <summary>Validate and apply a COMMAND body. Returns the COMMAND_ACK body.</summary>
        public Dictionary<string, object> HandleCommand(Dictionary<string, object> body, string clockDomain)
        {
            var reasons = new List<object>();
            string id = J.Str(body, "command_id");
            if (CommLoss) reasons.Add("COMM_LOSS");
            if (J.Str(body, "robot_config_digest") != _p.Digest) reasons.Add("WRONG_ROBOT_CONFIG_DIGEST");
            if (J.Str(body, "clock_domain") != clockDomain) reasons.Add("WRONG_CLOCK_DOMAIN");
            if (J.Long(body, "deadline_ns") <= _clock.NowNs) reasons.Add("EXPIRED_DEADLINE");
            if (J.Long(body, "issued_time_ns") > _clock.NowNs) reasons.Add("ISSUED_IN_FUTURE");
            if (_seenCommands.Contains(id)) reasons.Add("DUPLICATE_COMMAND_ID");
            string state = J.StrOrNull(body, "safety_state");
            var cmds = J.Obj(body, "thruster_commands");
            bool allZero = true;
            foreach (var kv in cmds) if (J.Num(kv.Value, kv.Key) != 0.0) allZero = false;
            bool stateOk = state != null && (MotionPermittedSafetyStates.Contains(state) || (allZero && ZeroOnlySafetyStates.Contains(state)));
            if (J.StrOrNull(body, "safety_authorization_id") == null || !stateOk)
                reasons.Add("MISSING_SAFETY_AUTHORIZATION");
            var expected = new HashSet<string>(_bank.Ids);
            if (!expected.SetEquals(cmds.Keys)) reasons.Add("INVALID_ACTUATOR_SET");
            foreach (var kv in cmds)
            {
                double u = J.Num(kv.Value, kv.Key);
                if (double.IsNaN(u) || double.IsInfinity(u) || u < -1 || u > 1) { reasons.Add("COMMAND_OUT_OF_ENVELOPE"); break; }
            }
            if (reasons.Count == 0)
            {
                _seenCommands.Add(id);
                _commandDeadlineNs = J.Long(body, "deadline_ns");
                foreach (var kv in cmds)
                    if (_bank.TryGet(kv.Key, out Thruster t)) t.Command(_clock.NowNs, J.Num(kv.Value, kv.Key));
            }
            return new Dictionary<string, object>
            {
                ["command_id"] = id, ["accepted"] = reasons.Count == 0, ["reason_codes"] = reasons,
                ["ack_time_ns"] = _clock.NowNs, ["sim_time_ns"] = _clock.NowNs,
            };
        }

        /// <summary>Command-timeout watchdog (same rule as the Python kernel): once the last accepted command's
        /// deadline has passed, every thruster is commanded to zero. Called every physics step.</summary>
        public void Watchdog()
        {
            if (!_commandDeadlineNs.HasValue || _clock.NowNs < _commandDeadlineNs.Value) return;
            foreach (var t in _bank.Thrusters) t.Command(_clock.NowNs, 0.0);
            _commandDeadlineNs = null;
            WatchdogTrips++;
        }

        /// <summary>Collect delivered sensor packets (called every physics step so latency is honoured).</summary>
        public void CollectDeliveries()
        {
            foreach (var s in _sensors) _undelivered.AddRange(s.Deliver(_clock.NowNs));
            const int maxQueued = 512;
            if (_undelivered.Count > maxQueued)
            {
                DroppedFrames += _undelivered.Count - maxQueued;
                _undelivered.RemoveRange(0, _undelivered.Count - maxQueued);
            }
        }

        /// <summary>Free-running PUB stream: hand over (and forget) every delivered packet.</summary>
        public List<SensorPacketData> TakeDeliveredPackets()
        {
            var taken = new List<SensorPacketData>(_undelivered);
            _undelivered.Clear();
            return taken;
        }

        /// <summary>STATE body. Sensor packets are handed over once; COMM_LOSS withholds them (they are lost).</summary>
        public Dictionary<string, object> BuildState(long? stepIndex, string clockDomain)
        {
            var sensors = new List<object>();
            if (CommLoss) DroppedFrames += _undelivered.Count;
            else foreach (var pkt in _undelivered) sensors.Add(pkt.ToWire(clockDomain));
            _undelivered.Clear();
            int backlog = 0; long dropped = DroppedFrames;
            foreach (var s in _sensors) { backlog += s.Backlog; dropped += s.DroppedFrames; }
            var thrusters = new List<object>();
            var devices = new Dictionary<string, object>();
            bool anyFault = false, anyDegraded = false;
            foreach (var t in _bank.Thrusters)
            {
                string h = t.Fault.Failed ? "FAULT" : (t.Fault.Effectiveness < 1 || t.Fault.StuckCommand.HasValue) ? "DEGRADED" : "OK";
                anyFault |= h == "FAULT"; anyDegraded |= h == "DEGRADED";
                devices["thruster:" + t.Parameters.Id] = h;
                thrusters.Add(new Dictionary<string, object>
                {
                    ["thruster_id"] = t.Parameters.Id, ["command"] = t.LastCommand, ["estimated_thrust_n"] = t.ThrustN,
                    ["rpm"] = null, ["current_a"] = null, ["temperature_c"] = null, ["health"] = h,
                });
            }
            foreach (var s in _sensors)
            {
                string h = s.Fault.Failed ? "FAULT" : (s.Fault.Dropout || s.Fault.NoiseScale > 1 || s.Fault.ExtraBias != 0 || s.Fault.Fouling > 0) ? "DEGRADED" : "OK";
                anyDegraded |= h != "OK";
                devices["sensor:" + s.Name] = h;
            }
            string overall = LeakDetected || anyFault ? "FAULT" : (anyDegraded || ActiveFaults.Count > 0) ? "DEGRADED" : "OK";
            var faults = new List<object>();
            foreach (string f in ActiveFaults) faults.Add(f);
            return new Dictionary<string, object>
            {
                ["step_index"] = stepIndex.HasValue ? (object)stepIndex.Value : null,
                ["sim_time_ns"] = _clock.NowNs, ["clock_domain"] = clockDomain, ["sensors"] = sensors,
                ["thrusters"] = thrusters,
                ["battery"] = new Dictionary<string, object>
                {
                    ["time_ns"] = _clock.NowNs, ["remaining_fraction"] = _power.RemainingFraction,
                    ["voltage_v"] = _power.NominalVoltageV,
                    ["current_a"] = _power.NominalVoltageV > 0 ? _power.LastPowerW / _power.NominalVoltageV : (object)null,
                    ["energy_used_j"] = _power.EnergyUsedJ,
                },
                ["health"] = new Dictionary<string, object>
                {
                    ["time_ns"] = _clock.NowNs, ["overall"] = overall, ["leak_detected"] = LeakDetected,
                    ["devices"] = devices, ["faults"] = faults,
                },
                ["queue_backlog"] = (long)backlog, ["dropped_frames"] = dropped,
            };
        }

        public void Reset()
        {
            _seenCommands.Clear();
            _undelivered.Clear();
            _commandDeadlineNs = null;
            WatchdogTrips = 0;
            DroppedFrames = 0;
            CommLoss = false;
            LeakDetected = false;
            ActiveFaults.Clear();
        }
    }
}
