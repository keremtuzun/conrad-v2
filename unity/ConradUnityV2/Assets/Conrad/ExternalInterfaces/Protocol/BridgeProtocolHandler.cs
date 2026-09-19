// Transport-independent server side of the Conrad <-> Unity wire protocol v1 (conrad/adapters/unity/protocol.py).
// Envelope: {protocol_version, schema_version, kind, session_id, seq, body}. Every request gets exactly one reply
// with the same seq. Anything malformed, incompatible or out of role gets an ERROR reply and changes nothing.
// Two instances exist: role CONTROL (RobotHardwareInterface surface, never truth) and role TRUTH (ground truth,
// only when the experiment enables it for training/evaluation).
using System;
using System.Collections.Generic;
using System.Text;
using Conrad.UnityV2.Core;
using Conrad.UnityV2.MissionExperimentRuntime;

namespace Conrad.UnityV2.ExternalInterfaces
{
    public enum EndpointRole { CONTROL, TRUTH }

    public sealed class BridgeProtocolHandler
    {
        public const string ProtocolVersion = "1.0.0";
        public const string SchemaVersion = "1.0.0";
        public const string SimulatorId = "conrad-unity-v2";
        public const string SimulatorVersion = "2.0.0";

        private readonly ExperimentRuntime _rt;
        private readonly EndpointRole _role;
        private string _sessionId = "";
        private int _sessionCounter;
        private long _expectedStep;

        public BridgeProtocolHandler(ExperimentRuntime runtime, EndpointRole role)
        {
            _rt = runtime ?? throw new ArgumentNullException(nameof(runtime));
            _role = role;
        }

        public bool HasSession => _sessionId.Length > 0;
        public string SessionId => _sessionId;

        private static bool SameMajor(string version, string expected)
        {
            int a = version.IndexOf('.'), b = expected.IndexOf('.');
            return a > 0 && b > 0 && string.CompareOrdinal(version, 0, expected, 0, Math.Max(a, b)) == 0 && a == b;
        }

        public static byte[] Encode(string kind, Dictionary<string, object> body, string sessionId, long seq)
        {
            var env = new Dictionary<string, object>
            {
                ["protocol_version"] = ProtocolVersion, ["schema_version"] = SchemaVersion, ["kind"] = kind,
                ["session_id"] = sessionId, ["seq"] = seq, ["body"] = body,
            };
            return Encoding.UTF8.GetBytes(MiniJson.Serialize(env));
        }

        private byte[] Error(string code, string detail, long seq) =>
            Encode("ERROR", new Dictionary<string, object> { ["code"] = code, ["detail"] = detail }, _sessionId, seq);

        public byte[] Handle(byte[] request)
        {
            long seq = 0;
            try
            {
                var env = MiniJson.ParseObject(Encoding.UTF8.GetString(request));
                seq = J.Long(env, "seq");
                if (!SameMajor(J.Str(env, "protocol_version"), ProtocolVersion)) return Error("PROTOCOL_VERSION", "incompatible protocol major", seq);
                if (!SameMajor(J.Str(env, "schema_version"), SchemaVersion)) return Error("SCHEMA_VERSION", "incompatible schema major", seq);
                string kind = J.Str(env, "kind");
                var body = J.Obj(env, "body");
                if (kind == "HANDSHAKE") return Handshake(body, seq);
                if (!HasSession || J.Str(env, "session_id") != _sessionId) return Error("NO_SESSION", "handshake first", seq);
                if (_role == EndpointRole.TRUTH)
                {
                    if (kind != "GET_GROUND_TRUTH") return Error("ROLE_FORBIDDEN", kind + " is not served on the truth endpoint", seq);
                    return Encode("GROUND_TRUTH", _rt.CurrentTruthWire(), _sessionId, seq);
                }
                switch (kind)
                {
                    case "RESET": return Reset(body, seq);
                    case "STEP": return Step(body, seq);
                    case "POLL": return Encode("STATE", _rt.Server.BuildState(null, _rt.Clock.Domain), _sessionId, seq);
                    case "COMMAND":
                    {
                        var ack = _rt.Server.HandleCommand(body, _rt.Clock.Domain);
                        _rt.LogEvent("command", new Dictionary<string, object> { ["request"] = body, ["ack"] = ack });
                        return Encode("COMMAND_ACK", ack, _sessionId, seq);
                    }
                    case "INJECT_FAULT": return InjectFault(body, seq);
                    case "GET_METRICS": return Metrics(seq);
                    case "CONFIGURE_SCENE": return ConfigureScene(body, seq);
                    case "FRAME_PROBE": return Encode("FRAME_PROBE_ACK", FrameProbe.Run(body), _sessionId, seq);
                    default: return Error("UNSUPPORTED", kind + " is not served on the control endpoint", seq);
                }
            }
            catch (Exception ex) when (ex is JsonException || ex is InvalidCastException || ex is FormatException || ex is ArgumentException
                                       || ex is KeyNotFoundException || ex is OverflowException)
            {
                return Error("BAD_REQUEST", ex.Message, seq);
            }
        }

        private byte[] Handshake(Dictionary<string, object> b, long seq)
        {
            string role = J.Str(b, "role");
            if (role != _role.ToString()) return Error("ROLE_FORBIDDEN", "this endpoint serves role " + _role, seq);
            if (_role == EndpointRole.TRUTH && !_rt.TruthEndpointEnabled)
                return Error("TRUTH_DISABLED", "ground truth is disabled for this experiment", seq);
            if (J.Str(b, "robot_config_digest") != _rt.Robot.Digest) return Error("CONFIG_DIGEST", "robot_config_digest mismatch", seq);
            if (J.Str(b, "clock_domain") != _rt.Clock.Domain) return Error("CLOCK_DOMAIN", "clock domain mismatch", seq);
            if (J.Str(b, "frame_convention") != ConradFrames.WireConvention) return Error("FRAME_CONVENTION", "unsupported wire convention", seq);
            if (_role == EndpointRole.CONTROL && J.Bool(b, "lock_step") != _rt.LockStep) return Error("STEP_MODE", "lock_step mismatch", seq);
            _sessionId = "unity-" + _rt.Seed + "-" + (++_sessionCounter);
            _expectedStep = 0;
            var ack = new Dictionary<string, object>
            {
                ["simulator_id"] = SimulatorId, ["simulator_version"] = SimulatorVersion,
                ["protocol_version"] = ProtocolVersion, ["schema_version"] = SchemaVersion,
                ["robot_config_digest"] = _rt.Robot.Digest, ["clock_domain"] = _rt.Clock.Domain,
                ["frame_convention"] = ConradFrames.WireConvention, ["lock_step"] = _rt.LockStep,
                ["session_id"] = _sessionId, ["nonce"] = J.Str(b, "nonce"), ["validity_level"] = _rt.ValidityLevel.ToString(),
                ["physics_dt_ns"] = _rt.PhysicsDtNs, ["sim_time_ns"] = _rt.Clock.NowNs,
                ["capabilities"] = _rt.Server.Capabilities(),
            };
            return Encode("HANDSHAKE_ACK", ack, _sessionId, seq);
        }

        private byte[] Reset(Dictionary<string, object> b, long seq)
        {
            long seed = J.Long(b, "seed");
            string scenarioRef = J.StrOrNull(b, "scenario_ref");
            string digest = _rt.ResetWorld((ulong)seed, scenarioRef);
            string expected = J.StrOrNull(b, "scenario_digest");
            if (expected != null && expected != digest) return Error("SCENARIO_DIGEST", "scenario file digest mismatch", seq);
            _expectedStep = 0;
            return Encode("RESET_ACK", new Dictionary<string, object>
            {
                ["seed"] = seed, ["sim_time_ns"] = _rt.Clock.NowNs, ["scenario_digest"] = digest,
            }, _sessionId, seq);
        }

        private byte[] Step(Dictionary<string, object> b, long seq)
        {
            if (!_rt.LockStep) return Error("STEP_MODE", "STEP is only valid in lock-step mode", seq);
            long index = J.Long(b, "step_index");
            long dtNs = J.Long(b, "dt_ns");
            if (index != _expectedStep) return Error("STEP_INDEX", "expected step " + _expectedStep, seq);
            if (dtNs <= 0 || dtNs % _rt.PhysicsDtNs != 0)
                return Error("STEP_DT", "dt_ns must be a positive multiple of physics_dt_ns=" + _rt.PhysicsDtNs, seq);
            for (long k = 0; k < dtNs / _rt.PhysicsDtNs; k++) _rt.StepOnce();
            _expectedStep++;
            return Encode("STATE", _rt.Server.BuildState(index, _rt.Clock.Domain), _sessionId, seq);
        }

        private byte[] ConfigureScene(Dictionary<string, object> b, long seq)
        {
            int count = _rt.ConfigureScene(b);
            return Encode("SCENE_ACK", new Dictionary<string, object>
            {
                ["scene_digest"] = _rt.SceneDigest, ["primitive_count"] = (long)count, ["sim_time_ns"] = _rt.Clock.NowNs,
            }, _sessionId, seq);
        }

        private byte[] InjectFault(Dictionary<string, object> b, long seq)
        {
            var spec = FaultSpec.FromWire(b);
            var reasons = _rt.Faults.TrySchedule(spec, _rt.Clock.NowNs);
            _rt.LogEvent("fault", new Dictionary<string, object> { ["request"] = b, ["reasons"] = reasons });
            return Encode("FAULT_ACK", new Dictionary<string, object>
            {
                ["fault_id"] = spec.FaultId, ["accepted"] = reasons.Count == 0, ["scheduled_time_ns"] = spec.StartNs,
                ["reason_codes"] = reasons,
            }, _sessionId, seq);
        }

        private byte[] Metrics(long seq)
        {
            var m = new Dictionary<string, object>
            {
                ["steps"] = (double)_rt.StepIndex, ["battery_remaining_fraction"] = _rt.Power.RemainingFraction,
                ["power_w"] = _rt.Power.LastPowerW, ["dropped_frames"] = (double)_rt.Server.DroppedFrames,
            };
            return Encode("METRICS", new Dictionary<string, object>
            {
                ["sim_time_ns"] = _rt.Clock.NowNs, ["validity_level"] = _rt.ValidityLevel.ToString(), ["metrics"] = m,
            }, _sessionId, seq);
        }
    }
}
