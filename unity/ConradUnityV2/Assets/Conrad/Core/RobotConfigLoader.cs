// Loads the RobotConfig document written by conrad.sim.unity.robot_export (format conrad.unity.robot_config.v1).
// It REFUSES any OPEN physical parameter: Unity never invents a physical fact. Values stay in the Conrad body
// convention (+X forward, +Y left, +Z up, SI) and are converted at the engine boundary by ConradFrames.
using System;
using System.Collections.Generic;

namespace Conrad.UnityV2.Core
{
    public sealed class OpenParameterException : Exception
    {
        public OpenParameterException(string message) : base(message) { }
    }

    public enum SourceKind { MEASURED, IDENTIFIED, LITERATURE_PRIOR, ENGINEERING_ESTIMATE, SYNTHETIC_ONLY, OPEN }

    public enum SimulationValidityLevel
    {
        L0_FUNCTIONAL, L1_APPROXIMATE_PHYSICS, L2_CHARACTERIZED, L3_IDENTIFIED, L4_VALIDATED_ENVELOPE
    }

    public sealed class SourcedValue
    {
        public string Path;
        public SourceKind Source;
        public string Units;
        public double? Sigma;
        public object Raw; // double, List<double> or string
    }

    public sealed class ThrusterParameters
    {
        public string Id;
        public Vec3d PositionBody;
        public Vec3d DirectionBody;
        public double MaxForwardN, MaxReverseN, Deadzone, TimeConstantS, LatencyS, K;
    }

    public sealed class SensorParameters
    {
        public string Name, Modality, FrameId, CalibrationRef;
        public Vec3d MountPositionBody;
        public Quatd MountRotationBody;
        public double RateHz, NoiseStd, Bias, DriftPerS, LatencyS;
        public Dictionary<string, object> Extra = new Dictionary<string, object>();
    }

    public sealed class RobotParameters
    {
        public const string Format = "conrad.unity.robot_config.v1";

        public string RobotId, ConfigName, ConfigVersion, Digest;
        public double MassKg, DisplacedVolumeM3;
        public Vec3d CenterOfMassBody, CenterOfBuoyancyBody, InertiaDiag, Dimensions;
        public double[] LinearDrag = new double[6], QuadraticDrag = new double[6], AddedMass = new double[6];
        public List<ThrusterParameters> Thrusters = new List<ThrusterParameters>();
        public List<SensorParameters> Sensors = new List<SensorParameters>();
        public double BatteryCapacityJ, BatteryNominalVoltageV;
        public readonly List<SourcedValue> Values = new List<SourcedValue>();

        private static readonly string[] MassPaths =
            { "mass_kg", "displaced_volume_m3", "center_of_mass_body_m", "center_of_buoyancy_body_m", "inertia_diag_kgm2" };
        private static readonly string[] DynamicsPaths = { "linear_drag", "quadratic_drag", "added_mass_diag" };

        /// <summary>Same rule as conrad.sim.unity.assess_validity_level. Unity alone never reaches L4.</summary>
        public SimulationValidityLevel AssessValidity(bool physicsEnabled, string validationReportRef)
        {
            if (!physicsEnabled) return SimulationValidityLevel.L0_FUNCTIONAL;
            bool Grounded(SourceKind s) => s == SourceKind.MEASURED || s == SourceKind.IDENTIFIED;
            bool massOk = true, thrOk = Thrusters.Count > 0, dynOk = true;
            foreach (var v in Values)
            {
                if (Array.IndexOf(MassPaths, v.Path) >= 0 && !Grounded(v.Source)) massOk = false;
                if (Array.IndexOf(DynamicsPaths, v.Path) >= 0 && v.Source != SourceKind.IDENTIFIED) dynOk = false;
                if (v.Path.StartsWith("thrusters[", StringComparison.Ordinal) &&
                    (v.Path.EndsWith(".thrust_coefficient", StringComparison.Ordinal) ||
                     v.Path.EndsWith(".time_constant_s", StringComparison.Ordinal) ||
                     v.Path.EndsWith(".deadzone_command", StringComparison.Ordinal) ||
                     v.Path.EndsWith(".position_body_m", StringComparison.Ordinal)) && !Grounded(v.Source))
                    thrOk = false;
            }
            if (!(massOk && thrOk)) return SimulationValidityLevel.L1_APPROXIMATE_PHYSICS;
            if (!dynOk) return SimulationValidityLevel.L2_CHARACTERIZED;
            return string.IsNullOrEmpty(validationReportRef)
                ? SimulationValidityLevel.L3_IDENTIFIED
                : SimulationValidityLevel.L4_VALIDATED_ENVELOPE;
        }
    }

    public static class RobotConfigLoader
    {
        public static RobotParameters LoadJson(string json)
        {
            var doc = MiniJson.ParseObject(json);
            if (J.Str(doc, "format") != RobotParameters.Format)
                throw new JsonException("unsupported robot config format " + J.Str(doc, "format"));
            if (J.Str(doc, "body_convention") != ConradFrames.BodyConvention)
                throw new JsonException("robot config body convention is not " + ConradFrames.BodyConvention);
            var p = new RobotParameters
            {
                RobotId = J.Str(doc, "robot_id"),
                ConfigName = J.Str(doc, "config_name"),
                ConfigVersion = J.Str(doc, "config_version"),
                Digest = J.Str(doc, "robot_config_digest"),
            };
            var open = new List<string>();
            p.MassKg = Scalar(p, doc, "mass_kg", open);
            p.DisplacedVolumeM3 = Scalar(p, doc, "displaced_volume_m3", open);
            p.CenterOfMassBody = Vector(p, doc, "center_of_mass_body_m", open);
            p.CenterOfBuoyancyBody = Vector(p, doc, "center_of_buoyancy_body_m", open);
            p.InertiaDiag = Vector(p, doc, "inertia_diag_kgm2", open);
            p.Dimensions = Vector(p, doc, "dimensions_m", open);
            p.LinearDrag = Array6(p, doc, "linear_drag", open);
            p.QuadraticDrag = Array6(p, doc, "quadratic_drag", open);
            p.AddedMass = Array6(p, doc, "added_mass_diag", open);
            foreach (object o in J.Arr(doc, "thrusters"))
            {
                var t = (Dictionary<string, object>)o;
                string id = J.Str(t, "thruster_id");
                string pre = "thrusters[" + id + "]";
                p.Thrusters.Add(new ThrusterParameters
                {
                    Id = id,
                    PositionBody = Vector(p, t, "position_body_m", open, pre),
                    DirectionBody = Vector(p, t, "direction_body", open, pre),
                    MaxForwardN = Scalar(p, t, "max_forward_thrust_n", open, pre),
                    MaxReverseN = Scalar(p, t, "max_reverse_thrust_n", open, pre),
                    Deadzone = Scalar(p, t, "deadzone_command", open, pre),
                    TimeConstantS = Scalar(p, t, "time_constant_s", open, pre),
                    LatencyS = Scalar(p, t, "latency_s", open, pre),
                    K = Scalar(p, t, "thrust_coefficient", open, pre),
                });
            }
            foreach (object o in J.Arr(doc, "sensors"))
            {
                var s = (Dictionary<string, object>)o;
                string name = J.Str(s, "sensor_name");
                string pre = "sensors[" + name + "]";
                double[] mount = Numbers(p, s, "mount_pose_body", open, pre, 7);
                var sp = new SensorParameters
                {
                    Name = name,
                    Modality = J.Str(s, "modality"),
                    FrameId = J.Str(s, "frame_id"),
                    CalibrationRef = J.StrOrNull(s, "calibration_ref"),
                    RateHz = Scalar(p, s, "rate_hz", open, pre),
                    NoiseStd = Scalar(p, s, "noise_std", open, pre),
                    Bias = Scalar(p, s, "bias", open, pre),
                    DriftPerS = Scalar(p, s, "drift_per_s", open, pre),
                    LatencyS = Scalar(p, s, "latency_s", open, pre),
                };
                if (mount != null)
                {
                    sp.MountPositionBody = new Vec3d(mount[0], mount[1], mount[2]);
                    sp.MountRotationBody = new Quatd(mount[3], mount[4], mount[5], mount[6]).Normalized;
                }
                if (s.TryGetValue("parameters", out object extra) && extra is Dictionary<string, object> ed) sp.Extra = ed;
                p.Sensors.Add(sp);
            }
            var battery = J.Obj(doc, "battery");
            p.BatteryCapacityJ = Scalar(p, battery, "capacity_j", open, "battery");
            p.BatteryNominalVoltageV = Scalar(p, battery, "nominal_voltage_v", open, "battery");
            if (open.Count > 0)
                throw new OpenParameterException("Unity refuses OPEN physical parameters: " + string.Join(", ", open));
            if (p.MassKg <= 0 || p.DisplacedVolumeM3 <= 0) throw new JsonException("mass and displaced volume must be positive");
            if (p.Thrusters.Count == 0) throw new OpenParameterException("thruster layout is empty");
            return p;
        }

        private static Dictionary<string, object> Field(RobotParameters p, Dictionary<string, object> o, string key,
                                                        List<string> open, string prefix, out SourcedValue sv)
        {
            string path = prefix == null ? key : prefix + "." + key;
            var f = J.Obj(o, key);
            sv = new SourcedValue
            {
                Path = path,
                Units = J.Str(f, "units"),
                Source = (SourceKind)Enum.Parse(typeof(SourceKind), J.Str(f, "source")),
                Sigma = J.Has(f, "sigma") ? J.Num(f, "sigma") : (double?)null,
            };
            if (sv.Source == SourceKind.OPEN || !J.Has(f, "value"))
            {
                open.Add(path);
                return null;
            }
            sv.Raw = f["value"];
            p.Values.Add(sv);
            return f;
        }

        private static double Scalar(RobotParameters p, Dictionary<string, object> o, string key, List<string> open,
                                     string prefix = null)
        {
            var f = Field(p, o, key, open, prefix, out _);
            return f == null ? double.NaN : J.Num(f["value"], key);
        }

        private static double[] Numbers(RobotParameters p, Dictionary<string, object> o, string key, List<string> open,
                                        string prefix, int n)
        {
            var f = Field(p, o, key, open, prefix, out _);
            if (f == null) return null;
            var list = f["value"] as List<object>;
            if (list == null || list.Count != n) throw new JsonException(key + " must have " + n + " components");
            var a = new double[n];
            for (int i = 0; i < n; i++) a[i] = J.Num(list[i], key);
            return a;
        }

        private static Vec3d Vector(RobotParameters p, Dictionary<string, object> o, string key, List<string> open,
                                    string prefix = null)
        {
            double[] a = Numbers(p, o, key, open, prefix, 3);
            return a == null ? new Vec3d(double.NaN, double.NaN, double.NaN) : new Vec3d(a[0], a[1], a[2]);
        }

        private static double[] Array6(RobotParameters p, Dictionary<string, object> o, string key, List<string> open)
        {
            return Numbers(p, o, key, open, null, 6) ?? new double[6];
        }
    }
}
