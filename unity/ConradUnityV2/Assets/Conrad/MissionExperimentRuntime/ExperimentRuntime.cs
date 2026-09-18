// Mission / Experiment runtime (ch21 Mission runtime, Experiment configuration, Deterministic replay).
// Experiment(config, seed): loads RobotConfig + shared Scenario + ExperimentConfig, builds every subsystem with
// seeded streams, and advances the world in fixed integer-ns steps. Physics runs in SimulationMode.Script:
// the runtime calls Physics.Simulate itself, from the bridge (lock-step) or from FixedUpdate (free-running).
using System;
using System.Collections.Generic;
using System.IO;
using Conrad.UnityV2.Core;
using Conrad.UnityV2.EnvironmentInteraction;
using Conrad.UnityV2.Propulsion;
using Conrad.UnityV2.RobotHardwareSimulation;
using Conrad.UnityV2.SensorSimulation;
using Conrad.UnityV2.VehicleDynamics;
using UnityEngine;

namespace Conrad.UnityV2.MissionExperimentRuntime
{
    [Serializable]
    public sealed class NamedCamera { public string sensorName; public Camera camera; }

    [Serializable]
    public sealed class NamedTransform { public string sensorName; public Transform mount; }

    public sealed class ExperimentRuntime : MonoBehaviour
    {
        public const string ExperimentFormat = "conrad.unity.experiment.v1";

        [Header("Inputs (overridable with -conradRobotConfig / -conradScenario / -conradExperiment)")]
        public string robotConfigPath = "robot_config.json";
        public string scenarioPath = "scenario.json";
        public string experimentPath = "experiment.json";

        [Header("Scene bindings")]
        public VehicleDynamics6Dof vehicle;
        public List<NamedCamera> cameras = new List<NamedCamera>();
        public List<NamedTransform> sonarMounts = new List<NamedTransform>();

        public RobotParameters Robot { get; private set; }
        public ScenarioDefinition Scenario { get; private set; }
        public Dictionary<string, object> Experiment { get; private set; }
        public EnvironmentModel Environment { get; private set; }
        public SimClock Clock { get; private set; }
        public ThrusterBank Thrusters { get; private set; }
        public List<ISimSensor> Sensors { get; private set; }
        public PowerModel Power { get; private set; }
        public RobotHardwareServer Server { get; private set; }
        public FaultInjector Faults { get; private set; }
        public SimulationValidityLevel ValidityLevel { get; private set; }
        public long PhysicsDtNs { get; private set; }
        public bool LockStep { get; private set; }
        public bool TruthEndpointEnabled { get; private set; }
        public ulong Seed { get; private set; }
        public long StepIndex { get; private set; }
        private ReplayLogger _replay;

        private static string Arg(string name, string fallback)
        {
            string[] args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++) if (args[i] == name) return args[i + 1];
            return fallback;
        }

        private void Awake()
        {
            // Any failure here leaves the runtime disabled: the bridge refuses to serve a half-built world.
            Robot = RobotConfigLoader.LoadJson(File.ReadAllText(Arg("-conradRobotConfig", robotConfigPath)));
            Scenario = ScenarioDefinition.LoadFile(Arg("-conradScenario", scenarioPath));
            Experiment = MiniJson.ParseObject(File.ReadAllText(Arg("-conradExperiment", experimentPath)));
            if (J.Str(Experiment, "format") != ExperimentFormat) throw new JsonException("experiment format must be " + ExperimentFormat);
            if (vehicle == null) throw new InvalidOperationException("ExperimentRuntime.vehicle is not assigned");
            Clock = new SimClock(J.Str(Experiment, "clock_domain"));
            PhysicsDtNs = J.Long(Experiment, "physics_dt_ns");
            if (PhysicsDtNs <= 0) throw new JsonException("physics_dt_ns must be positive");
            LockStep = J.Bool(Experiment, "lock_step");
            TruthEndpointEnabled = J.Has(Experiment, "truth_endpoint_enabled") && J.Bool(Experiment, "truth_endpoint_enabled");
            Physics.simulationMode = SimulationMode.Script;
            Time.fixedDeltaTime = PhysicsDtNs / (float)SimClock.NsPerSecond;
            ValidityLevel = Robot.AssessValidity(true, J.StrOrNull(Experiment, "validation_report_ref"));
            long seed = J.Has(Experiment, "seed") ? J.Long(Experiment, "seed") : Scenario.Seed;
            Build((ulong)seed);
        }

        private Dictionary<string, object> RenderingParams(SensorParameters sp)
        {
            var merged = new Dictionary<string, object>(sp.Extra);
            if (Experiment.TryGetValue("sensor_rendering", out object r) && r is Dictionary<string, object> all
                && all.TryGetValue(sp.Name, out object mine) && mine is Dictionary<string, object> d)
                foreach (var kv in d) if (!merged.ContainsKey(kv.Key)) merged[kv.Key] = kv.Value; // RobotConfig wins
            return merged;
        }

        private void Build(ulong seed)
        {
            Seed = seed;
            Environment = EnvironmentModel.FromScenario(Scenario.Environment, seed);
            double thrusterNoise = J.Has(Experiment, "thruster_noise_fraction") ? J.Num(Experiment, "thruster_noise_fraction") : 0.0;
            Thrusters = new ThrusterBank(Robot, seed, thrusterNoise);
            Sensors = new List<ISimSensor>();
            foreach (var sp in Robot.Sensors)
            {
                sp.Extra = RenderingParams(sp);
                switch (sp.Modality)
                {
                    case "IMU": Sensors.Add(new ImuSensor(sp, seed)); break;
                    case "PRESSURE_DEPTH": Sensors.Add(new DepthSensor(sp, seed, Environment.DepthBelowSurface)); break;
                    case "RGB":
                        var cam = cameras.Find(c => c.sensorName == sp.Name);
                        if (cam == null || cam.camera == null) throw new InvalidOperationException("no Camera bound for sensor " + sp.Name);
                        Sensors.Add(new CameraSensor(sp, seed, cam.camera, () => Environment.Turbidity));
                        break;
                    case "SONAR":
                        var mount = sonarMounts.Find(m => m.sensorName == sp.Name);
                        if (mount == null || mount.mount == null) throw new InvalidOperationException("no mount bound for sonar " + sp.Name);
                        Sensors.Add(new SonarSensor(sp, seed, mount.mount));
                        break;
                    default:
                        throw new InvalidOperationException("unsupported sensor modality " + sp.Modality);
                }
            }
            var pw = J.Obj(Experiment, "power");
            Power = new PowerModel(Robot.BatteryCapacityJ, Robot.BatteryNominalVoltageV, J.Num(pw, "compute_w"),
                J.Num(pw, "sensors_w"), J.Num(pw, "comms_w"), J.Num(pw, "thruster_coefficient"));
            Server = new RobotHardwareServer(Robot, Thrusters, Sensors, Power, Clock);
            Faults = new FaultInjector(Thrusters, Sensors, Power, Server);
            vehicle.Configure(Robot, Environment, Thrusters);
            ResetWorld(seed, null);
        }

        /// <summary>Deterministic reset: same seed + same config -> same initial world and same fault schedule.</summary>
        public string ResetWorld(ulong seed, string scenarioFile)
        {
            if (scenarioFile != null || seed != Seed)
            {
                // A new scenario (environment, initial pose) or a new seed rebuilds every seeded subsystem;
                // Build() calls back into ResetWorld(seed, null) with Seed already updated.
                if (scenarioFile != null) Scenario = ScenarioDefinition.LoadFile(scenarioFile);
                Build(seed);
                return Scenario.FileDigest;
            }
            Clock.Reset();
            StepIndex = 0;
            Thrusters.Reset();
            foreach (var s in Sensors) s.Reset();
            Power.Reset();
            Server.Reset();
            Faults.Clear();
            vehicle.TeleportConrad(Scenario.InitialPositionWorld, Scenario.InitialOrientationWorld);
            if (Experiment.TryGetValue("faults", out object fl) && fl is List<object> list)
                foreach (object f in list)
                {
                    var reasons = Faults.TrySchedule(FaultSpec.FromWire((Dictionary<string, object>)f), 0);
                    if (reasons.Count > 0) throw new JsonException("experiment fault rejected: " + string.Join(",", reasons));
                }
            _replay?.Dispose();
            string log = J.StrOrNull(Experiment, "replay_log_path");
            _replay = log == null ? null : new ReplayLogger(log, J.Has(Experiment, "replay_truth_every_n_steps")
                ? (int)J.Long(Experiment, "replay_truth_every_n_steps") : 1);
            _replay?.Write("header", 0, new Dictionary<string, object>
            {
                ["seed"] = (long)Seed, ["robot_config_digest"] = Robot.Digest, ["scenario_digest"] = Scenario.FileDigest,
                ["scenario_version"] = Scenario.ScenarioVersion, ["validity_level"] = ValidityLevel.ToString(),
                ["physics_dt_ns"] = PhysicsDtNs, ["lock_step"] = LockStep, ["experiment"] = Experiment,
            });
            return Scenario.FileDigest;
        }

        public void LogEvent(string kind, Dictionary<string, object> data) => _replay?.Write(kind, Clock.NowNs, data);

        /// <summary>One fixed physics step. The ONLY place simulation time advances.</summary>
        public void StepOnce()
        {
            double dtS = PhysicsDtNs / (double)SimClock.NsPerSecond;
            Faults.Apply(Clock.NowNs);
            Thrusters.Step(Clock.NowNs, dtS);
            vehicle.ApplyDynamics(Clock.Seconds);
            Physics.Simulate((float)dtS);
            Clock.Advance(PhysicsDtNs);
            Power.Step(Thrusters, dtS);
            TruthSnapshot truth = vehicle.Snapshot(Clock.NowNs, Clock.Seconds);
            foreach (var s in Sensors) s.Tick(truth, Clock);
            Server.CollectDeliveries();
            StepIndex++;
            if (_replay != null && _replay.ShouldLogTruth(StepIndex)) _replay.Write("truth", Clock.NowNs, TruthWire(truth));
        }

        private void FixedUpdate()
        {
            if (!LockStep) StepOnce();
        }

        /// <summary>GROUND_TRUTH body in the UNITY wire convention (truth endpoint only).</summary>
        public Dictionary<string, object> TruthWire(TruthSnapshot t)
        {
            Vec3d vWorld = t.WorldFromBody.Rotate(t.LinearVelocityBody);
            Vec3d wWorld = t.WorldFromBody.Rotate(t.AngularVelocityBody);
            var faults = new List<object>();
            foreach (string f in Server.ActiveFaults) faults.Add(f);
            return new Dictionary<string, object>
            {
                ["sim_time_ns"] = t.TimeNs,
                ["position_m"] = V(ConradFrames.PointToUnity(t.PositionWorld)),
                ["orientation_wxyz"] = Q(ConradFrames.QuatToUnity(t.WorldFromBody)),
                ["linear_velocity_mps"] = V(ConradFrames.PointToUnity(vWorld)),
                ["angular_velocity_rps"] = V(ConradFrames.AxialToUnity(wWorld)),
                ["water_current_mps"] = V(ConradFrames.PointToUnity(t.WaterCurrentWorld)),
                ["active_faults"] = faults,
            };
        }

        public Dictionary<string, object> CurrentTruthWire() => TruthWire(vehicle.Snapshot(Clock.NowNs, Clock.Seconds));

        private static List<object> V(Vec3d v) => new List<object> { v.X, v.Y, v.Z };
        private static List<object> Q(Quatd q) => new List<object> { q.W, q.X, q.Y, q.Z };

        private void OnDestroy() => _replay?.Dispose();
    }
}
