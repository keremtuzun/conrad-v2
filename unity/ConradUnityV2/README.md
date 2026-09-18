# Conrad Unity V2 (simulator source tree)

Unity V2 is the vehicle-physics, sensor-viewpoint and mission-execution simulator of Conrad (spec ch20/ch21).
It owns robot physics, robot/environment interaction, sensor rendering and mission execution. It consumes the
shared scenario and never evolves twin (2T/2E/2S) truth.

> **Status: source only, never compiled or executed.** The Unity editor could not run in the environment this
> tree was written in. Every C# file was written against the Unity 2022.3 LTS API and checked by
> `tests/contract/unity/test_csharp_python_parity.py`, which reads the C# source. That test checks wire keys,
> layouts, the fault vocabulary, format strings, and the frame-conversion formulas (evaluated numerically
> against the Python mapper). Nothing has been compiled, played, or measured. Unity execution is an external
> blocker (`EXT-UNITY-01`).

## Layout

```
Assets/Conrad/
  ConradUnityV2.asmdef                    core assembly, no third-party dependencies
  Core/                                   MiniJson, Vec3d/Quatd, ConradFrames (axis map), SeededRandom, SimClock,
                                          RobotConfigLoader (refuses OPEN parameters), validity levels
  VehicleDynamics/VehicleDynamics6Dof.cs  M nu_dot + C nu + D nu + g = tau + tau_env, applied as Rigidbody accelerations
  Hydrodynamics/HydrodynamicsModel.cs     linear+quadratic damping, added mass (+ Coriolis), buoyancy/weight at CoB/CoG
  Propulsion/ThrusterModel.cs             T = k u|u|, deadzone, saturation, asymmetry, latency, first-order lag, noise, faults
  EnvironmentInteraction/                 water density, surface, turbidity, current fields (constant / depth profile / time-varying)
  SensorSimulation/                       ISimSensor + SimSensorBase (acquisition vs delivery time), IMU, depth, camera, sonar
  RobotHardwareSimulation/                RHI server side: command validation, power model, health, state packets
  MissionExperimentRuntime/               ExperimentRuntime (fixed integer-ns stepping, seeds), shared scenario loader,
                                          fault injector, JSONL replay logger
  ExternalInterfaces/Protocol/            BridgeProtocolHandler: transport-independent wire protocol v1 (CONTROL + TRUTH roles)
  ExternalInterfaces/Transport/           ZmqBridgeServer (NetMQ REQ/REP + PUB), own asmdef referencing NetMQ DLLs
Assets/Plugins/NetMQ/README.md            which NetMQ DLLs to drop in (not vendored)
Examples/experiment.example.json          experiment config (format conrad.unity.experiment.v1)
```

## How the physics is applied

`VehicleDynamics6Dof` evaluates the full 6-DOF model in double precision in the Conrad body frame (+X forward,
+Y left, +Z up). It hands the resulting linear and angular accelerations to the Rigidbody with
`ForceMode.Acceleration`. PhysX only integrates and resolves contacts. `Rigidbody.drag`, `angularDrag` and
gravity are disabled, so none of Unity's built-in damping leaks in. Anisotropic added mass is therefore honoured,
which a scalar `Rigidbody.mass` cannot do. Physics runs in `SimulationMode.Script`: the runtime calls
`Physics.Simulate(dt)` itself, driven either by the bridge (lock-step) or by `FixedUpdate` (free-running).

Documented L1 assumptions: dynamics about the body origin, diagonal inertia and added mass, and CoG offset
entering only through restoring moments. The same assumptions are made in `conrad.sim.kernel`.

## Opening and running (once a machine with Unity is available)

1. Install Unity **2022.3 LTS** (Hub → Installs). Add this folder (`unity/ConradUnityV2`) as a project.
2. Copy the NetMQ DLLs as described in `Assets/Plugins/NetMQ/README.md`.
3. Scene setup: create a `Vehicle` GameObject with a `Rigidbody`, a collider approximating the hull,
   `VehicleDynamics6Dof`, `ExperimentRuntime` and `ZmqBridgeServer`. For every RGB sensor in the RobotConfig,
   add a child `Camera` and bind it in `ExperimentRuntime.cameras` by sensor name. For every sonar, add a
   child `Transform` mount and bind it in `sonarMounts`. Add static colliders for the scenario geometry (see
   "open items").
4. Produce the inputs from the Python side (repo root):
   ```
   python -m uv run python -c "from conrad.robotics.hardware.config import load_robot_config; from conrad.sim.unity import write_unity_robot_config; print(write_unity_robot_config(load_robot_config('configs/robot/sim_reference.yaml'), 'artifacts/unity/robot_config.json'))"
   ```
   The scenario file is a `conrad.schemas.world.Scenario` dumped with `model_dump(mode="json")`. Its
   `environment` must contain `water_density_kgm3`; see `docs/UNITY.md` for the full schema.
5. Build a player (or press Play) and start it with
   `ConradSim -conradRobotConfig artifacts/unity/robot_config.json -conradScenario scenario.json -conradExperiment Examples/experiment.example.json`.
   Do **not** pass `-nographics` when cameras are enabled, because cameras need a graphics device.
6. Connect from Python with `conrad.adapters.unity.UnityRobotHardware` using `configs/sim/unity_lockstep.yaml`.

The simulator reports its validity level in the handshake. With `configs/robot/sim_reference.yaml`, where every
value is SYNTHETIC_ONLY, it reports **L1_APPROXIMATE_PHYSICS**. L4 requires identified parameters plus a held-out
validation report, and Unity alone never claims it.

## Development sequence status (spec ch21)

| Stage | Content | Status in this tree |
|---|---|---|
| V2.0 | 6-DOF, buoyancy, drag, thrusters, simple geometry, ideal sensors, Python interface | Source written; Python client + protocol tested against a mock server. **Not compiled or run in Unity.** |
| V2.1 | currents, sensor noise, IMU/depth, camera, navigation benchmarks | Currents/noise/IMU/depth/camera source written. Navigation benchmarks NAV-001..008 are not part of this workstream. |
| V2.2 | sonar, fault injection, power, domain randomization | Sonar (geometric proxy), 15 of 16 fault types (LOCALIZATION_DEGRADATION refused: no localization sensor in Unity), power model written. Fault schedules are seeded (`conrad.sim.unity.draw_fault_schedule`). **Physical-parameter domain randomization inside Unity is OPEN.** |
| V2.3 | shared scenario, Twin 2S integration, Twin 2T/E hooks | Scenario loader reads seed, WORLD convention, environment, initial pose and mission. **Instantiating Twin 2S geometry and 2T/2E visual hooks is OPEN** (needs a geometry-reference format from the twins workstream). |
| V2.4 | parameter identification, hardware measurements, HIL | Python identification tooling, characterization ingestion and HIL harness implemented and tested on synthetic data. **Real measurements BLOCKED_EXTERNAL (hardware owner).** |
| V2.5 | full integrated Model 1/2, Active Intelligence, BAAC, mission benchmarks | Not started in this workstream. |

## Open items

- `EXT-UNITY-01`: compile, play and profile the project on a machine with Unity 2022.3 LTS. Then run the Python
  client against the real player instead of the mock server.
- Scenario geometry instantiation (Twin 2S → colliders/meshes) and 2T/2E appearance hooks.
- Physical-parameter domain randomization and structured randomization inside Unity.
- Camera noise units: `noise_std` is interpreted as a fraction of full scale for RGB (documented, not measured).
