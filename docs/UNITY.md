# Unity V2 bridge

Unity V2 simulates the vehicle: physics, sensors, faults and mission execution. The Conrad stack talks to it
only through `RobotHardwareInterface`, so the same stack runs against Unity, the Python kernel
(`conrad.sim.kernel`) or the physical robot.

Status (2026-09-19): the project compiles under **Unity 6000.5.9f1** with zero compiler errors and zero compiler
warnings, a headless Windows64 player is built by `BuildScript.BuildWindows64Player`, and gate U0 is evidenced by
executed tests (`tests/unity_live`, evidence in `artifacts/gates/U0/`, audit in `docs/audits/U0_EVIDENCE.md`).
Every physical parameter is SYNTHETIC_ONLY, so the simulator reports validity **L1_APPROXIMATE_PHYSICS**.
The former blocker `EXT-UNITY-01` is resolved for V2.0.

## Pieces

| Piece | Where | What it does |
|---|---|---|
| Wire protocol v1 | `conrad/adapters/unity/{vocabulary,messages,sensor_packet,protocol}.py` | pydantic message models, envelope encode/decode, version checks |
| Frame map | `conrad/adapters/unity/frames.py` | Conrad (RH, +X fwd, +Z up) <-> Unity (LH, +Y up, +Z fwd) |
| Client | `conrad/adapters/unity/hardware.py` | `UnityRobotHardware(RobotHardwareInterface)`, fail-closed; `configure_scene`, `frame_probe` |
| Transports | `conrad/adapters/unity/transport.py` | **TCP length-prefixed JSON (default)**, optional ZeroMQ, recording and byte-exact replay |
| Player launcher | `conrad/sim/unity/player.py` | `UnityPlayerSession`: writes inputs, starts the headless player, connects control + truth |
| Scene geometry | `conrad/sim/unity/scene.py` | `SceneGeometry` (box / capsule / heightfield in Conrad WORLD) -> `CONFIGURE_SCENE` |
| Export / experiment / truth | `conrad/sim/unity/` | RobotConfig export (refuses OPEN), ExperimentConfig, fault schedules, validity levels, truth-only client |
| Simulator | `unity/ConradUnityV2/Assets/Conrad/` | eight C# modules + `Editor/BuildScript.cs`, see its README |
| Configs | `configs/sim/unity_lockstep.yaml`, `configs/sim/unity_realtime.yaml` | bridge endpoints and modes |

## Transport

The Unity player serves **length-prefixed JSON frames over TCP** (`System.Net.Sockets`, no third-party DLLs):
every frame is a 4-byte unsigned big-endian payload length followed by one UTF-8 JSON envelope; frames above
64 MiB are refused. The control connection is strict request/reply. Python: `TcpBridgeTransport`; C#:
`TcpBridgeServer` + `LengthPrefixedFraming`.

- Both sides accept only literal loopback/private IPs. Python refuses the endpoint before connecting (and checks
  the connected peer address); C# refuses wildcard binds and drops any connecting peer that is not
  loopback/private.
- A timeout, short read, oversized frame or closed peer closes the Python transport; the adapter faults.
- One control client at a time: a newer control connection replaces an older one and must handshake again.
- The ZeroMQ transport (`ZmqBridgeTransport`, `UnityBridgeConfig.transport="zmq"`) remains only for Python
  peers (the test mock). The Unity side no longer has a ZeroMQ server: NetMQ was never vendored and the
  project must compile without missing DLLs.

## Frames (contract tests in `tests/contract/unity`, live probes in `tests/unity_live`)

Conrad WORLD/ROBOT is right-handed with +X forward, +Y left, +Z up. Unity is left-handed with +X right,
+Y up, +Z forward. With `S` the 3x3 axis map (`det S = -1`):

| Quantity | Conrad -> Unity |
|---|---|
| point, velocity, force, acceleration | `(x, y, z) -> (-y, z, x)` |
| angular velocity, torque (axial) | `(x, y, z) -> (y, -z, -x)` |
| quaternion (w, x, y, z) | `(w, x, y, z) -> (w, y, -z, -x)` |

Conrad yaw-left (+Z) is a negative rotation about Unity +Y. The mapper is built from the scenario's declared
`FrameConvention`; the physical WORLD convention stays OPEN (ADR-0002). `FRAME_PROBE` pushes Conrad poses
through the C# map into a real engine `Transform` and returns what the engine reports.

## Protocol v1

Envelope (UTF-8 JSON, one frame):
`{"protocol_version": "1.0.0", "schema_version": "1.0.0", "kind": ..., "session_id": ..., "seq": n, "body": {...}}`.
A major-version mismatch in either version is a hard error. A reply carries the request's `seq`.

| Request | Reply | Notes |
|---|---|---|
| `HANDSHAKE` | `HANDSHAKE_ACK` | carries `robot_config_digest`, `clock_domain`, `frame_convention`, `lock_step`, `nonce`, `validity_level`, capabilities |
| `RESET` | `RESET_ACK` | seed, optional scenario file + sha256. Every reset rebuilds the seeded subsystems (same seed -> same noise streams) |
| `STEP` | `STATE` | lock-step only; `dt_ns` must be a multiple of `physics_dt_ns`; `step_index` must be consecutive |
| `POLL` | `STATE` | state without advancing time |
| `COMMAND` | `COMMAND_ACK` | built from `AllocatedCommand` (`CommandPacket.from_allocated`) |
| `INJECT_FAULT` | `FAULT_ACK` | 16 fault types; Unity refuses `LOCALIZATION_DEGRADATION` |
| `GET_METRICS` | `METRICS` | |
| `CONFIGURE_SCENE` | `SCENE_ACK` | static colliders in Conrad WORLD (box, capsule, heightfield); all-or-nothing; RESET keeps them |
| `FRAME_PROBE` | `FRAME_PROBE_ACK` | frame-contract diagnostic; reads/changes no world state |
| `SENSOR` (push stream) | | free-running mode only |
| `GET_GROUND_TRUTH` | `GROUND_TRUTH` | **truth endpoint only**, never registered on the control endpoint |

Sensor packets carry `acquisition_time_ns` and `delivery_time_ns` separately. Payloads are little-endian binary
in base64 with a sha256 `payload_digest` over the raw bytes. Layouts are `imu_v1` (10 x f64: accel, gyro,
quaternion or NaN), `depth_v1` (1 x f64, m), `rgb8_hwc_v1` (H x W x 3 u8, top row first) and
`sonar_beam_bin_v1` (beams x bins f32 intensity proxy). Acquisitions lie on the `k * period` grid after reset.

## Fail-closed rules of `UnityRobotHardware`

The adapter enters `FAULTED`, and every later call raises `UnityBridgeError` (a `HardwareUnavailableError`), on
any of the following:

- an endpoint that is not a literal loopback/private IP on `allowed_peers`, or a wildcard address;
- an unknown `simulator_id`, or a nonce that is not echoed;
- a protocol or schema major mismatch;
- a clock-domain, `robot_config_digest`, frame-convention, stepping-mode or thruster-set mismatch;
- a validity level below `minimum_validity_level`;
- a reply that is late (`request_timeout_ms`), malformed, oversized, carries the wrong `seq` or session, reports
  a payload digest or size mismatch, uses the wrong layout, stamps a sensor in another clock domain, or moves
  simulation time backwards;
- a `CONFIGURE_SCENE` that Unity refuses.

`send()` also refuses, without touching the wire, any command with the wrong digest, wrong clock domain, no
matching safety authorization, a thruster set that isn't exact, or values outside [-1, 1]. Only
`conrad.runtime.command_gateway` calls `send()`.

## Commands

```
# compile check (zero `error CS` expected; first open of a fresh clone takes minutes)
"C:/Program Files/Unity/Hub/Editor/6000.5.9f1/Editor/Unity.exe" -batchmode -nographics -quit \
    -projectPath unity/ConradUnityV2 -logFile unity/ConradUnityV2/Logs/compile.log

# build the headless player -> unity/ConradUnityV2/Builds/Win64/ConradSim.exe (gitignored)
"C:/Program Files/Unity/Hub/Editor/6000.5.9f1/Editor/Unity.exe" -batchmode -nographics -quit \
    -projectPath unity/ConradUnityV2 -logFile unity/ConradUnityV2/Logs/u0_build.log \
    -executeMethod Conrad.UnityV2.Editor.BuildScript.BuildWindows64Player

# tests (the live ones skip with a reason when the player is absent)
python -m uv run pytest tests/contract/unity tests/unit/adapters/unity tests/hardware_stub -q
python -m uv run pytest tests/unity_live -v -rA      # gate U0, writes artifacts/gates/U0/u0_summary.json
```

Connecting from Python:

```python
from conrad.sim.unity import UnityPlayerSession, scenario_document, SceneGeometry, BoxPrimitive

scenario = scenario_document(
    ids,
    robot_config,
    seed=7,
    initial_position_m=(0.0, 0.0, -10.0),
    environment={"water_density_kgm3": 1022.2, "surface_z_m": 0.0},
)
with UnityPlayerSession(robot_config, "artifacts/unity/run", scenario) as player:
    hw = player.hardware(ids, mission_id, run_id, payload_store=object_store)
    hw.connect()
    hw.configure_scene(
        SceneGeometry(primitives=(BoxPrimitive(id="b", center_m=(5, 0, -10), size_m=(1, 1, 1)),)).to_json()
    )
    hw.reset(seed=7)
    state = hw.step(hw.physics_dt_ns * 20)  # the experiment runner owns time in lock-step mode
```

## Headless player

`ConradSim.exe -batchmode -logFile <log> -conradRobotConfig <json> -conradScenario <json> -conradExperiment <json>
[-conradParentPid <pid>] [-conradIdleExitS <s>]`. Run it **without** `-nographics`: on this host (AMD Radeon 890M)
the batch-mode player gets a Direct3D11 device and the camera renders (verified by the U0 camera test). With
`-nographics` the camera sensor refuses to start (exit code 3) rather than returning blank images. Exit codes:
3 = inputs refused (e.g. an OPEN RobotConfig parameter), 4 = parent process gone, 5 = idle timeout.

## Scenario fields Unity reads

`seed`, `coordinate_system` (must be RIGHT / +Z / +X / m), `robots[0].initial_pose` (WORLD), `mission` and
`environment`:

```json
{"water_density_kgm3": 998.2, "surface_z_m": 0.0, "turbidity": 0.2,
 "current": {"kind": "depth_profile", "surface_velocity_mps": [0.3, 0, 0], "reference_velocity_mps": [0.1, 0, 0],
             "reference_depth_m": 20.0,
             "time_variation": {"amplitude_mps": [0.05, 0, 0], "period_s": 60.0, "turbulence_sigma_mps": 0.02}}}
```

`water_density_kgm3` is required. Unity has no default, because density is a physical fact of the scenario.

## Scene geometry (`CONFIGURE_SCENE`)

```json
{"frame": "WORLD", "replace": true, "primitives": [
  {"kind": "box", "id": "wall", "center_m": [6, 0, -10], "size_m": [0.5, 1, 3], "orientation_wxyz": [1, 0, 0, 0]},
  {"kind": "capsule", "id": "pipe", "p0_m": [12, -40, -29.7], "p1_m": [12, 40, -29.7], "radius_m": 0.3},
  {"kind": "heightfield", "id": "floor", "origin_m": [-4, -4, -12], "spacing_m": [1, 1], "heights_m": [[0, 0], [0, 0]]}]}
```

Heightfield vertex `(i, j)` is `origin + (i*dx, j*dy, heights[i][j])`. `replace: true` also removes the default
world (seafloor top at z = -30 m, a pipe and an obstacle at x = 60 m). Twin 2S wiring is not part of V2.0.
