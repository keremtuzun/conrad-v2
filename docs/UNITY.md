# Unity V2 bridge

Unity V2 simulates the vehicle: physics, sensors, faults and mission execution. The Conrad stack talks to it
only through `RobotHardwareInterface`, so the same stack runs against Unity, the Python kernel
(`conrad.sim.kernel`) or the physical robot. Status: the Python side is implemented and tested over real
ZeroMQ sockets against a mock player. The C# tree in `unity/ConradUnityV2` has **not been compiled or executed**
(external blocker `EXT-UNITY-01`).

## Pieces

| Piece | Where | What it does |
|---|---|---|
| Wire protocol v1 | `conrad/adapters/unity/{vocabulary,messages,sensor_packet,protocol}.py` | pydantic message models, envelope encode/decode, version checks |
| Frame map | `conrad/adapters/unity/frames.py` | Conrad (RH, +X fwd, +Z up) <-> Unity (LH, +Y up, +Z fwd) |
| Client | `conrad/adapters/unity/hardware.py` | `UnityRobotHardware(RobotHardwareInterface)`, fail-closed |
| Transports | `conrad/adapters/unity/transport.py` | ZeroMQ REQ/SUB, recording and byte-exact replay |
| Export / experiment / truth | `conrad/sim/unity/` | RobotConfig export (refuses OPEN), ExperimentConfig, fault schedules, validity levels, truth-only client |
| Simulator | `unity/ConradUnityV2/Assets/Conrad/` | eight C# modules, see its README |
| Configs | `configs/sim/unity_lockstep.yaml`, `configs/sim/unity_realtime.yaml` | bridge endpoints and modes |

## Frames (contract tests in `tests/contract/unity`)

Conrad WORLD/ROBOT is right-handed with +X forward, +Y left, +Z up. Unity is left-handed with +X right,
+Y up, +Z forward. With `S` the 3x3 axis map (`det S = -1`):

| Quantity | Conrad -> Unity |
|---|---|
| point, velocity, force, acceleration | `(x, y, z) -> (-y, z, x)` |
| angular velocity, torque (axial) | `(x, y, z) -> (y, -z, -x)` |
| quaternion (w, x, y, z) | `(w, x, y, z) -> (w, y, -z, -x)` |

Conrad yaw-left (+Z) is a negative rotation about Unity +Y. The mapper is built from the scenario's declared
`FrameConvention`; the physical WORLD convention stays OPEN (ADR-0002).

## Protocol v1

Envelope (UTF-8 JSON, one ZeroMQ frame):
`{"protocol_version": "1.0.0", "schema_version": "1.0.0", "kind": ..., "session_id": ..., "seq": n, "body": {...}}`.
A major-version mismatch in either version is a hard error. A reply carries the request's `seq`.

| Request | Reply | Notes |
|---|---|---|
| `HANDSHAKE` | `HANDSHAKE_ACK` | carries `robot_config_digest`, `clock_domain`, `frame_convention`, `lock_step`, `nonce`, `validity_level`, capabilities |
| `RESET` | `RESET_ACK` | seed, optional scenario file + sha256 |
| `STEP` | `STATE` | lock-step only; `dt_ns` must be a multiple of `physics_dt_ns`; `step_index` must be consecutive |
| `POLL` | `STATE` | state without advancing time |
| `COMMAND` | `COMMAND_ACK` | built from `AllocatedCommand` (`CommandPacket.from_allocated`) |
| `INJECT_FAULT` | `FAULT_ACK` | 16 fault types; Unity refuses `LOCALIZATION_DEGRADATION` |
| `GET_METRICS` | `METRICS` | |
| `SENSOR` (PUB) | | free-running stream |
| `GET_GROUND_TRUTH` | `GROUND_TRUTH` | **truth endpoint only**, never registered on the control endpoint |

Sensor packets carry `acquisition_time_ns` and `delivery_time_ns` separately. Payloads are little-endian binary
in base64 with a sha256 `payload_digest` over the raw bytes. Layouts are `imu_v1` (10 x f64: accel, gyro,
quaternion or NaN), `depth_v1` (1 x f64, m), `rgb8_hwc_v1` (H x W x 3 u8, top row first) and
`sonar_beam_bin_v1` (beams x bins f32 intensity proxy).

## Fail-closed rules of `UnityRobotHardware`

The adapter enters `FAULTED`, and every later call raises `UnityBridgeError` (a `HardwareUnavailableError`), on
any of the following:

- an endpoint that is not a literal loopback/private IP on `allowed_peers`, or a wildcard address;
- an unknown `simulator_id`, or a nonce that is not echoed;
- a protocol or schema major mismatch;
- a clock-domain, `robot_config_digest`, frame-convention, stepping-mode or thruster-set mismatch;
- a validity level below `minimum_validity_level`;
- a reply that is late (`request_timeout_ms`), malformed, carries the wrong `seq` or session, reports a
  payload digest or size mismatch, uses the wrong layout, stamps a sensor in another clock domain, or moves
  simulation time backwards.

`send()` also refuses, without touching the wire, any command with the wrong digest, wrong clock domain, no
matching safety authorization, a thruster set that isn't exact, or values outside [-1, 1]. Only
`conrad.runtime.command_gateway` calls `send()`.

## Commands

```
# export the RobotConfig Unity loads (prints the digest Unity must echo)
python -m uv run python -c "from conrad.robotics.hardware.config import load_robot_config; from conrad.sim.unity import write_unity_robot_config; print(write_unity_robot_config(load_robot_config('configs/robot/sim_reference.yaml'), 'artifacts/unity/robot_config.json'))"

# tests
python -m uv run pytest tests/contract/unity tests/unit/adapters/unity -q
```

Connecting (after `EXT-UNITY-01` is resolved):

```python
from conrad.adapters.unity import UnityRobotHardware
from conrad.sim.unity import load_unity_settings

settings, unity = load_unity_settings("configs/sim/unity_lockstep.yaml")
hw = UnityRobotHardware(unity.bridge, robot_config, ids, mission_id, run_id, payload_store=object_store)
hw.connect()
hw.reset(seed=settings.run.seed)
state = hw.step(unity.physics_dt_ns)  # the experiment runner owns time in lock-step mode
```

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
