# Robot hardware interface and robotics stack

Every robot, simulated or physical, sits behind one interface. Upper layers never branch on adapter type.

## The interface (`conrad/robotics/hardware/interface.py`, FROZEN_CONTRACT)

`RobotHardwareInterface` (ABC): `capabilities()`, `robot_config_digest()`, `clock_domain()`, `now_ns()`,
`get_imu()`, `get_depth()`, `get_camera()`, `get_sonar()`, `get_thruster_state()`, `get_power_state()`,
`get_health()`, `send(AllocatedCommand) -> CommandAck`. Class attributes `adapter_name` and `is_physical`.

Only `conrad.runtime.command_gateway` may call `send` (static test
`tests/leakage/test_static_boundaries.py::test_only_the_gateway_calls_hardware_send`).

| Adapter | Code | Status |
|---|---|---|
| `SimRobotHardware` | `conrad/sim/kernel/hardware.py` | Runs. RK4 Fossen-form 6-DOF kernel, validity L1, fault injection, clock `SIM`. Truth only via `truth_access()` (`TruthAccess`). |
| `UnityRobotHardware` | `conrad/adapters/unity/hardware.py` | Tested against a mock player over ZeroMQ; the Unity player has not been executed (EXT-UNITY-01). See [UNITY.md](UNITY.md). |
| `PhysicalRobotHardware` | `conrad/robotics/hardware/interface.py` | Every method raises `HardwareUnavailableError` (EXT-HW-01). |
| `MissionHardware` | `conrad/sim/mission/hardware.py` | Integrated mission (in progress): `SimRobotHardware` plus `get_payload_observations()` for payload sensors declared in `RobotCapabilities.environmental_sensors`. |

## RobotConfig (`conrad/schemas/robot.py`, `conrad/robotics/hardware/config.py`)

Every physical parameter is a `Sourced` value with a `SourceKind`: MEASURED, IDENTIFIED, LITERATURE_PRIOR,
ENGINEERING_ESTIMATE, SYNTHETIC_ONLY or OPEN. An OPEN parameter has no value; MEASURED and IDENTIFIED need
provenance. `validate_for_lane` refuses OPEN parameters in simulation, dev and HIL (except `safety.*`) and any
parameter not MEASURED or IDENTIFIED in the physical lane.

- `configs/robot/sim_reference.yaml`: SYNTHETIC_ONLY vehicle, 11.5 kg, 8 thrusters.
- `configs/robot/physical_template.yaml`: every parameter OPEN, `thrusters: []`.

## Robotics stack (`conrad/robotics`)

`NavigationStack.step` (`navigation/stack.py`) reads hardware through a read-only wrapper and runs:

1. `EkfStateEstimator` (15-state error-state EKF): IMU predict, depth, visual/sonar and position-fix updates.
2. Goal and trajectory: `GoalManager`, `AStarPlanner` (UNKNOWN space FORBID by default), `TrajectoryGenerator`
   (trapezoidal), `PotentialFieldLocalPlanner` (bends the reference only).
3. `SafetySupervisor.assess`: states NORMAL, DEGRADED, HOLD, RETURN, RECOVER, EMERGENCY_STOP; worst wins; leak and
   operator E-stop latch.
4. `CascadedPidController` -> wrench -> `ThrusterAllocator` (bounded least squares) -> `AllocatedCommand`.
5. `SafetySupervisor.authorize` attaches a `SafetyAuthorization` or refuses.

The caller submits the command to the `CommandGateway` (reason codes in [ARCHITECTURE.md](ARCHITECTURE.md)).
MPC and learned-residual controllers exist as interfaces only (OPEN_BLOCKED).

## Characterization, identification, HIL

- `conrad/robotics/hardware/characterization`: merges a hardware handoff into a new RobotConfig and keeps the
  readiness (R0..R6) and autonomy gate ledger. `python -m conrad.robotics.hardware.characterization --help`.
- `conrad/robotics/hardware/identification`: fits parameters from logs A..F with held-out validation.
  Synthetic results are never promoted. `python -m conrad.robotics.hardware.identification --help`.
- `conrad/sim/hil`: HOST mode runs; TARGET mode is BLOCKED_EXTERNAL. See [HIL.md](HIL.md) and
  [PHYSICAL_INTEGRATION.md](PHYSICAL_INTEGRATION.md).

## Tests and results

```
uv run pytest tests/unit/robotics tests/property/robotics tests/unit/sim_kernel tests/simulation/nav -q
```

NAV-001..NAV-008 (L1 kernel, SYNTHETIC_ONLY vehicle, seeds 1 and 2) passed their scenario checks with 0 collisions
and 0 gateway rejections, final error 0.04 to 0.23 m. In NAV-007 the true error reached 5.8 m during the fix
outage while the EKF sigma stayed below the 1.5 m hold limit: the estimator is overconfident under unmodelled IMU
noise.

## Blockers

EXT-HW-01 physical RHI driver, EXT-HW-02 measured characterization, EXT-HW-03 frame contract (ADR-0002),
EXT-HW-04 identification logs, EXT-HW-05 hardware safety contract (RETURN/RECOVER motion and safe-hold behaviour),
EXT-UNITY-01 Unity execution, EXT-HIL-01 target compute.
