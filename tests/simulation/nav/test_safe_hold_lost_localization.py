"""The FLAGSHIP-UNITY safe-hold condition on the Python kernel: a position fix lost mid-mission, never regained.

FLAGSHIP-UNITY (2026-09-20) lost the fix at 90 s of a 150 s mission. The supervisor reached HOLD with
LOCALIZATION_LOST, but the configured safe-hold action was STATION_KEEP, so the stack kept running a position
controller against the dead-reckoning estimate: the estimate walked 4.74 m away, the controller chased it and the
TRUE position left the mission boundary at 122.6 s (`MISSION_BOUNDARY_VIOLATION`).

This test runs the same condition on the L1 kernel (SYNTHETIC_ONLY physics, no Unity player): 150 s, control
period 0.1 s, 1 Hz USBL-like fixes until 90 s and none afterwards, plus a 5x IMU noise fault at the outage so the
dead reckoning really drifts (as NAV-007 does). It asserts the safe outcome: once the estimate is declared
untrustworthy no motion is commanded, the TRUE speed stays near zero and the TRUE position stays inside the
mission boundary for the whole declared duration.
"""

from functools import lru_cache

import numpy as np
import pytest

from conrad.evaluation.nav_benchmarks.scenarios import SyntheticPositionFixRenderer
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.navigation import NavigationStack, NavigationStackConfig
from conrad.robotics.safety import MissionBoundary, Reason, SafeHoldAction, SafetyConfig, SafetyState
from conrad.runtime.command_gateway import CommandGateway
from conrad.schemas.decision import NavigationGoal
from conrad.schemas.frames import WORLD, Pose
from conrad.schemas.ids import IdFactory
from conrad.settings import CommandMode, ExecutionLane, RuntimeSettings
from conrad.sim.kernel import FaultType, SimKernelConfig, build_sim_hardware

CFG = load_robot_config("configs/robot/sim_reference.yaml")  # safe_hold_action = STATION_KEEP
START = (0.0, 0.0, -5.0)
GOAL = (6.0, 0.0, -5.0)
BOUNDARY = MissionBoundary(min_xyz_m=(-6.0, -5.0, -12.0), max_xyz_m=(12.0, 5.0, 0.0))
DT, DURATION_S, OUTAGE_S, SEED = 0.1, 150.0, 90.0, 9
DEADLINE_S = 40.0  # evaluation allowance, as declared for the flagship (LOCALIZATION_LOST after the outage)
STOPPED_MPS = 0.1  # same evaluation threshold as the I2 fault cases (conrad/sim/mission/unity_faults.py)


@lru_cache(maxsize=1)
def flagship_condition():
    """One 150 s run: transit, arrive, then lose the fix for good. Returns per-step truth and assessments."""
    ids = IdFactory(seed=SEED)
    mission_id, run_id = ids.new(), ids.new()
    renderer = SyntheticPositionFixRenderer(
        mission_id, run_id, ids, np.random.default_rng([SEED, 17]), 1.0, 0.1, [[OUTAGE_S, 1e9]]
    )
    hw = build_sim_hardware(CFG, SEED, SimKernelConfig(), renderer=renderer)
    hw.kernel.reset(np.array(START))
    truth = hw.truth_access()
    stack = NavigationStack(
        hw,
        CFG,
        ids,
        mission_id,
        run_id,
        Pose(frame_id=WORLD, position_m=START),
        config=NavigationStackConfig(control_period_s=DT, safety=SafetyConfig(boundary=BOUNDARY)),
    )
    hw.set_estimated_pose_provider(lambda: stack.estimator.get_state().pose)
    gateway = CommandGateway(
        hw,
        CFG,
        RuntimeSettings(command_mode=CommandMode.SIMULATED),
        ExecutionLane.SIMULATION,
        mission_id,
        run_id,
        state_age_s=stack.state_age_s,
    )
    hw.advance(0.1)
    assert stack.set_goal(
        NavigationGoal(
            goal_id=ids.new(),
            trace_id=ids.new(),
            target_pose=Pose(frame_id=WORLD, position_m=GOAL),
            position_tolerance_m=0.3,
            orientation_tolerance_rad=0.3,
            observation_constraints={},
            risk_limit=0.1,
        )
    )
    rows, fault_fired = [], False
    for _ in range(round(DURATION_S / DT)):
        if not fault_fired and hw.kernel.t_s >= OUTAGE_S:
            fault_fired = True
            hw.inject_fault(FaultType.SENSOR_NOISE, "imu", 5.0)
        res = stack.step()
        if res.decision.authorized:
            gateway.submit(res.command)
        hw.advance(DT)
        rows.append(
            {
                "t_s": hw.kernel.t_s,
                "assessment": res.assessment,
                "authorized": res.decision.authorized,
                "commands": dict(res.command.thruster_commands),
                "true_m": truth.true_state().position_world_m.copy(),
                "estimated_m": np.asarray(res.state.pose.position_m),
            }
        )
    assert fault_fired and renderer.delivered > 0
    return rows, truth.collision_count


def _hold_index(rows):
    for i, r in enumerate(rows):
        a = r["assessment"]
        if a.hold_required and Reason.LOCALIZATION_LOST in a.reason_codes:
            return i
    raise AssertionError("the run never reached HOLD with LOCALIZATION_LOST")


def test_localization_is_declared_lost_after_the_outage():
    rows, _ = flagship_condition()
    k = _hold_index(rows)
    a = rows[k]["assessment"]
    assert a.state is SafetyState.HOLD
    assert OUTAGE_S < rows[k]["t_s"] <= OUTAGE_S + DEADLINE_S
    assert not a.state_trustworthy and Reason.STATE_NOT_TRUSTWORTHY in a.reason_codes
    assert a.safe_hold_action is SafeHoldAction.ZERO_THRUST  # not the configured STATION_KEEP
    assert all(r["assessment"].state is SafetyState.NORMAL for r in rows[: round(OUTAGE_S / DT) - 200])


def test_no_motion_is_commanded_once_the_estimate_is_untrustworthy():
    """The defect: STATION_KEEP would keep thrusting against the drifting estimate."""
    rows, _ = flagship_condition()
    after = rows[_hold_index(rows) :]
    assert len(after) > 300  # at least 30 s of hold to observe
    authorized = [r for r in after if r["authorized"]]
    assert len(authorized) == len(after)  # the stack sends an explicit zero command, never a refused one
    assert max(abs(v) for r in authorized for v in r["commands"].values()) == 0.0
    assert all(r["assessment"].zero_thrust_required for r in after)


def test_true_speed_stays_near_zero_and_the_vehicle_stays_inside_the_boundary():
    rows, collisions = flagship_condition()
    t = np.array([r["t_s"] for r in rows])
    true_m = np.array([r["true_m"] for r in rows])
    last = t >= t[-1] - 10.0
    speed = float((np.linalg.norm(np.diff(true_m[last], axis=0), axis=1) / np.diff(t[last])).mean())
    assert speed <= STOPPED_MPS
    assert all(BOUNDARY.contains(p) for p in true_m)
    assert collisions == 0
    k = _hold_index(rows)
    est = np.array([r["estimated_m"] for r in rows])
    # the estimate really did run away, so this is the flagship condition and not a quiet run: the vehicle
    # simply did not follow it horizontally. The residual vertical motion is the vehicle's +0.3 N net
    # buoyancy (11.5 kg against 0.01125 m^3), i.e. the declared passive fail-safe ascent, not a chase.
    assert np.linalg.norm(est[k:] - est[k], axis=1).max() > 1.0
    assert np.linalg.norm(true_m[k:, :2] - true_m[k, :2], axis=1).max() < 0.5
    assert np.abs(true_m[k:, 2] - true_m[k, 2]).max() < 2.0


@pytest.mark.parametrize("action", [SafeHoldAction.STATION_KEEP, SafeHoldAction.ZERO_THRUST])
def test_the_rule_does_not_depend_on_the_configured_action(action):
    """Whatever RobotConfig declares, an untrustworthy estimate gets the open-loop action."""
    safety = CFG.safety.model_copy(
        update={"safe_hold_action": CFG.safety.safe_hold_action.model_copy(update={"value": action.value})}
    )
    cfg = CFG.model_copy(update={"safety": safety})
    ids = IdFactory(seed=3)
    hw = build_sim_hardware(cfg, 3, SimKernelConfig())  # no fix renderer: blind from the first step
    hw.kernel.reset(np.array(START))
    stack = NavigationStack(
        hw,
        cfg,
        ids,
        ids.new(),
        ids.new(),
        Pose(frame_id=WORLD, position_m=START),
        config=NavigationStackConfig(control_period_s=DT),
    )
    holds = []
    for _ in range(round(60.0 / DT)):
        res = stack.step()
        hw.advance(DT)
        if res.assessment.hold_required and Reason.LOCALIZATION_LOST in res.assessment.reason_codes:
            holds.append(res)
    assert holds
    assert all(r.assessment.safe_hold_action is SafeHoldAction.ZERO_THRUST for r in holds)
    assert all(v == 0.0 for r in holds for v in r.command.thruster_commands.values())
