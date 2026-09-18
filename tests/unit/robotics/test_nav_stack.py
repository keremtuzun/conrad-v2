import numpy as np

from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.navigation import GoalStatus, NavigationStack, NavRecordType
from conrad.robotics.safety import SafetyState
from conrad.runtime.command_gateway import CommandGateway
from conrad.schemas.decision import NavigationGoal
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.settings import CommandMode, ExecutionLane, RuntimeSettings
from conrad.sim.kernel import FaultType, build_sim_hardware

CFG = load_robot_config("configs/robot/sim_reference.yaml")


def setup(seed=2):
    ids = IdFactory(seed=seed)
    mission, run = ids.new(), ids.new()
    hw = build_sim_hardware(CFG, seed)
    hw.kernel.reset(np.array([0.0, 0.0, -4.0]))
    stack = NavigationStack(hw, CFG, ids, mission, run, Pose(frame_id="WORLD", position_m=(0.0, 0.0, -4.0)))
    gw = CommandGateway(
        hw,
        CFG,
        RuntimeSettings(command_mode=CommandMode.SIMULATED),
        ExecutionLane.SIMULATION,
        mission,
        run,
        state_age_s=stack.state_age_s,
    )
    hw.advance(0.1)
    return ids, hw, stack, gw


def goal(ids, target, **oc):
    return NavigationGoal(
        goal_id=ids.new(),
        trace_id=ids.new(),
        target_pose=Pose(frame_id="WORLD", position_m=target),
        position_tolerance_m=0.4,
        orientation_tolerance_rad=0.3,
        observation_constraints=oc,
        risk_limit=0.1,
    )


def run(hw, stack, gw, seconds):
    acks = []
    for _ in range(round(seconds / 0.02)):
        res = stack.step()
        if res.decision.authorized:
            acks.append(gw.submit(res.command))
        hw.advance(0.02)
    return acks


def test_causal_trace_chain_and_gateway_acceptance():
    ids, hw, stack, gw = setup()
    g = goal(ids, (2.0, 0.0, -4.0))
    traj = stack.set_goal(g)
    acks = run(hw, stack, gw, 15.0)
    assert acks and all(a.accepted for a in acks)
    assert stack.status is GoalStatus.COMPLETE
    recs = stack.records
    assert {r.trace_id for r in recs} == {g.trace_id}
    kinds = [r.record_type for r in recs]
    assert kinds[:2] == [NavRecordType.GOAL_ACCEPTED, NavRecordType.TRAJECTORY_PLANNED]
    alloc = [r for r in recs if r.record_type == NavRecordType.COMMAND_ALLOCATED]
    wrench_ids = {r.wrench_id for r in recs if r.record_type == NavRecordType.WRENCH_REQUESTED}
    assert all(r.goal_id == g.goal_id and r.trajectory_id == traj.trajectory_id for r in alloc)
    assert all(r.wrench_id in wrench_ids for r in alloc)
    assert hw.truth_access().true_state().position_world_m[0] > 1.5  # evaluation-only check


def test_goal_below_max_depth_is_rejected_and_vehicle_holds():
    ids, hw, stack, gw = setup()
    assert stack.set_goal(goal(ids, (0.0, 0.0, -80.0))) is None
    assert stack.status is GoalStatus.REJECTED
    assert stack.records[-1].payload["reason"] == "GOAL_OUTSIDE_ENVELOPE"
    run(hw, stack, gw, 3.0)
    assert np.linalg.norm(hw.truth_access().true_state().position_world_m - [0, 0, -4]) < 0.3


def test_leak_overrides_autonomy_and_only_explicit_stop_is_authorized():
    ids, hw, stack, gw = setup()
    stack.set_goal(goal(ids, (5.0, 0.0, -4.0)))
    run(hw, stack, gw, 1.0)
    hw.inject_fault(FaultType.LEAK_SIGNAL)
    res = stack.step()
    assert res.assessment.state is SafetyState.RECOVER and res.assessment.zero_thrust_required
    # autonomy is overridden: the only thing that may pass is an explicit all-zero stop command
    assert res.decision.authorized and all(v == 0.0 for v in res.command.thruster_commands.values())
    run(hw, stack, gw, 1.0)
    assert abs(hw.kernel.thrusters.applied_command).max() == 0.0


def test_imu_dropout_makes_state_stale_and_forces_zero_thrust():
    _ids, hw, stack, gw = setup()
    hw.inject_fault(FaultType.SENSOR_DROPOUT, "imu", duration_s=5.0)
    run(hw, stack, gw, 1.0)
    res = stack.step()
    assert res.assessment.state is SafetyState.HOLD and "STATE_STALE" in res.assessment.reason_codes
    assert res.decision.authorized and all(v == 0.0 for v in res.command.thruster_commands.values())
