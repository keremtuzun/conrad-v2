import numpy as np

from conrad.robotics.allocation import ThrusterAllocator
from conrad.robotics.control import CascadedPidController, ControlReference
from conrad.robotics.hardware.config import load_robot_config
from conrad.schemas.frames import quat_from_euler
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import HealthLevel, RobotState
from conrad.schemas.timebase import stamp
from conrad.sim.kernel import build_sim_hardware

CFG = load_robot_config("configs/robot/sim_reference.yaml")


def truth_state(hw):
    """Test-only: isolate the controller from the estimator by feeding it the true state."""
    s = hw.truth_access().true_state()
    return RobotState(
        timestamp=stamp(s.t_s, "SIM"),
        pose=s.pose(),
        linear_velocity_body_mps=tuple(s.linear_velocity_body_mps),
        angular_velocity_body_rps=tuple(s.angular_velocity_body_rps),
        estimator_health=HealthLevel.OK,
        estimator_name="TEST_TRUTH",
    )


def run_step(target, yaw_quat=(1.0, 0.0, 0.0, 0.0), seconds=25.0, current=None):
    ids = IdFactory(seed=5)
    hw = build_sim_hardware(CFG, 3, current_field=current)
    ctrl, alloc = CascadedPidController(CFG, ids), ThrusterAllocator(CFG, ids)
    ref = ControlReference(np.array(target, dtype=float), np.array(yaw_quat))
    trace, dt, log = ids.new(), 0.02, []
    for _ in range(round(seconds / dt)):
        st = truth_state(hw)
        w = ctrl.compute(st, ref, dt, trace, st.timestamp)
        r = alloc.solve(np.array([*w.force_n, *w.torque_nm]))
        hw.kernel.thrusters.command(hw.kernel.t_s, r.commands)
        hw.advance(dt)
        log.append(hw.truth_access().true_state().position_world_m.copy())
    return np.array(log), hw


def test_position_step_response():
    log, _ = run_step([1.0, 0.0, 0.0])
    x = log[:, 0]
    assert x.max() < 1.2  # overshoot below 20 %
    assert np.argmax(x > 0.9) * 0.02 < 8.0  # rise time
    assert np.abs(log[-100:] - [1.0, 0.0, 0.0]).max() < 0.03


def test_holds_depth_against_net_buoyancy_and_current():
    log, _ = run_step([0.0, 0.0, -2.0], current=lambda p, t: np.array([0.0, 0.25, 0.0]), seconds=30.0)
    assert np.abs(log[-100:] - [0.0, 0.0, -2.0]).max() < 0.05  # integral action rejects the current


def test_yaw_step_uses_quaternion_error():
    q = (np.cos(np.pi / 2), 0.0, 0.0, np.sin(np.pi / 2))  # yaw = pi, the Euler wrap-around case
    _, hw = run_step([0.0, 0.0, 0.0], yaw_quat=q, seconds=15.0)
    qt = hw.truth_access().true_state().orientation_wxyz
    assert abs(abs(qt[3]) - 1.0) < 0.01


def test_pitch_hold_compensates_synthetic_buoyancy_restoring_moment():
    target = np.asarray(quat_from_euler(0.0, 1.0, 0.0))
    _, hw = run_step([0.0, 0.0, 0.0], yaw_quat=target, seconds=25.0)
    actual = hw.truth_access().true_state().orientation_wxyz
    assert abs(float(actual @ target)) > 0.999


def test_anti_windup_limits_integral_under_saturation():
    ids = IdFactory(seed=5)
    ctrl = CascadedPidController(CFG, ids)
    hw = build_sim_hardware(CFG, 3)
    st = truth_state(hw)
    ref = ControlReference(np.array([1000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    for _ in range(2000):
        w = ctrl.compute(st, ref, 0.02, ids.new(), st.timestamp)
    assert abs(w.force_n[0]) <= ctrl.config.force_limit_n[0]
    assert abs(ctrl._integral[0] * ctrl.config.ki_velocity[0]) <= ctrl.config.integral_limit_n[0] + 1e-9
    assert w.frame_id == "ROBOT"
