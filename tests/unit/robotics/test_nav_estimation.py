import numpy as np
import pytest

from conrad.robotics.estimation import EkfStateEstimator, MapConstraint
from conrad.robotics.hardware.config import load_robot_config
from conrad.schemas.frames import Pose
from conrad.schemas.robot import HealthLevel
from conrad.schemas.timebase import stamp
from conrad.sim.kernel import build_sim_hardware

CFG = load_robot_config("configs/robot/sim_reference.yaml")


def run(seconds, fix_period_s, initial_offset=(0.0, 0.0, 0.0), seed=4):
    hw = build_sim_hardware(CFG, seed)
    hw.kernel.reset(np.array([0.0, 0.0, -3.0]))
    hw.advance(0.05)
    rng = np.random.default_rng(99)
    start = Pose(frame_id="WORLD", position_m=tuple(np.array([0.0, 0.0, -3.0]) + initial_offset))
    ekf = EkfStateEstimator(CFG, start, stamp(0.0, "SIM"))
    dt, sigmas, errors = 0.02, [], []
    for i in range(round(seconds / dt)):
        t = i * dt
        hw.kernel.thrusters.command(
            hw.kernel.t_s,
            {"H1": 0.3 * np.sin(0.4 * t) + 0.2, "H2": 0.3, "V1": 0.1, "V2": 0.1, "V3": 0.1, "V4": 0.1},
        )
        hw.advance(dt)
        ekf.predict(hw.get_imu(), hw.get_thruster_state(), dt)
        if i % 3 == 0:
            ekf.update_depth(hw.get_depth())
        if fix_period_s and i % round(fix_period_s / dt) == 0:
            truth = hw.truth_access().true_state().position_world_m  # test-only synthetic USBL-like fix
            ekf.update_map_constraint(MapConstraint(truth + 0.05 * rng.standard_normal(3), 0.05))
        state = ekf.get_state()
        errors.append(
            np.linalg.norm(np.array(state.pose.position_m) - hw.truth_access().true_state().position_world_m)
        )
        sigmas.append(state.pose.position_sigma_m())
    return ekf, np.array(errors), np.array(sigmas)


def test_converges_with_position_fixes():
    ekf, err, sig = run(30.0, 0.5, initial_offset=(0.4, -0.3, 0.2))
    assert err[0] > 0.3
    assert err[-250:].mean() < 0.1
    assert sig[-1] < 0.1
    state = ekf.get_state()
    assert state.estimator_health is HealthLevel.OK and state.pose.frame_id == "WORLD"
    assert len(state.pose.covariance_6x6) == 36 and state.timestamp.clock_domain == "SIM"


def test_covariance_grows_without_fixes_until_localization_lost():
    ekf, _, sig = run(40.0, None)
    assert np.all(np.diff(sig[50::50]) > 0)  # monotone growth of horizontal uncertainty
    assert ekf.localization_lost and ekf.get_state().estimator_health is HealthLevel.FAULT
    cov = np.array(ekf.get_state().pose.covariance_6x6).reshape(6, 6)
    assert cov[2, 2] < 0.01 < cov[0, 0]  # depth keeps z observable


def test_inconsistent_fixes_are_gated_and_flag_loss():
    ekf, _, _ = run(5.0, 0.5)
    before = ekf.get_state().pose.position_m
    for _ in range(5):
        assert not ekf.update_map_constraint(MapConstraint(np.array([50.0, 50.0, -3.0]), 0.05))
    assert ekf.get_state().pose.position_m == before
    assert ekf.localization_lost and "LOCALIZATION_LOST" in ekf.health_reasons()


def test_rejects_non_world_and_bad_dt():
    ekf, _, _ = run(0.1, None)
    assert not ekf.update_map_constraint(MapConstraint(np.zeros(3), 0.1, frame_id="ASSET"))
    with pytest.raises(ValueError):
        EkfStateEstimator(CFG, Pose(frame_id="ROBOT", position_m=(0, 0, 0)), stamp(0.0, "SIM"))
