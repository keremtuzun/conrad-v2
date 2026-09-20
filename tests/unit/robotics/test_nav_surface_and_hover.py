"""I2 repair: the surface datum comes from the mission context, and the hover-quiet velocity prior."""

import numpy as np
import pytest

from conrad.evaluation.nav_benchmarks.runner import run_benchmark
from conrad.robotics.estimation import EkfConfig, EkfStateEstimator
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.navigation import NavigationStackConfig
from conrad.robotics.safety import Reason, SafetyConfig
from conrad.schemas.frames import WORLD, Pose
from conrad.schemas.robot import DepthSample
from conrad.schemas.timebase import stamp
from conrad.sim.kernel import SimKernelConfig, build_sim_hardware

CFG = load_robot_config("configs/robot/sim_reference.yaml")
SURFACE = 30.0


def _ekf(surface: float, z0: float) -> EkfStateEstimator:
    return EkfStateEstimator(
        CFG,
        Pose(frame_id=WORLD, position_m=(0.0, 0.0, z0)),
        stamp(0.0, "SIM"),
        EkfConfig(water_surface_z_m=surface),
    )


def test_depth_is_measured_below_the_configured_surface():
    ekf = _ekf(SURFACE, 2.0)  # 28 m below a surface at z = +30
    for i in range(20):
        ekf.update_depth(DepthSample(timestamp=stamp(0.1 * (i + 1), "SIM"), frame_id="D", depth_m=27.9))
    assert ekf.get_state().pose.position_m[2] == pytest.approx(2.1, abs=0.02)
    assert not ekf.health_reasons()


def test_kernel_reports_true_depth_below_its_surface():
    hw = build_sim_hardware(CFG, 3, SimKernelConfig(water_surface_z_m=SURFACE))
    hw.kernel.reset(np.array([0.0, 0.0, 2.0]))
    hw.advance(0.5)
    depth = hw.get_depth()
    assert depth is not None
    assert depth.depth_m == pytest.approx(SURFACE - 2.0, abs=0.1)


def test_stack_config_needs_one_surface():
    with pytest.raises(ValueError, match="water_surface_z_m"):
        NavigationStackConfig(estimator=EkfConfig(water_surface_z_m=SURFACE))
    cfg = NavigationStackConfig.for_surface(SURFACE, control_period_s=0.05, safety=SafetyConfig(sigma_k=3.0))
    assert cfg.estimator.water_surface_z_m == cfg.safety.water_surface_z_m == SURFACE
    assert cfg.safety.sigma_k == 3.0 and cfg.control_period_s == 0.05


def test_supervisor_depth_limit_uses_the_surface():
    from conrad.robotics.safety import SafetyInputs, SafetySupervisor
    from conrad.schemas.ids import IdFactory

    max_depth = float(CFG.safety.max_depth_m.require("max_depth_m"))
    state = _ekf(SURFACE, SURFACE - max_depth - 1.0).get_state()  # 1 m below max depth under the real surface
    for surface, expect in ((SURFACE, True), (0.0, False)):
        sup = SafetySupervisor(CFG, IdFactory(seed=1), SafetyConfig(water_surface_z_m=surface))
        a = sup.assess(SafetyInputs(now_ns=state.timestamp.time_ns, clock_domain="SIM", state=state))
        assert (Reason.DEPTH_LIMIT in a.reason_codes) is expect


def test_quiet_prior_turns_into_the_full_prior_when_turning():
    ekf = _ekf(0.0, -5.0)
    full = np.asarray(ekf.config.velocity_prior_sigma_mps)
    # blind (no accepted fix yet): the full prior, so the position sigma grows honestly (NAV-007)
    assert np.allclose(ekf._prior_sigma(), full)
    ekf._last_fix_ns = ekf._stamp.time_ns  # a fix just arrived
    quiet = ekf.config.velocity_prior_quiet_sigma_mps
    assert quiet is not None
    assert np.allclose(ekf._prior_sigma(), quiet)
    ekf._w = np.array([0.0, 0.0, ekf.config.velocity_prior_turn_rate_ref_rps])
    assert np.allclose(ekf._prior_sigma(), full)
    legacy = EkfStateEstimator(
        CFG,
        Pose(frame_id=WORLD, position_m=(0.0, 0.0, -5.0)),
        stamp(0.0, "SIM"),
        EkfConfig.legacy_baseline(),
    )
    assert np.allclose(legacy._prior_sigma(), full)


def test_station_keeping_dev_seed_has_margin():
    """NAV-005 on a development noise seed (configs/eval/partitions_nav.yaml) passes with margin."""
    r = run_benchmark("NAV-005", 7410001)
    assert r["success"] and r["station_rms_error_m"] < 0.12
