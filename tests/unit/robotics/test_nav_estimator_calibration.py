"""Estimator honesty during fix loss (NAV-007 finding) and the safety consequences of estimator health."""

from functools import lru_cache

import numpy as np
import pytest

from conrad.domains.spatial.config import spatial_config
from conrad.domains.spatial.model import Model2S
from conrad.robotics.estimation import (
    DEPTH_REJECTED,
    ESTIMATOR_INCONSISTENT,
    LOCALIZATION_LOST,
    EkfConfig,
    EkfStateEstimator,
    MapConstraint,
)
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.navigation import NavigationStack
from conrad.robotics.safety import Reason, SafetyInputs, SafetyState, SafetySupervisor
from conrad.schemas.decision import NavigationGoal
from conrad.schemas.frames import ROBOT, WORLD, Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.robot import DepthSample, HealthLevel
from conrad.schemas.timebase import stamp
from conrad.schemas.world import SensorSpec
from conrad.sim.kernel import FaultType, SimKernelConfig, build_sim_hardware

CFG = load_robot_config("configs/robot/sim_reference.yaml")
CURRENT = np.array([0.0, 0.15, 0.0])  # NAV-007's cross current (SYNTHETIC_ONLY)


def outage_run(config, fix_s=20.0, outage_s=30.0, seed=5, imu_noise=5.0):
    return _outage_run(config.legacy_baseline() == config, fix_s, outage_s, seed, imu_noise)


@lru_cache(maxsize=8)
def _outage_run(legacy, fix_s, outage_s, seed, imu_noise):
    config = EkfConfig.legacy_baseline() if legacy else EkfConfig()
    """Open-loop transit in a cross current: fixes for fix_s, then none; IMU noise fault at the outage."""
    hw = build_sim_hardware(CFG, seed, SimKernelConfig(), current_field=lambda p, t: CURRENT)
    hw.kernel.reset(np.array([0.0, 0.0, -5.0]))
    hw.advance(0.05)
    rng = np.random.default_rng(seed)
    ekf = EkfStateEstimator(CFG, Pose(frame_id=WORLD, position_m=(0.0, 0.0, -5.0)), stamp(0.0, "SIM"), config)
    dt, out = 0.02, []
    for i in range(round((fix_s + outage_s) / dt)):
        t = i * dt
        if abs(t - fix_s) < dt / 2:
            hw.inject_fault(FaultType.SENSOR_NOISE, "imu", imu_noise)
        hw.kernel.thrusters.command(
            hw.kernel.t_s, {"H1": 0.25, "H2": 0.25, "V1": 0.1, "V2": 0.1, "V3": 0.1, "V4": 0.1}
        )
        hw.advance(dt)
        imu = hw.get_imu()
        assert imu is not None
        ekf.predict(imu, hw.get_thruster_state(), dt)
        if i % 3 == 0:
            depth = hw.get_depth()
            assert depth is not None
            ekf.update_depth(depth)
        truth = hw.truth_access().true_state().position_world_m  # test-only synthetic USBL-like fix
        if t < fix_s and i % 50 == 0:
            ekf.update_map_constraint(MapConstraint(truth + 0.1 * rng.standard_normal(3), 0.1))
        if t >= fix_s:
            e = np.asarray(ekf.get_state().pose.position_m) - truth
            out.append((np.linalg.norm(e), np.sqrt(np.trace(ekf.position_covariance)), ekf.localization_lost))
    return ekf, np.array(out)


def test_legacy_filter_is_overconfident_and_fixed_filter_is_not():
    _, legacy = outage_run(EkfConfig.legacy_baseline())
    ekf, fixed = outage_run(EkfConfig())
    assert legacy[-1, 0] > legacy[-1, 1]  # the defect: error outgrows the legacy covariance
    assert fixed[-1, 0] < 0.5 * fixed[-1, 1]  # the fixed covariance keeps growing with the drift
    assert (fixed[:, 0] / fixed[:, 1]).max() < 1.5
    assert ekf.name == "EST-B1-ekf18"
    lost = np.nonzero(fixed[:, 2])[0]
    assert lost.size and fixed[lost[0], 0] < 3.0  # LOST declared before the error reaches 2*max_sigma


def test_imu_noise_is_inferred_from_data_not_configured():
    assert not any("inject" in name or "multiplier" in name for name in EkfConfig.model_fields)
    quiet, _ = outage_run(EkfConfig(), fix_s=2.0, outage_s=4.0, imu_noise=1.0)
    noisy, _ = outage_run(EkfConfig(), fix_s=2.0, outage_s=4.0, imu_noise=5.0)
    assert noisy.imu_accel_var_est > 4.0 * quiet.imu_accel_var_est
    assert noisy.imu_gyro_var_est > quiet.imu_gyro_var_est


def static_ekf(config=None):
    return EkfStateEstimator(
        CFG, Pose(frame_id=WORLD, position_m=(0.0, 0.0, -5.0)), stamp(0.0, "SIM"), config
    )


def test_inconsistent_fix_innovations_declare_degraded_and_inflate_process_noise():
    ekf = static_ekf()
    rng = np.random.default_rng(3)
    for _ in range(20):  # fixes claim 5 cm but scatter by 10 cm: NIS far above its dof, below the gate
        ekf.update_map_constraint(
            MapConstraint(np.array([0.0, 0.0, -5.0]) + 0.1 * rng.standard_normal(3), 0.05)
        )
    assert ekf.nis_ratio_ewma > ekf.config.nis_ratio_limit
    assert ESTIMATOR_INCONSISTENT in ekf.health_reasons()
    assert ekf.get_state().estimator_health is HealthLevel.DEGRADED
    assert ekf.process_noise_scale > 1.0


def test_depth_outlier_is_gated():
    ekf = static_ekf()
    z = ekf.get_state().pose.position_m[2]
    ekf.update_depth(DepthSample(timestamp=stamp(0.1, "SIM"), frame_id="SENSOR_DEPTH", depth_m=25.0))
    assert ekf.get_state().pose.position_m[2] == z and DEPTH_REJECTED in ekf.health_reasons()
    ekf.update_depth(DepthSample(timestamp=stamp(0.2, "SIM"), frame_id="SENSOR_DEPTH", depth_m=5.01))
    assert DEPTH_REJECTED not in ekf.health_reasons()


def test_supervisor_slows_on_estimator_degraded_even_with_small_sigma():
    ekf = static_ekf()
    rng = np.random.default_rng(3)
    for _ in range(20):
        ekf.update_map_constraint(
            MapConstraint(np.array([0.0, 0.0, -5.0]) + 0.1 * rng.standard_normal(3), 0.05)
        )
    st = ekf.get_state()
    assert st.pose.position_sigma_m() < 0.5 * 1.5  # sigma alone would say NORMAL
    sup = SafetySupervisor(CFG, IdFactory(seed=1))
    a = sup.assess(
        SafetyInputs(
            now_ns=st.timestamp.time_ns, clock_domain="SIM", state=st, estimator_reasons=ekf.health_reasons()
        )
    )
    assert a.state is SafetyState.DEGRADED and a.speed_scale < 1.0
    assert Reason.ESTIMATOR_DEGRADED in a.reason_codes and ESTIMATOR_INCONSISTENT in a.reason_codes


def _hold_config(action):
    safety = CFG.safety.model_copy(
        update={"safe_hold_action": CFG.safety.safe_hold_action.model_copy(update={"value": action})}
    )
    return CFG.model_copy(update={"safety": safety})


@pytest.mark.parametrize("action", ["STATION_KEEP", "ZERO_THRUST"])
def test_growing_sigma_slows_then_holds_with_configured_policy(action):
    cfg = _hold_config(action)
    ids = IdFactory(seed=2)
    hw = build_sim_hardware(cfg, 2)  # no position-fix renderer: the vehicle is blind from the start
    hw.kernel.reset(np.array([0.0, 0.0, -4.0]))
    stack = NavigationStack(
        hw, cfg, ids, ids.new(), ids.new(), Pose(frame_id=WORLD, position_m=(0.0, 0.0, -4.0))
    )
    hw.advance(0.1)
    stack.set_goal(
        NavigationGoal(
            goal_id=ids.new(),
            trace_id=ids.new(),
            target_pose=Pose(frame_id=WORLD, position_m=(30.0, 0.0, -4.0)),
            position_tolerance_m=0.4,
            orientation_tolerance_rad=0.3,
            observation_constraints={},
            risk_limit=0.1,
        )
    )
    seen: list[SafetyState] = []
    hold_refs: list[np.ndarray] = []
    for _ in range(round(30.0 / 0.02)):
        res = stack.step()
        a, sigma = res.assessment, res.state.pose.position_sigma_m()
        assert sigma is not None  # the stack always carries a covariance
        if not seen or seen[-1] is not a.state:
            seen.append(a.state)
        if a.state is SafetyState.DEGRADED and Reason.POSE_SIGMA_ELEVATED in a.reason_codes:
            assert 0.75 < sigma <= 1.5 and a.speed_scale == stack.config.safety.degraded_speed_scale
        if a.state is SafetyState.HOLD:
            assert sigma > 1.5 and Reason.LOCALIZATION_LOST in a.reason_codes
            assert f"{Reason.SAFE_HOLD}:{action}" in a.reason_codes
            hold_refs.append(res.reference.position_world_m.copy())
            if action == "ZERO_THRUST":
                assert a.zero_thrust_required and all(
                    v == 0.0 for v in res.command.thruster_commands.values()
                )
            else:
                assert not a.zero_thrust_required
            if len(hold_refs) > 50:
                break
        hw.advance(0.02)
    assert seen[:3] == [SafetyState.NORMAL, SafetyState.DEGRADED, SafetyState.HOLD]
    assert hold_refs and np.allclose(hold_refs[0], hold_refs[-1])  # the hold target never moves


def _depth_obs(ids, store, spec, robot, wall_x, t):
    p = spec.parameters
    f = 0.5 * p["width_px"] / np.tan(np.radians(p["hfov_deg"]) / 2)
    u = (np.arange(p["width_px"]) + 0.5 - 0.5 * p["width_px"]) / f
    v = (np.arange(p["height_px"]) + 0.5 - 0.5 * p["height_px"]) / f
    uu, _ = np.meshgrid(u, v)
    rng = (wall_x - robot.position_m[0]) * np.sqrt(1 + uu**2 + np.meshgrid(u, v)[1] ** 2)
    return Observation(
        observation_id=ids.new(),
        mission_id=ids.new(),
        run_id=ids.new(),
        trace_id=ids.new(),
        sensor_id=spec.sensor_id,
        modality=Modality.DEPTH_RANGE,
        timestamp=stamp(t, "SIM"),
        sensor_frame=spec.frame_id,
        robot_pose_estimate=robot,
        payload_ref=store.put_array(rng.astype(np.float32)),
        sensor_context={"settings": dict(p)},
    )


def test_degraded_estimator_covariance_reaches_model2s(store):
    """The estimator's own RobotState pose (with covariance) is what 2S receives on each observation."""
    healthy, _ = outage_run(EkfConfig(), fix_s=10.0, outage_s=0.1)
    blind, _ = outage_run(EkfConfig())
    assert blind.get_state().pose.position_sigma_m() > 5 * healthy.get_state().pose.position_sigma_m()
    ids = IdFactory(7)
    spec = SensorSpec(
        sensor_id=ids.new(),
        modality="DEPTH_RANGE",
        frame_id="SENSOR_DEPTH_RANGE",
        mount_pose=Pose(frame_id=ROBOT, position_m=(0.0, 0.0, 0.0)),
        rate_hz=5.0,
        parameters={
            "width_px": 32,
            "height_px": 24,
            "hfov_deg": 60.0,
            "max_range_m": 10.0,
            "range_noise_sigma_m": 0.01,
        },
    )
    stats = {}
    for name, ekf in (("healthy", healthy), ("degraded", blind)):
        est = ekf.get_state().pose
        robot = Pose(frame_id=WORLD, position_m=(0.0, 0.0, 0.0), covariance_6x6=est.covariance_6x6)
        m = Model2S(ids, store, spatial_config({"refinement": {"enabled": False}}))
        m.initialize({"sensors": [spec]})
        for k in range(5):
            m.ingest_observations([_depth_obs(ids, store, spec, robot, 3.1, 1.0 + k)])
            m.update_beliefs(stamp(1.0 + k, "SIM"))
        base = m.map.base
        rows = base.all_rows()
        hit = rows[base.f["w_hit"][rows] > 0]
        stats[name] = (len(hit), float(base.probability(hit).max()))
    assert stats["degraded"][0] > stats["healthy"][0]  # larger footprint
    assert stats["degraded"][1] < stats["healthy"][1]  # lower confidence
    assert LOCALIZATION_LOST in blind.health_reasons()
