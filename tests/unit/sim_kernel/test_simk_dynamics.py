import math

import numpy as np
import pytest

from conrad.robotics.hardware.config import load_robot_config
from conrad.schemas.frames import quat_from_euler
from conrad.schemas.robot import OpenParameterError, SourceKind
from conrad.sim.kernel import (
    SimKernelConfig,
    SimValidityLevel,
    build_sim_hardware,
    vehicle_params_from_config,
)

CFG = load_robot_config("configs/robot/sim_reference.yaml")
NEUTRAL_RHO = 11.5 / 0.01125


def quiet(**kw):
    return SimKernelConfig(thrust_noise_fraction=0.0, **kw)


def test_neutral_buoyancy_hover():
    hw = build_sim_hardware(CFG, 1, quiet(water_density_kgm3=NEUTRAL_RHO))
    hw.kernel.reset(np.array([0.0, 0.0, -5.0]))
    hw.advance(5.0)
    s = hw.truth_access().true_state()
    assert np.allclose(s.position_world_m, [0, 0, -5.0], atol=1e-6)
    assert np.linalg.norm(s.linear_velocity_body_mps) < 1e-6


def test_positive_buoyancy_rises():
    hw = build_sim_hardware(CFG, 1, quiet(water_density_kgm3=1100.0))
    hw.kernel.reset(np.array([0.0, 0.0, -5.0]))
    hw.advance(2.0)
    assert hw.truth_access().true_state().position_world_m[2] > -5.0


def test_terminal_velocity_under_constant_thrust():
    hw = build_sim_hardware(CFG, 1, quiet(water_density_kgm3=NEUTRAL_RHO))
    hw.kernel.thrusters.command(0.0, {"H1": 0.5, "H2": 0.5, "H3": 0.5, "H4": 0.5})
    hw.advance(20.0)
    fx = 4 * 40.0 * 0.25 * math.sqrt(0.5)
    expected = (-4.0 + math.sqrt(16.0 + 4 * 18.2 * fx)) / (2 * 18.2)
    v = hw.truth_access().true_state().linear_velocity_body_mps
    assert v[0] == pytest.approx(expected, rel=1e-3)
    assert abs(v[1]) < 1e-6


def test_restoring_moment_rights_the_vehicle():
    hw = build_sim_hardware(CFG, 1, quiet(water_density_kgm3=NEUTRAL_RHO))
    hw.kernel.reset(np.zeros(3), np.array(quat_from_euler(0.5, 0.0, 0.0)))
    hw.advance(0.05)
    assert hw.truth_access().true_state().angular_velocity_body_rps[0] < 0  # restoring roll
    hw.advance(60.0)
    q = hw.truth_access().true_state().orientation_wxyz
    roll = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    assert abs(roll) < 0.02


def test_current_drags_vehicle_along():
    hw = build_sim_hardware(
        CFG, 1, quiet(water_density_kgm3=NEUTRAL_RHO), current_field=lambda p, t: np.array([0.0, 0.3, 0.0])
    )
    hw.advance(30.0)
    s = hw.truth_access().true_state()
    assert s.position_world_m[1] > 3.0
    assert s.linear_velocity_body_mps[1] == pytest.approx(0.3, abs=0.01)


def test_thruster_lag_deadzone_asymmetry_latency():
    hw = build_sim_hardware(CFG, 1, quiet())
    bank = hw.kernel.thrusters
    bank.command(0.0, {"V1": 1.0, "V2": -1.0, "V3": 0.04})
    hw.advance(0.02)
    assert bank.thrust[bank.index["V1"]] == 0.0  # 20 ms latency not yet elapsed
    hw.advance(0.15)
    assert 0.5 * 40 < bank.thrust[bank.index["V1"]] < 0.75 * 40  # ~1 time constant
    hw.advance(2.0)
    assert bank.thrust[bank.index["V1"]] == pytest.approx(40.0, rel=1e-3)
    assert bank.thrust[bank.index["V2"]] == pytest.approx(-30.0, rel=1e-3)
    assert bank.thrust[bank.index["V3"]] == 0.0


def test_open_parameter_raises_and_level_is_l1():
    open_mass = CFG.mass_kg.model_copy(update={"value": None, "source": SourceKind.OPEN})
    with pytest.raises(OpenParameterError):
        vehicle_params_from_config(CFG.model_copy(update={"mass_kg": open_mass}))
    hw = build_sim_hardware(CFG, 1)
    assert hw.truth_access().validity_level is SimValidityLevel.L1_APPROXIMATE_PHYSICS


def test_collision_is_counted_and_blocked():
    def wall(p):
        return 2.0 - p[:, 0]

    hw = build_sim_hardware(CFG, 1, quiet(water_density_kgm3=NEUTRAL_RHO), sdf=wall)
    hw.kernel.thrusters.command(0.0, {"H1": 0.7, "H2": 0.7, "H3": 0.7, "H4": 0.7})
    hw.advance(10.0)
    truth = hw.truth_access()
    assert truth.collision_count == 1
    assert truth.true_state().position_world_m[0] <= 2.0 - hw.kernel.vehicle_radius_m + 1e-6


def test_determinism_same_seed():
    def run(seed):
        hw = build_sim_hardware(CFG, seed)
        hw.kernel.thrusters.command(0.0, {"H1": 0.4, "H2": 0.2, "V1": -0.3})
        hw.advance(3.0)
        s = hw.truth_access().true_state()
        imu = hw.get_imu()
        return s.position_world_m, imu.linear_acceleration_mps2

    a, b, c = run(7), run(7), run(8)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert not np.array_equal(a[0], c[0])
