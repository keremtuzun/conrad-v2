"""I2 "faults reach defined safe states": the Unity device-name mapping, a thruster-only FAULT, and the checker."""

import numpy as np
import pytest

from conrad.adapters.unity.conversion import device_id
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.safety import Reason, SafetyInputs, SafetyState, SafetySupervisor
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import BatteryState, HealthLevel, SystemHealth
from conrad.schemas.timebase import stamp
from conrad.sim.mission.unity_faults import check_safe_state, fault_cases

CFG = load_robot_config("configs/robot/sim_reference.yaml")


def test_unity_device_names_map_to_rhi_ids():
    assert device_id("thruster:H1") == "H1" and device_id("sensor:imu") == "imu" and device_id("x") == "x"


def _assess(devices, overall, leak=False):
    sup = SafetySupervisor(CFG, IdFactory(seed=1))
    health = SystemHealth(timestamp=stamp(0.0, "SIM"), overall=overall, leak_detected=leak, devices=devices)
    battery = BatteryState(timestamp=stamp(0.0, "SIM"), remaining_fraction=0.9)
    return sup.assess(SafetyInputs(now_ns=0, clock_domain="SIM", state=None, health=health, battery=battery))


def test_thruster_only_fault_is_degraded_maneuverability_not_hardware_hold():
    a = _assess({"H1": HealthLevel.FAULT}, HealthLevel.FAULT)
    assert Reason.DEGRADED_MANEUVERABILITY in a.reason_codes and Reason.HARDWARE_FAULT not in a.reason_codes
    assert "H1" in a.faulted_actuators
    b = _assess({"H1": HealthLevel.FAULT, "imu": HealthLevel.FAULT}, HealthLevel.FAULT)
    assert Reason.HARDWARE_FAULT in b.reason_codes and b.state is SafetyState.HOLD


def _log(states, t0=0.0, dt=0.5, cmd=None):
    return [[round((t0 + i * dt) * 1e9), st, list(rs), cmd] for i, (st, rs) in enumerate(states)]


TRAJ = {"t_s": [0.0, 1.0], "true_m": [[0, 0, 0], [0, 0, 0]], "estimated_m": [[0, 0, 0], [0, 0, 0]]}


def test_checker_passes_only_the_declared_response():
    expect = {"state": "RECOVER", "reason": "LEAK_DETECTED", "motion": "zero", "deadline_s": 1.0}
    zero = {"H1": 0.0}
    good = _log([("NORMAL", [])] * 4 + [("RECOVER", ["LEAK_DETECTED"])] * 4, cmd=zero)
    assert check_safe_state(good, TRAJ, expect, 2.0, None, 0)["passed"]
    late = _log([("NORMAL", [])] * 4 + [("NORMAL", [])] * 3 + [("RECOVER", ["LEAK_DETECTED"])], cmd=zero)
    assert not check_safe_state(late, TRAJ, expect, 2.0, None, 0)["checks"]["reached_in_time"]
    moving = _log([("NORMAL", [])] * 4 + [("RECOVER", ["LEAK_DETECTED"])] * 4, cmd={"H1": 0.2})
    assert not check_safe_state(moving, TRAJ, expect, 2.0, None, 0)["checks"]["motion_rule"]
    released = good[:-1] + _log([("NORMAL", [])], t0=3.5, cmd=zero)
    assert not check_safe_state(released, TRAJ, expect, 2.0, None, 0)["checks"]["held_to_end"]


def test_fault_cases_cover_the_required_faults():
    cfg = fault_cases()
    kinds = {c["fault"]["kind"] for c in cfg["cases"].values()}
    assert {"FIX_OUTAGE", "THRUSTER_FAILURE", "IMU_DROPOUT", "BATTERY_DEGRADATION", "LEAK_SIGNAL"} <= kinds
    for c in cfg["cases"].values():
        SafetyState(c["expect"]["state"])
        assert c["expect"]["deadline_s"] > 0
    assert np.isfinite(float(cfg["fault_t_s"])) and cfg["duration_s"] > cfg["fault_t_s"]


@pytest.mark.parametrize("motion", ["refused", "avoid_target"])
def test_checker_motion_rules(motion):
    expect = {"state": "RETURN", "reason": "BATTERY_LOW", "motion": motion, "deadline_s": 1.0}
    log = _log([("NORMAL", [])] * 2 + [("RETURN", ["BATTERY_LOW"])] * 2, cmd=None)
    out = check_safe_state(log, TRAJ, expect, 1.0, "H1", 0)
    assert out["checks"]["motion_rule"] is (motion == "refused")
