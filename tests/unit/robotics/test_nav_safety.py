import math

import numpy as np
import pytest

from conrad.robotics.allocation import ThrusterAllocator
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.safety import (
    MissionBoundary,
    Reason,
    SafetyConfig,
    SafetyInputs,
    SafetyState,
    SafetySupervisor,
    collision_envelope,
)
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import BatteryState, HealthLevel, RobotState, SystemHealth, WrenchCommand
from conrad.schemas.timebase import stamp

CFG = load_robot_config("configs/robot/sim_reference.yaml")
NOW = 10_000_000_000


def state(t_s=10.0, sigma=0.1, pos=(0, 0, -5), speed=0.0, health=HealthLevel.OK):
    cov = np.diag([sigma**2] * 3 + [0.01] * 3).reshape(-1)
    return RobotState(
        timestamp=stamp(t_s, "SIM"),
        pose=Pose(frame_id="WORLD", position_m=pos, covariance_6x6=tuple(float(x) for x in cov)),
        linear_velocity_body_mps=(speed, 0.0, 0.0),
        angular_velocity_body_rps=(0.0, 0.0, 0.0),
        estimator_health=health,
        estimator_name="test",
    )


def inputs(**kw):
    base = {
        "now_ns": NOW,
        "clock_domain": "SIM",
        "state": state(),
        "health": SystemHealth(timestamp=stamp(10.0, "SIM"), overall=HealthLevel.OK, leak_detected=False),
        "battery": BatteryState(timestamp=stamp(10.0, "SIM"), remaining_fraction=0.9),
    }
    return SafetyInputs(**{**base, **kw})


def sup(**cfg):
    ids = IdFactory(seed=4)
    return SafetySupervisor(CFG, ids, SafetyConfig(**cfg)), ThrusterAllocator(CFG, ids), ids


def command(alloc, ids, force=(5.0, 0, 0), now=NOW, clock="SIM"):
    w = WrenchCommand(
        command_id=ids.new(),
        trace_id=ids.new(),
        timestamp=stamp(now / 1e9, clock),
        frame_id="ROBOT",
        force_n=force,
        torque_nm=(0, 0, 0),
    )
    return alloc.allocate(w, mission_id=ids.new(), run_id=ids.new(), now_ns=now, clock_domain=clock)[0]


def test_normal_authorizes_with_reason_codes():
    s, alloc, ids = sup()
    a = s.assess(inputs())
    assert a.state is SafetyState.NORMAL and a.speed_scale == 1.0
    d = s.authorize(command(alloc, ids), a, NOW, "SIM")
    assert d.authorized and d.command.safety_authorization.safety_state == "NORMAL"
    assert d.command.safety_authorization.command_id == d.command.command_id
    assert Reason.ALTITUDE_UNMONITORED in d.reason_codes  # no altimeter: said, not hidden


@pytest.mark.parametrize(
    ("kw", "expected", "code"),
    [
        ({"state": None}, SafetyState.HOLD, Reason.STATE_MISSING),
        ({"state": state(t_s=9.0)}, SafetyState.HOLD, Reason.STATE_STALE),
        ({"state": state(sigma=2.0)}, SafetyState.HOLD, Reason.LOCALIZATION_LOST),
        ({"state": state(sigma=1.0)}, SafetyState.DEGRADED, Reason.POSE_SIGMA_ELEVATED),
        ({"state": state(health=HealthLevel.FAULT)}, SafetyState.HOLD, Reason.LOCALIZATION_LOST),
        ({"estimator_reasons": ("LOCALIZATION_LOST",)}, SafetyState.HOLD, Reason.LOCALIZATION_LOST),
        ({"state": state(pos=(0, 0, -59.9))}, SafetyState.HOLD, Reason.DEPTH_LIMIT),
        ({"altitude_m": 0.2}, SafetyState.HOLD, Reason.ALTITUDE_LIMIT),
        (
            {"battery": BatteryState(timestamp=stamp(10.0, "SIM"), remaining_fraction=0.1)},
            SafetyState.RETURN,
            Reason.BATTERY_LOW,
        ),
        (
            {"battery": BatteryState(timestamp=stamp(10.0, "SIM"), remaining_fraction=0.18)},
            SafetyState.DEGRADED,
            Reason.BATTERY_RESERVE,
        ),
        ({"battery": None}, SafetyState.DEGRADED, Reason.BATTERY_UNKNOWN),
        ({"obstacle_clearance_m": 0.3}, SafetyState.HOLD, Reason.COLLISION_ENVELOPE),
        ({"obstacle_clearance_m": 0.5}, SafetyState.DEGRADED, Reason.COLLISION_MARGIN),
        ({"tracking_error_m": 3.0}, SafetyState.DEGRADED, Reason.TRACKING_ERROR),
        (
            {
                "health": SystemHealth(
                    timestamp=stamp(10.0, "SIM"),
                    overall=HealthLevel.DEGRADED,
                    devices={"H1": HealthLevel.FAULT},
                )
            },
            SafetyState.DEGRADED,
            Reason.DEGRADED_MANEUVERABILITY,
        ),
        (
            {
                "health": SystemHealth(
                    timestamp=stamp(10.0, "SIM"),
                    overall=HealthLevel.DEGRADED,
                    devices={"imu": HealthLevel.FAULT},
                )
            },
            SafetyState.DEGRADED,
            Reason.SENSOR_FAULT,
        ),
        (
            {
                "health": SystemHealth(
                    timestamp=stamp(10.0, "SIM"), overall=HealthLevel.FAULT, leak_detected=True
                )
            },
            SafetyState.RECOVER,
            Reason.LEAK,
        ),
    ],
)
def test_each_trigger(kw, expected, code):
    s, _, _ = sup()
    a = s.assess(inputs(**kw))
    assert a.state is expected and code in a.reason_codes


def test_boundary_and_hold_uses_configured_action_not_surface():
    s, alloc, ids = sup(boundary=MissionBoundary(min_xyz_m=(-5, -5, -20), max_xyz_m=(5, 5, 0)))
    a = s.assess(inputs(state=state(pos=(8, 0, -5))))
    assert a.state is SafetyState.HOLD and Reason.BOUNDARY in a.reason_codes
    assert f"{Reason.SAFE_HOLD}:STATION_KEEP" in a.reason_codes and not a.zero_thrust_required
    assert s.authorize(command(alloc, ids), a, NOW, "SIM").authorized  # station keeping may still thrust


def test_stale_state_requires_zero_thrust():
    s, alloc, ids = sup()
    a = s.assess(inputs(state=state(t_s=9.0)))
    assert a.zero_thrust_required
    d = s.authorize(command(alloc, ids), a, NOW, "SIM")
    assert not d.authorized and Reason.ZERO_REQUIRED in d.reason_codes
    zero = alloc.zero_command(
        trace_id=ids.new(),
        mission_id=ids.new(),
        run_id=ids.new(),
        now_ns=NOW,
        clock_domain="SIM",
        source_wrench_id=ids.new(),
    )
    assert s.authorize(zero, a, NOW, "SIM").authorized


def test_emergency_stop_latches_and_overrides_everything():
    s, alloc, ids = sup()
    s.emergency_stop()
    a = s.assess(inputs())
    assert a.state is SafetyState.EMERGENCY_STOP and a.zero_thrust_required
    d = s.authorize(command(alloc, ids), a, NOW, "SIM")
    assert not d.authorized and Reason.STATE_FORBIDS_MOTION in d.reason_codes
    assert s.assess(inputs()).state is SafetyState.EMERGENCY_STOP  # latched
    s.reset_latch()
    s2 = s.assess(inputs())
    assert s2.state is SafetyState.NORMAL


def test_relocalization_timeout_escalates_to_return():
    s, _, _ = sup(relocalization_timeout_s=5.0)
    s.assess(inputs(state=state(sigma=2.0)))
    a = s.assess(inputs(now_ns=NOW + 6_000_000_000, state=state(t_s=16.0, sigma=2.0)))
    assert a.state is SafetyState.RETURN and Reason.RELOCALIZATION_TIMEOUT in a.reason_codes


def test_command_level_refusals():
    s, alloc, ids = sup()
    a = s.assess(inputs())
    assert Reason.DEADLINE in s.authorize(command(alloc, ids, now=NOW - 10**9), a, NOW, "SIM").reason_codes
    assert Reason.CLOCK in s.authorize(command(alloc, ids, clock="WALL"), a, NOW, "SIM").reason_codes
    cmd = command(alloc, ids)
    assert (
        Reason.PRODUCER
        in s.authorize(cmd.model_copy(update={"producer": "model1"}), a, NOW, "SIM").reason_codes
    )
    bad = dict(cmd.thruster_commands)
    bad.pop("H1")
    assert (
        Reason.ACTUATOR_SET
        in s.authorize(cmd.model_copy(update={"thruster_commands": bad}), a, NOW, "SIM").reason_codes
    )
    nan = {**cmd.thruster_commands, "H1": math.nan}
    assert (
        Reason.ACTUATOR_RANGE
        in s.authorize(cmd.model_copy(update={"thruster_commands": nan}), a, NOW, "SIM").reason_codes
    )
    faulted = s.assess(
        inputs(
            health=SystemHealth(
                timestamp=stamp(10.0, "SIM"), overall=HealthLevel.DEGRADED, devices={"H1": HealthLevel.FAULT}
            )
        )
    )
    push = {**cmd.thruster_commands, "H1": 0.5}
    assert (
        Reason.ACTUATOR_FAULTED
        in s.authorize(cmd.model_copy(update={"thruster_commands": push}), faulted, NOW, "SIM").reason_codes
    )


def test_collision_envelope_grows_with_speed_and_uncertainty():
    c = SafetyConfig()
    base = collision_envelope(0.23, 0.0, 0.1, c)
    assert collision_envelope(0.23, 0.8, 0.1, c) > base
    assert collision_envelope(0.23, 0.0, 0.8, c) > base
    assert collision_envelope(0.23, 0.0, None, c) == math.inf


def test_unsupported_safe_hold_action_is_rejected():
    bad = CFG.safety.model_copy(
        update={"safe_hold_action": CFG.safety.safe_hold_action.model_copy(update={"value": "PRAY"})}
    )
    with pytest.raises(ValueError, match="unsupported safe_hold_action"):
        SafetySupervisor(CFG.model_copy(update={"safety": bad}), IdFactory(seed=1))
