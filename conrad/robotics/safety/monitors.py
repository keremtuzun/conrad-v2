"""Safety states, reason codes, configuration and the collision envelope D_safe (ch20 Safety Supervisor)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pydantic import Field, field_validator

from conrad.schemas.base import ConradModel
from conrad.schemas.robot import BatteryState, RobotState, SystemHealth, ThrusterState


class SafetyState(str, Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    HOLD = "HOLD"
    RETURN = "RETURN"
    RECOVER = "RECOVER"
    EMERGENCY_STOP = "EMERGENCY_STOP"


SEVERITY = {s: i for i, s in enumerate(SafetyState)}


class SafeHoldAction(str, Enum):
    """Values ``RobotConfig.safety.safe_hold_action`` may take. Nothing is assumed beyond these."""

    STATION_KEEP = "STATION_KEEP"
    ZERO_THRUST = "ZERO_THRUST"
    SURFACE = "SURFACE"


#: Safe-hold actions that run a closed control loop against the vehicle's own position estimate. They are only
#: legitimate while that estimate is trustworthy: holding station on an estimate the supervisor has just declared
#: lost makes the controller chase the dead-reckoning drift (FLAGSHIP-UNITY, 2026-09-20).
ESTIMATE_CLOSED_LOOP_ACTIONS = frozenset({SafeHoldAction.STATION_KEEP})


class Reason:
    STATE_MISSING = "STATE_MISSING"
    STATE_STALE = "STATE_STALE"
    LOCALIZATION_LOST = "LOCALIZATION_LOST"
    POSE_SIGMA_HIGH = "POSE_SIGMA_HIGH"
    POSE_SIGMA_ELEVATED = "POSE_SIGMA_ELEVATED"
    RELOCALIZATION_TIMEOUT = "RELOCALIZATION_TIMEOUT"
    ESTIMATOR_DEGRADED = "ESTIMATOR_DEGRADED"
    LEAK = "LEAK_DETECTED"
    RECOVERY_BEHAVIOUR_OPEN = "RECOVERY_BEHAVIOUR_OPEN"
    HARDWARE_FAULT = "HARDWARE_FAULT"
    SENSOR_FAULT = "SENSOR_FAULT"
    DEGRADED_MANEUVERABILITY = "DEGRADED_MANEUVERABILITY"
    BATTERY_RESERVE = "BATTERY_RESERVE"
    BATTERY_LOW = "BATTERY_LOW"
    BATTERY_UNKNOWN = "BATTERY_UNKNOWN"
    BOUNDARY = "MISSION_BOUNDARY_VIOLATION"
    BOUNDARY_RISK = "MISSION_BOUNDARY_RISK"
    BOUNDARY_BREACH_UNLOCALIZED = "MISSION_BOUNDARY_BREACH_UNLOCALIZED"
    DEPTH_LIMIT = "DEPTH_LIMIT"
    ALTITUDE_LIMIT = "ALTITUDE_LIMIT"
    ALTITUDE_UNMONITORED = "ALTITUDE_UNMONITORED"
    COLLISION_ENVELOPE = "COLLISION_ENVELOPE"
    COLLISION_MARGIN = "COLLISION_MARGIN_REDUCED"
    TRACKING_ERROR = "TRACKING_ERROR"
    DEGRADED_DWELL = "DEGRADED_DWELL"
    OPERATOR_ESTOP = "OPERATOR_EMERGENCY_STOP"
    DEADLINE = "COMMAND_DEADLINE_EXPIRED"
    CLOCK = "WRONG_CLOCK_DOMAIN"
    DIGEST = "WRONG_ROBOT_CONFIG_DIGEST"
    PRODUCER = "NOT_PRODUCED_BY_ALLOCATOR"
    ACTUATOR_SET = "INVALID_ACTUATOR_SET"
    ACTUATOR_RANGE = "ACTUATOR_COMMAND_OUT_OF_RANGE"
    ACTUATOR_FAULTED = "COMMAND_TO_FAULTED_ACTUATOR"
    ZERO_REQUIRED = "ZERO_THRUST_REQUIRED"
    STATE_FORBIDS_MOTION = "SAFETY_STATE_FORBIDS_MOTION"
    SAFE_HOLD = "SAFE_HOLD_ACTION"
    STATE_NOT_TRUSTWORTHY = "STATE_ESTIMATE_NOT_TRUSTWORTHY"


class MissionBoundary(ConradModel):
    """Axis-aligned WORLD box the mission is allowed to use (mission context, not RobotConfig)."""

    frame_id: str = "WORLD"
    min_xyz_m: tuple[float, float, float]
    max_xyz_m: tuple[float, float, float]

    def contains(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= np.asarray(self.min_xyz_m)) and np.all(p <= np.asarray(self.max_xyz_m)))

    def contains_ball(self, p: np.ndarray, radius_m: float) -> bool:
        """True when the whole ball of that radius around ``p`` is inside the box (worst-case containment)."""
        r = max(0.0, float(radius_m))
        return bool(
            np.all(p - r >= np.asarray(self.min_xyz_m)) and np.all(p + r <= np.asarray(self.max_xyz_m))
        )


class SafetyConfig(ConradModel):
    degraded_sigma_fraction: float = Field(default=0.5, gt=0, le=1)
    degraded_speed_scale: float = Field(default=0.5, gt=0, le=1)
    relocalization_timeout_s: float = Field(default=60.0, gt=0)
    reaction_time_s: float = Field(default=0.5, ge=0)
    brake_decel_mps2: float = Field(default=0.3, gt=0)
    sigma_k: float = Field(default=2.0, ge=0)
    map_margin_m: float = Field(default=0.1, ge=0)
    depth_sigma_k: float = Field(default=3.0, ge=0)
    tracking_error_limit_m: float = Field(default=2.0, gt=0)
    degraded_dwell_s: float = Field(default=1.0, ge=0, description="min time in DEGRADED before NORMAL")
    boundary: MissionBoundary | None = None
    water_surface_z_m: float = Field(
        default=0.0, description="WORLD z of the water surface from the mission context; depth = surface - z"
    )
    untrusted_state_hold_action: SafeHoldAction = Field(
        default=SafeHoldAction.ZERO_THRUST,
        description=(
            "safe-hold action that replaces RobotConfig.safety.safe_hold_action while the state estimate is "
            "not trustworthy (missing, stale, estimator FAULT, LOCALIZATION_LOST or sigma above the limit). "
            "It may not be an action that closes a control loop on that estimate."
        ),
    )
    boundary_sigma_k: float = Field(
        default=3.0,
        ge=0,
        description="k of the worst-case mission-boundary test: the estimate is inflated by k*sigma_pose",
    )
    boundary_breach_escalates_to_estop: bool = Field(
        default=True,
        description="a boundary breach while the state estimate is not trustworthy latches EMERGENCY_STOP",
    )

    @field_validator("untrusted_state_hold_action")
    @classmethod
    def _untrusted_action_is_open_loop(cls, v: SafeHoldAction) -> SafeHoldAction:
        if v in ESTIMATE_CLOSED_LOOP_ACTIONS:
            raise ValueError(
                f"untrusted_state_hold_action={v.value} closes a control loop on the very estimate that was "
                f"declared untrustworthy; allowed: "
                f"{sorted(a.value for a in SafeHoldAction if a not in ESTIMATE_CLOSED_LOOP_ACTIONS)}"
            )
        return v


@dataclass(frozen=True)
class SafetyInputs:
    now_ns: int
    clock_domain: str
    state: RobotState | None
    estimator_reasons: tuple[str, ...] = ()
    health: SystemHealth | None = None
    battery: BatteryState | None = None
    thrusters: tuple[ThrusterState, ...] = ()
    obstacle_clearance_m: float | None = None
    altitude_m: float | None = None
    tracking_error_m: float | None = None


@dataclass(frozen=True)
class SafetyAssessment:
    state: SafetyState
    reason_codes: tuple[str, ...]
    speed_scale: float
    hold_required: bool
    zero_thrust_required: bool
    d_safe_m: float
    faulted_actuators: frozenset[str] = field(default_factory=frozenset)
    #: the safe-hold action every consumer must execute for THIS assessment. It is the configured
    #: ``RobotConfig.safety.safe_hold_action`` while the estimate is trustworthy and
    #: ``SafetyConfig.untrusted_state_hold_action`` otherwise; never the configured one blindly.
    safe_hold_action: SafeHoldAction = SafeHoldAction.ZERO_THRUST
    #: False when the estimate is missing, stale, or the estimator/sigma declares localization lost.
    state_trustworthy: bool = True


def collision_envelope(
    vehicle_radius_m: float, speed_mps: float, pose_sigma_m: float | None, config: SafetyConfig
) -> float:
    """D_safe = geometry + reaction distance + stopping distance + k*sigma_pose + map margin.

    An unknown pose sigma is treated as the worst case the envelope can express (infinite margin).
    """
    if pose_sigma_m is None:
        return math.inf
    v = max(0.0, speed_mps)
    return (
        vehicle_radius_m
        + v * config.reaction_time_s
        + v * v / (2.0 * config.brake_decel_mps2)
        + config.sigma_k * pose_sigma_m
        + config.map_margin_m
    )
