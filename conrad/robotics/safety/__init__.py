"""Safety Supervisor (ch20 Safety Supervisor / Safety states / Collision envelope / Localization loss)."""

from conrad.robotics.safety.monitors import (
    ESTIMATE_CLOSED_LOOP_ACTIONS,
    MissionBoundary,
    Reason,
    SafeHoldAction,
    SafetyAssessment,
    SafetyConfig,
    SafetyInputs,
    SafetyState,
    collision_envelope,
)
from conrad.robotics.safety.supervisor import SUPERVISOR_VERSION, SafetyDecision, SafetySupervisor

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch20 Safety Supervisor",
        "ch20 Safety states",
        "ch20 Collision envelope",
        "ch20 Localization-loss behavior",
        "ch34 command gateway",
    ],
    "configuration_keys": ["RobotConfig.safety.*", "RobotConfig.battery.reserve_fraction", "SafetyConfig"],
    "assumptions": [
        "safe_hold_action vocabulary: STATION_KEEP | ZERO_THRUST | SURFACE; anything else is rejected",
        "leak -> RECOVER with RECOVERY_BEHAVIOUR_OPEN (physical recovery behaviour is the hardware owner's)",
        "battery below reserve_fraction -> DEGRADED; below min_battery_fraction -> RETURN",
        "estimator health DEGRADED (NIS inconsistency, gated fixes/depth) -> DEGRADED + ESTIMATOR_DEGRADED",
        "pose sigma > degraded_sigma_fraction*max_pose_sigma_m -> DEGRADED (speed scaled, SYNTHETIC config)",
        "pose sigma > max_pose_sigma_m or estimator FAULT -> HOLD with the configured safe_hold_action",
        "an untrustworthy estimate (missing/stale/FAULT/LOCALIZATION_LOST/sigma>limit) replaces that action "
        "by SafetyConfig.untrusted_state_hold_action (ZERO_THRUST default); STATION_KEEP is refused there",
        "the mission boundary is tested on the estimate inflated by boundary_sigma_k*sigma "
        "(MISSION_BOUNDARY_RISK -> HOLD); a real breach with an untrustworthy estimate latches EMERGENCY_STOP",
        "D_safe = radius + v*t_react + v^2/(2 a_brake) + k*sigma_pose + map margin (SYNTHETIC_ONLY constants)",
    ],
    "baselines": ["rule-based deterministic supervisor"],
    "acceptance_tests": ["tests/unit/robotics/test_nav_safety.py"],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "ESTIMATE_CLOSED_LOOP_ACTIONS",
    "IMPLEMENTATION_METADATA",
    "SUPERVISOR_VERSION",
    "MissionBoundary",
    "Reason",
    "SafeHoldAction",
    "SafetyAssessment",
    "SafetyConfig",
    "SafetyDecision",
    "SafetyInputs",
    "SafetyState",
    "SafetySupervisor",
    "collision_envelope",
]
