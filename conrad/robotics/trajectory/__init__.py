"""Trajectory generation (ch20 Trajectory object): dynamically feasible, schema ``decision.Trajectory``."""

from conrad.robotics.trajectory.generator import (
    PLANNER_NAME,
    TrajectoryConfig,
    TrajectoryGenerator,
    TrajectorySampler,
    YawMode,
    derive_accel_limit,
)

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": ["ch20 Trajectory object", "ch20 Local planner"],
    "configuration_keys": ["TrajectoryConfig", "RobotConfig.safety.max_speed_mps", "RobotConfig.thrusters"],
    "assumptions": [
        "RobotConfig has no acceleration limit; derived as a fraction of thrust authority / total mass",
        "attitude reference is yaw-only (level flight); decision.Trajectory has no angular-velocity field",
    ],
    "baselines": ["trapezoidal arc-length profile with corner speed limits"],
    "acceptance_tests": ["tests/unit/robotics/test_nav_trajectory.py"],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "IMPLEMENTATION_METADATA",
    "PLANNER_NAME",
    "TrajectoryConfig",
    "TrajectoryGenerator",
    "TrajectorySampler",
    "YawMode",
    "derive_accel_limit",
]
