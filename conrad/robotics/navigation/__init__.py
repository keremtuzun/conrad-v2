"""Navigation (ch20): goal manager, global/local planners, inspection primitives, executive stack."""

from conrad.robotics.navigation.global_planner import (
    AStarPlanner,
    CostHook,
    IsFree,
    KnowledgeQuery,
    PlannerConfig,
    PlanningError,
    UnknownSpacePolicy,
)
from conrad.robotics.navigation.goals import (
    GoalManager,
    GoalRejectedError,
    NavigationObjective,
    ObjectiveKind,
    incidence_angle,
)
from conrad.robotics.navigation.local_planner import (
    LocalDistance,
    LocalPlannerConfig,
    PotentialFieldLocalPlanner,
)
from conrad.robotics.navigation.stack import (
    GoalStatus,
    NavigationStack,
    NavigationStackConfig,
    StepResult,
)
from conrad.robotics.navigation.trace import NavRecordType, NavTraceRecord, NavTraceSink

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch20 Goal Manager",
        "ch20 Global planner / baselines",
        "ch20 Knowledge-aware planning",
        "ch20 Local planner",
        "ch20 Inspection primitives / Observation geometry",
    ],
    "configuration_keys": ["PlannerConfig", "LocalPlannerConfig", "NavigationGoal.observation_constraints"],
    "assumptions": [
        "primitive vocabulary is carried in NavigationGoal.observation_constraints (no schema field exists)",
        "UNKNOWN space policy defaults to FORBID; it is a safety policy separate from the map",
        "inspection sensor is forward-looking and level (yaw-only viewpoint)",
    ],
    "baselines": ["A* 26-connected lattice + line-of-sight shortcut", "potential-field local avoidance"],
    "acceptance_tests": ["tests/unit/robotics/test_nav_planning.py", "tests/simulation/nav"],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "IMPLEMENTATION_METADATA",
    "AStarPlanner",
    "CostHook",
    "GoalManager",
    "GoalRejectedError",
    "GoalStatus",
    "IsFree",
    "KnowledgeQuery",
    "LocalDistance",
    "LocalPlannerConfig",
    "NavRecordType",
    "NavTraceRecord",
    "NavTraceSink",
    "NavigationObjective",
    "NavigationStack",
    "NavigationStackConfig",
    "ObjectiveKind",
    "PlannerConfig",
    "PlanningError",
    "PotentialFieldLocalPlanner",
    "StepResult",
    "UnknownSpacePolicy",
    "incidence_angle",
]
