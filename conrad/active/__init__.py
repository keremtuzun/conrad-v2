"""MCBR: Mission-Conditioned Belief Resolution, active information acquisition (spec ch16, ch18, ch33).

DECISION PLANE. Input: InformationNeed + beliefs + belief-derived callables. Output: ObservationPlan.
MCBR never moves the robot and never produces motion or actuator commands.
"""

from conrad.active.baselines import make_planners
from conrad.active.candidates import (
    FeasibilityFilter,
    MissionBounds,
    RawCandidate,
    SensorOption,
    ViewpointGenerator,
    look_at,
)
from conrad.active.config import CostWeights, MCBRConfig
from conrad.active.eig import GainBreakdown, InformationGainEstimator
from conrad.active.gap import KnowledgeGap, PriorView, build_knowledge_gaps, need_satisfied
from conrad.active.planner import (
    MCBRPlanner,
    ObservationPlanner,
    PlanningRequest,
    PlanResult,
    ScoredCandidate,
)

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": ["ch16 Active Intelligence", "ch18", "ch33 MCBR exact implementation"],
    "configuration_keys": ["MCBRConfig.*", "MCBRRankerConfig.* (conrad.active.learned)"],
    "assumptions": [
        "per-cause analytic gain terms and sensor quality/discrimination numbers are ENGINEERING_ESTIMATE",
        "prior views come from the evidence archive supplied by the caller",
        "feasibility filter is shared by every baseline (safety rule, not an MCBR feature)",
    ],
    "baselines": [
        "A-B0 random",
        "A-B1 fixed inspection",
        "A-B2 coverage",
        "A-B3 frontier",
        "A-B4 geometric NBV",
        "A-B5 entropy NBV",
        "A-B6 standard EIG",
        "A-B7 uncertainty NBV",
        "A-B9 MCBR without mission conditioning",
        "A-B10 full MCBR",
    ],
    "open": ["A-B8 RL active perception baseline not implemented"],
    "acceptance_tests": ["tests/unit/active", "ACTIVE-MCBR-E001"],
    "claim_status": "EVALUATED",
}

__all__ = [
    "IMPLEMENTATION_METADATA",
    "CostWeights",
    "FeasibilityFilter",
    "GainBreakdown",
    "InformationGainEstimator",
    "KnowledgeGap",
    "MCBRConfig",
    "MCBRPlanner",
    "MissionBounds",
    "ObservationPlanner",
    "PlanResult",
    "PlanningRequest",
    "PriorView",
    "RawCandidate",
    "ScoredCandidate",
    "SensorOption",
    "ViewpointGenerator",
    "build_knowledge_gaps",
    "look_at",
    "make_planners",
    "need_satisfied",
]
