"""Model 1 / EGDC: Evidence-Grounded Deliberative Controller (spec ch16, ch17, ch28, ch33).

DECISION PLANE. Consumes structured beliefs from the Belief Bus; never twin truth, never raw sensor
data. Proposes actions inside a frozen vocabulary; a deterministic constraint engine decides
permission; the router hands accepted actions to MCBR / Navigation / BAAC / the operator.
"""

from conrad.decision.actions import ACTION_VOCABULARY, CandidateActionGenerator, question_for_cause
from conrad.decision.claims import (
    ClaimGraph,
    ClaimGraphBuilder,
    RequirementAssessment,
    diagnose_causes,
)
from conrad.decision.config import (
    ConstraintConfig,
    DecisionConfig,
    UncertaintyThresholds,
    UtilityWeights,
    load_config,
)
from conrad.decision.consequence import ConsequenceConfig, ConsequenceEstimator, ConsequenceVector
from conrad.decision.constraints import ConstraintEngine, validate_claim_support
from conrad.decision.context import DecisionContext, DecisionSummary, MissionRequirement
from conrad.decision.egdc import EGDC, DecisionOutcome
from conrad.decision.monitor import OutcomeMonitor, OutcomeRecord
from conrad.decision.policy import DecisionPolicy, NaiveActOnClaimsPolicy, StructuredReasoningPolicy
from conrad.decision.query_engine import BeliefQueryEngine
from conrad.decision.router import (
    ROUTE_TABLE,
    BeliefQueryRequest,
    ExecutionRouter,
    ExecutiveDirective,
    OperatorEscalation,
    RejectedAction,
    RoutedAction,
    RouteTarget,
    TransmissionRequest,
)
from conrad.decision.uir import UIRReport, uir_report, unsupported_inference_rate

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "frozen_boundaries": [
        "ActionType vocabulary",
        "grounding invariant Decision->Claim->Belief->Evidence",
        "deterministic ConstraintEngine (never optimised)",
        "router mapping",
        "DecisionRecord + SourceType.DECISION provenance",
    ],
    "source_sections": [
        "ch16",
        "ch17",
        "ch28 Runtime and Safety Contract",
        "ch33 Model 1 EGDC exact implementation",
    ],
    "configuration_keys": [
        "DecisionConfig.thresholds.*",
        "DecisionConfig.constraints.*",
        "DecisionConfig.utility.*",
        "DecisionConfig.max_claim_nodes",
        "DecisionConfig.consequence_matters_above",
        "DecisionConfig.escalate_consequence_above",
        "ConsequenceConfig.costs/resolvability",
        "EGDCScorerConfig.* (conrad.decision.learned)",
    ],
    "assumptions": [
        "uncertainty thresholds, action costs and resolvability priors are ENGINEERING_ESTIMATE simulation defaults",
        "UIR counts world claims a decision RELIED ON; honestly labelled UNSUPPORTED claims are not inferences",
        "an OBSERVED belief without evidence_support has no evidence path and is UNSUPPORTED",
        "uncalibrated beliefs on consequential requirements get a configured epistemic floor",
    ],
    "baselines": ["NaiveActOnClaimsPolicy (M1-UIR-E001)", "StructuredReasoningPolicy", "LearnedEGDCPolicy"],
    "acceptance_tests": ["tests/unit/decision", "tests/property/decision", "M1-UIR-E001"],
    "claim_status": "EVALUATED",
}

__all__ = [
    "ACTION_VOCABULARY",
    "EGDC",
    "IMPLEMENTATION_METADATA",
    "ROUTE_TABLE",
    "BeliefQueryEngine",
    "BeliefQueryRequest",
    "CandidateActionGenerator",
    "ClaimGraph",
    "ClaimGraphBuilder",
    "ConsequenceConfig",
    "ConsequenceEstimator",
    "ConsequenceVector",
    "ConstraintConfig",
    "ConstraintEngine",
    "DecisionConfig",
    "DecisionContext",
    "DecisionOutcome",
    "DecisionPolicy",
    "DecisionSummary",
    "ExecutionRouter",
    "ExecutiveDirective",
    "MissionRequirement",
    "NaiveActOnClaimsPolicy",
    "OperatorEscalation",
    "OutcomeMonitor",
    "OutcomeRecord",
    "RejectedAction",
    "RequirementAssessment",
    "RouteTarget",
    "RoutedAction",
    "StructuredReasoningPolicy",
    "TransmissionRequest",
    "UIRReport",
    "UncertaintyThresholds",
    "UtilityWeights",
    "diagnose_causes",
    "load_config",
    "question_for_cause",
    "uir_report",
    "unsupported_inference_rate",
    "validate_claim_support",
]
