"""Decision Claim Graph data structures and issue codes (ch16 'Decision Claim Graph', ch17 'Claim graph').

implementation_status: FROZEN_CONTRACT (node/edge types, grounding invariant)
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from pydantic import Field

from conrad.decision.config import DecisionConfig
from conrad.schemas.base import ConradModel
from conrad.schemas.belief import BeliefMessage, PropertyClaim
from conrad.schemas.decision import (
    WORLD_DEPENDENT_CLAIMS,
    ClaimEdge,
    ClaimEdgeType,
    ClaimType,
    DecisionClaim,
    GroundingStatus,
    UncertaintyType,
)
from conrad.schemas.uncertainty import Uncertainty

ISSUE_MISSING = "MISSING_BELIEF"
ISSUE_STALE = "STALE_BELIEF"
ISSUE_WRONG_ASSOCIATION = "WRONG_ASSOCIATION"
ISSUE_DOMAIN_UNAVAILABLE = "DOMAIN_UNAVAILABLE"
ISSUE_DOMAIN_DEGRADED = "DOMAIN_DEGRADED"
ISSUE_UNKNOWN_STATUS = "UNKNOWN_STATUS"
ISSUE_NO_EVIDENCE_PATH = "NO_EVIDENCE_PATH"
ISSUE_CROSS_DOMAIN = "CROSS_DOMAIN_DISAGREEMENT"
ISSUE_UNCALIBRATED = "UNCALIBRATED_SOURCE"
ISSUE_EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"


class RequirementAssessment(ConradModel):
    """Structured reasoning result for one requirement: what is known, uncertain, conflicting, missing."""

    requirement_id: UUID
    requirement_claim_id: UUID
    belief_claim_ids: tuple[UUID, ...]
    grounded_claim_ids: tuple[UUID, ...]
    unsupported_claim_ids: tuple[UUID, ...]
    target_belief_ids: tuple[UUID, ...]
    issues: tuple[str, ...]
    causes: tuple[UncertaintyType, ...]
    effective_uncertainty: Uncertainty | None
    consequence: float
    matters: bool
    satisfied: bool
    calibration_only_epistemic: bool = Field(
        default=False,
        description="U_E exceeds its threshold ONLY because of the uncalibrated-source floor (not OOD evidence)",
    )


class ClaimGraph:
    """Bounded temporary graph G_t^D = (V_D, E_D). It is working state, not a world model."""

    def __init__(self, max_nodes: int) -> None:
        self.max_nodes = max_nodes
        self.claims: list[DecisionClaim] = []
        self.edges: list[ClaimEdge] = []
        self.assessments: list[RequirementAssessment] = []
        self.route_blocking_claim_ids: list[UUID] = []  # grounded obstacle claims on the planned route
        self.dropped_for_budget = 0
        self._by_id: dict[UUID, DecisionClaim] = {}

    def add(self, claim: DecisionClaim) -> bool:
        if len(self.claims) >= self.max_nodes:
            self.dropped_for_budget += 1
            return False
        self.claims.append(claim)
        self._by_id[claim.claim_id] = claim
        return True

    def connect(self, source: UUID, target: UUID, edge_type: ClaimEdgeType) -> None:
        if source in self._by_id and target in self._by_id and source != target:
            self.edges.append(ClaimEdge(source_claim_id=source, target_claim_id=target, edge_type=edge_type))

    def get(self, claim_id: UUID) -> DecisionClaim | None:
        return self._by_id.get(claim_id)

    def of_type(self, claim_type: ClaimType) -> list[DecisionClaim]:
        return [c for c in self.claims if c.claim_type is claim_type]

    def world_claims(self) -> list[DecisionClaim]:
        return [c for c in self.claims if c.claim_type in WORLD_DEPENDENT_CLAIMS]

    def unsupported_ids(self) -> tuple[UUID, ...]:
        return tuple(c.claim_id for c in self.world_claims() if c.grounding is GroundingStatus.UNSUPPORTED)

    def assessment_for(self, requirement_id: UUID) -> RequirementAssessment | None:
        for a in self.assessments:
            if a.requirement_id == requirement_id:
                return a
        return None


def _prop(message: BeliefMessage, name: str) -> PropertyClaim | None:
    return next((c for c in message.state_summary if c.name == name), None)


def diagnose_causes(u: Uncertainty, config: DecisionConfig) -> tuple[UncertaintyType, ...]:
    """Which channels exceed their configured threshold, strongest excess first."""
    t = config.thresholds
    pairs = [
        (UncertaintyType.ALEATORIC, u.aleatoric - t.aleatoric),
        (UncertaintyType.EPISTEMIC, u.epistemic - t.epistemic),
        (UncertaintyType.CONTRADICTION, u.contradiction - t.contradiction),
        (UncertaintyType.OBSERVATIONAL, u.observational - t.observational),
    ]
    over = [p for p in pairs if p[1] >= 0]
    over.sort(key=lambda p: -p[1])
    return tuple(p[0] for p in over)


def _max_uncertainty(items: Iterable[Uncertainty]) -> Uncertainty | None:
    rows = [u.as_tuple() for u in items]
    if not rows:
        return None
    return Uncertainty(
        aleatoric=max(r[0] for r in rows),
        epistemic=max(r[1] for r in rows),
        contradiction=max(r[2] for r in rows),
        observational=max(r[3] for r in rows),
    )
