"""Knowledge gap K_i = (B_i, U_i, V_i, Q_i) derived from an InformationNeed (ch16, ch18).

implementation_status: FROZEN_CONTRACT (representation)
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from pydantic import Field

from conrad.active.config import MCBRConfig
from conrad.schemas.base import ConradModel
from conrad.schemas.belief import BeliefMessage
from conrad.schemas.decision import InformationNeed, QuestionType, UncertaintyType
from conrad.schemas.frames import SpatialSupport, Vec3
from conrad.schemas.uncertainty import Uncertainty

QUESTION_CHANNEL: dict[QuestionType, UncertaintyType | None] = {
    QuestionType.IMPROVE_MEASUREMENT: UncertaintyType.ALEATORIC,
    QuestionType.EXTEND_COVERAGE: UncertaintyType.OBSERVATIONAL,
    QuestionType.RESOLVE_CONTRADICTION: UncertaintyType.CONTRADICTION,
    QuestionType.DISCRIMINATE_HYPOTHESES: UncertaintyType.CONTRADICTION,
    QuestionType.CONFIRM_CONDITION: None,
}
CHANNEL_FIELD: dict[UncertaintyType, str] = {
    UncertaintyType.ALEATORIC: "aleatoric",
    UncertaintyType.EPISTEMIC: "epistemic",
    UncertaintyType.CONTRADICTION: "contradiction",
    UncertaintyType.OBSERVATIONAL: "observational",
}


class PriorView(ConradModel):
    """An observation already taken for this belief (from the evidence archive, not from truth)."""

    position_m: Vec3
    modality: str


class AbandonedView(ConradModel):
    """A view the mission ACCEPTED and did not fly, with the reason the runtime recorded.

    Belief plane: the commanded pose comes from the plan and the closest approach from the runtime's own
    state estimate (``conrad.orchestration.view_execution``). No truth, no twin state. A planner that reads
    it can stop re-offering a pose the mission has already failed to reach; a planner that ignores it
    behaves exactly as before, because the field defaults to empty.
    """

    position_m: Vec3
    aim_point_m: Vec3
    reason: str
    closest_approach_m: float = Field(
        default=float("inf"), description="closest the vehicle got to the commanded pose, in metres"
    )
    plan_id: UUID | None = None


class KnowledgeGap(ConradModel):
    need_id: UUID
    belief_id: UUID
    belief_revision: int = Field(ge=0)
    independent_observation_count: int = Field(ge=0)
    uncertainty: Uncertainty
    coverage: float = Field(ge=0, le=1, description="V_i: current observability/coverage")
    question: QuestionType
    causes: tuple[UncertaintyType, ...]
    targeted: tuple[UncertaintyType, ...]
    target_region: SpatialSupport
    properties: tuple[str, ...]
    hypotheses: tuple[str, ...] = ()
    prior_views: tuple[PriorView, ...] = ()
    provenance_refs: tuple[UUID, ...] = ()

    def level(self, channel: UncertaintyType) -> float:
        return float(getattr(self.uncertainty, CHANNEL_FIELD[channel]))


def _targets(need: InformationNeed, causes: tuple[UncertaintyType, ...]) -> tuple[UncertaintyType, ...]:
    explicit = tuple(
        t
        for t in UncertaintyType
        if CHANNEL_FIELD[t] in need.desired_uncertainty_reduction
        or t.value in need.desired_uncertainty_reduction
    )
    if explicit:
        return explicit
    channel = QUESTION_CHANNEL[need.question_type]
    if channel is not None:
        return (channel,)
    cause = need.constraints.get("cause")
    if isinstance(cause, str) and cause in UncertaintyType.__members__:
        return (UncertaintyType(cause),)
    return causes or (UncertaintyType.OBSERVATIONAL,)


def build_knowledge_gaps(
    need: InformationNeed,
    beliefs: Sequence[BeliefMessage],
    config: MCBRConfig,
    prior_views: Sequence[PriorView] = (),
) -> list[KnowledgeGap]:
    """One gap per target belief that is actually present. A missing belief yields no gap (caller escalates)."""
    by_id = {m.belief_id: m for m in beliefs}
    gaps: list[KnowledgeGap] = []
    for belief_id in need.target_belief_ids:
        m = by_id.get(belief_id)
        if m is None:
            continue
        region = m.spatial_support
        raw = need.constraints.get("target_region")
        if region is None and isinstance(raw, dict):
            region = SpatialSupport.model_validate(raw)
        if region is None:
            continue
        u = m.uncertainty
        levels = {
            UncertaintyType.ALEATORIC: u.aleatoric,
            UncertaintyType.EPISTEMIC: u.epistemic,
            UncertaintyType.CONTRADICTION: u.contradiction,
            UncertaintyType.OBSERVATIONAL: u.observational,
        }
        causes = tuple(
            t for t, v in sorted(levels.items(), key=lambda kv: -kv[1]) if v >= config.cause_threshold
        )
        coverage = m.spatial.coverage if m.spatial is not None else max(0.0, 1.0 - min(1.0, u.observational))
        hyp = need.constraints.get("hypotheses")
        hypotheses = tuple(str(h) for h in hyp) if isinstance(hyp, list | tuple) else ()
        if not hypotheses and QUESTION_CHANNEL[need.question_type] is UncertaintyType.CONTRADICTION:
            hypotheses = ("H1", "H2")
        gaps.append(
            KnowledgeGap(
                need_id=need.need_id,
                belief_id=belief_id,
                belief_revision=m.revision,
                independent_observation_count=m.independent_observation_count,
                uncertainty=u,
                coverage=coverage,
                question=need.question_type,
                causes=causes,
                targeted=_targets(need, causes),
                target_region=region,
                properties=need.target_properties,
                hypotheses=hypotheses,
                prior_views=tuple(prior_views),
                provenance_refs=m.provenance_refs,
            )
        )
    return gaps


def need_satisfied(need: InformationNeed, gaps: Sequence[KnowledgeGap], config: MCBRConfig) -> bool:
    """Whether every explicit freshness requirement and targeted uncertainty bound is satisfied."""
    if not gaps:
        return False
    for gap in gaps:
        minimum_revision = need.minimum_belief_revisions.get(gap.belief_id)
        if minimum_revision is not None and gap.belief_revision < minimum_revision:
            return False
        minimum_observations = need.minimum_independent_observation_counts.get(gap.belief_id)
        if minimum_observations is not None and gap.independent_observation_count < minimum_observations:
            return False
        for channel in gap.targeted:
            name = CHANNEL_FIELD[channel]
            target = need.desired_uncertainty_reduction.get(
                name, need.desired_uncertainty_reduction.get(channel.value, config.default_target_uncertainty)
            )
            if gap.level(channel) > float(target):
                return False
    return True
