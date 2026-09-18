"""Candidate action generation from deterministic templates (ch16 'Candidate generation').

Actions come ONLY from the frozen ``ActionType`` vocabulary. The generator never emits motor
commands, poses for thrusters or free text commands. Uncertainty cause decides the information need:

    high U_A -> IMPROVE_MEASUREMENT      high U_O -> EXTEND_COVERAGE
    high U_C -> RESOLVE_CONTRADICTION / DISCRIMINATE_HYPOTHESES
    high U_E -> alternate evidence (other modality) or ESCALATE_TO_OPERATOR

implementation_status: FROZEN_CONTRACT (vocabulary) / EXPERIMENTAL_CANDIDATE (templates)
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from conrad.decision.claims import (
    ISSUE_CROSS_DOMAIN,
    ISSUE_DOMAIN_UNAVAILABLE,
    ISSUE_EVIDENCE_CONFLICT,
    ISSUE_MISSING,
    ISSUE_STALE,
    ISSUE_WRONG_ASSOCIATION,
    ClaimGraph,
    RequirementAssessment,
)
from conrad.decision.config import DecisionConfig
from conrad.decision.context import DecisionContext, MissionRequirement
from conrad.schemas.comms import LinkStatus
from conrad.schemas.decision import ActionProposal, ActionType, QuestionType, UncertaintyType
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import HealthLevel

ACTION_VOCABULARY: frozenset[ActionType] = frozenset(ActionType)

INFORMATION_ACTIONS = frozenset(
    {
        ActionType.QUERY_BELIEF,
        ActionType.REQUEST_INFORMATION,
        ActionType.REVISIT_REGION,
        ActionType.CHANGE_SENSOR_MODE,
    }
)
NON_ADVANCING_ACTIONS = frozenset(
    {
        ActionType.WAIT,
        ActionType.ESCALATE_TO_OPERATOR,
        ActionType.RETURN_TO_SAFE_STATE,
        ActionType.ABORT_MISSION,
        ActionType.STORE_AND_FORWARD,
    }
)


def question_for_cause(cause: UncertaintyType, has_evidence_conflict: bool) -> QuestionType | None:
    """The cause-specific sensing response. EPISTEMIC has no generic autonomous question (returns None)."""
    if cause is UncertaintyType.ALEATORIC:
        return QuestionType.IMPROVE_MEASUREMENT
    if cause is UncertaintyType.OBSERVATIONAL:
        return QuestionType.EXTEND_COVERAGE
    if cause is UncertaintyType.CONTRADICTION:
        return (
            QuestionType.RESOLVE_CONTRADICTION
            if has_evidence_conflict
            else QuestionType.DISCRIMINATE_HYPOTHESES
        )
    return None


class CandidateActionGenerator:
    def __init__(self, id_factory: IdFactory, config: DecisionConfig) -> None:
        self._ids = id_factory
        self.config = config

    def generate(self, graph: ClaimGraph, ctx: DecisionContext) -> list[ActionProposal]:
        out: list[ActionProposal] = []
        requirements = {r.requirement_id: r for r in ctx.requirements}
        relied: list[UUID] = []
        for a in graph.assessments:
            relied.extend(a.grounded_claim_ids if self.config.enforce_grounding else a.belief_claim_ids)
        if not self.config.enforce_grounding:
            # naive arm: every belief claim, including context claims, is taken at face value
            relied = [c.claim_id for c in graph.world_claims()]
        out.append(self._make(ActionType.CONTINUE_MISSION, supporting=tuple(relied)))
        for assessment in graph.assessments:
            if assessment.satisfied or not assessment.matters:
                continue
            out.extend(self._for_assessment(assessment, requirements[assessment.requirement_id], ctx))
        out.append(self._make(ActionType.WAIT, parameters={"duration_s": 5.0}))
        out.append(
            self._make(
                ActionType.ESCALATE_TO_OPERATOR,
                parameters={
                    "reason": sorted({i for a in graph.assessments if not a.satisfied for i in a.issues})
                    + sorted({c.value for a in graph.assessments if not a.satisfied for c in a.causes}),
                    "options": [
                        t.value for t in (ActionType.CONTINUE_MISSION, ActionType.RETURN_TO_SAFE_STATE)
                    ],
                    "motivating_claims": [str(c) for a in graph.assessments for c in a.belief_claim_ids],
                },
            )
        )
        out.append(self._make(ActionType.RETURN_TO_SAFE_STATE))
        unhealthy = ctx.system_health is not None and ctx.system_health.overall is HealthLevel.FAULT
        if unhealthy:
            out.append(self._make(ActionType.ABORT_MISSION, parameters={"reason": "SYSTEM_HEALTH_FAULT"}))
        pending = ctx.mission.notes.get("pending_report_belief_ids")
        if pending:
            ids = tuple(UUID(str(p)) for p in pending)
            link_up = ctx.link_state is not None and ctx.link_state.status is not LinkStatus.DOWN
            kind = ActionType.TRANSMIT_INFORMATION if link_up else ActionType.STORE_AND_FORWARD
            out.append(self._make(kind, targets=ids, parameters={"intent": "REPORT_FINDING"}))
        if any(a.matters and not a.satisfied for a in graph.assessments) and self._exhausted(graph, ctx):
            out.append(self._make(ActionType.REPLAN, parameters={"reason": "INFORMATION_ATTEMPTS_EXHAUSTED"}))
        assert all(a.action_type in ACTION_VOCABULARY for a in out)
        return out

    # ------------------------------------------------------------------ templates
    def _for_assessment(
        self, a: RequirementAssessment, req: MissionRequirement, ctx: DecisionContext
    ) -> list[ActionProposal]:
        out: list[ActionProposal] = []
        motivating = [str(c) for c in a.belief_claim_ids]
        base: dict[str, Any] = {"requirement_id": str(a.requirement_id), "motivating_claims": motivating}
        retrieval_issue = {ISSUE_MISSING, ISSUE_STALE, ISSUE_WRONG_ASSOCIATION, ISSUE_DOMAIN_UNAVAILABLE}
        found = retrieval_issue.intersection(a.issues)
        if found:
            out.append(
                self._make(
                    ActionType.QUERY_BELIEF,
                    targets=a.target_belief_ids,
                    parameters={**base, "domain": req.domain.value, "issues": sorted(found)},
                )
            )
            if req.region is not None:
                out.append(
                    self._make(
                        ActionType.REVISIT_REGION,
                        targets=a.target_belief_ids,
                        region=req,
                        parameters={**base, "issues": sorted(found)},
                    )
                )
        conflict = ISSUE_EVIDENCE_CONFLICT in a.issues
        for cause in a.causes:
            if not a.target_belief_ids:
                break
            question = question_for_cause(cause, conflict)
            params: dict[str, Any] = {**base, "cause": cause.value, "properties": list(req.properties)}
            if cause is UncertaintyType.EPISTEMIC:
                used = set(ctx.mission.notes.get("modalities_used", []))
                alternates = [m for m in ctx.available_modalities if m not in used]
                if not alternates:
                    continue  # no autonomous resolution: ESCALATE stays the legitimate option
                question = QuestionType.CONFIRM_CONDITION
                params["require_alternate_modality"] = True
                params["alternate_modalities"] = alternates
            if cause is UncertaintyType.ALEATORIC and len(ctx.available_modalities) > 1:
                out.append(
                    self._make(
                        ActionType.CHANGE_SENSOR_MODE,
                        targets=a.target_belief_ids,
                        parameters={
                            **base,
                            "cause": cause.value,
                            "modalities": list(ctx.available_modalities),
                        },
                    )
                )
            if cause is UncertaintyType.OBSERVATIONAL and ISSUE_CROSS_DOMAIN in a.issues:
                params["cross_domain_disagreement"] = True
            assert question is not None
            params["question_type"] = question.value
            out.append(
                self._make(
                    ActionType.REQUEST_INFORMATION,
                    targets=a.target_belief_ids,
                    region=req,
                    parameters=params,
                    supporting=a.grounded_claim_ids if cause is not UncertaintyType.EPISTEMIC else (),
                )
            )
        return out

    def _exhausted(self, graph: ClaimGraph, ctx: DecisionContext) -> bool:
        for a in graph.assessments:
            if a.matters and not a.satisfied and a.target_belief_ids:
                tries = ctx.attempts_on(a.target_belief_ids, ActionType.REQUEST_INFORMATION)
                if tries >= self.config.max_information_attempts:
                    return True
        return False

    def _make(
        self,
        action_type: ActionType,
        targets: tuple[UUID, ...] = (),
        region: MissionRequirement | None = None,
        parameters: dict[str, Any] | None = None,
        supporting: tuple[UUID, ...] = (),
    ) -> ActionProposal:
        return ActionProposal(
            action_id=self._ids.new(),
            action_type=action_type,
            target_belief_ids=targets,
            target_region=None if region is None else region.region,
            parameters=parameters or {},
            supporting_claims=supporting,
        )
