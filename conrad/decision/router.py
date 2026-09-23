"""Execution / tool router (ch16 'Tool router', ch28 execute_proposal).

An ACCEPTED proposal becomes exactly one typed downstream request. A rejected proposal becomes a
``RejectedAction`` record and nothing else. The router never produces wrench or thruster objects:
Model 1 says "inspect Region 4 again", Navigation determines the path.

implementation_status: FROZEN_CONTRACT (mapping)
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import Field

from conrad.decision.config import DecisionConfig
from conrad.decision.context import DecisionContext
from conrad.schemas.base import ConradModel
from conrad.schemas.belief import BeliefQuery
from conrad.schemas.decision import (
    ActionProposal,
    ActionType,
    ConstraintDecision,
    InformationNeed,
    NavigationGoal,
    QuestionType,
)
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Domain


class RouteTarget(str, Enum):
    MCBR = "MCBR"
    NAVIGATION = "NAVIGATION"
    BAAC = "BAAC"
    OPERATOR = "OPERATOR"
    BELIEF_BUS = "BELIEF_BUS"
    MISSION_EXECUTIVE = "MISSION_EXECUTIVE"
    SENSOR_MANAGER = "SENSOR_MANAGER"
    SAFETY_SUPERVISOR = "SAFETY_SUPERVISOR"


ROUTE_TABLE: dict[ActionType, RouteTarget] = {
    ActionType.CONTINUE_MISSION: RouteTarget.MISSION_EXECUTIVE,
    ActionType.QUERY_BELIEF: RouteTarget.BELIEF_BUS,
    ActionType.REQUEST_INFORMATION: RouteTarget.MCBR,
    ActionType.REPLAN: RouteTarget.MISSION_EXECUTIVE,
    ActionType.CHANGE_SENSOR_MODE: RouteTarget.SENSOR_MANAGER,
    ActionType.REVISIT_REGION: RouteTarget.NAVIGATION,
    ActionType.WAIT: RouteTarget.MISSION_EXECUTIVE,
    ActionType.TRANSMIT_INFORMATION: RouteTarget.BAAC,
    ActionType.STORE_AND_FORWARD: RouteTarget.BAAC,
    ActionType.ESCALATE_TO_OPERATOR: RouteTarget.OPERATOR,
    ActionType.RETURN_TO_SAFE_STATE: RouteTarget.SAFETY_SUPERVISOR,
    ActionType.ABORT_MISSION: RouteTarget.SAFETY_SUPERVISOR,
}


class RejectedAction(ConradModel):
    action_id: UUID
    action_type: ActionType
    reason_codes: tuple[str, ...] = Field(min_length=1)
    trace_id: UUID


class TransmissionRequest(ConradModel):
    """Communication INTENT only. BAAC decides representation, fidelity, timing and link (ch16)."""

    request_id: UUID
    trace_id: UUID
    belief_ids: tuple[UUID, ...]
    intent: str
    priority: float = Field(ge=0, le=1)
    store_and_forward: bool


class OperatorEscalation(ConradModel):
    """ESCALATE(reason, evidence, options) (ch17 'Human escalation')."""

    escalation_id: UUID
    trace_id: UUID
    reasons: tuple[str, ...]
    belief_ids: tuple[UUID, ...]
    claim_ids: tuple[UUID, ...]
    options: tuple[str, ...]
    priority: float = Field(ge=0, le=1)


class BeliefQueryRequest(ConradModel):
    request_id: UUID
    trace_id: UUID
    tool: str
    query: BeliefQuery


class ExecutiveDirective(ConradModel):
    """Typed, non-motion directive for the mission executive, sensor manager or safety supervisor."""

    directive_id: UUID
    trace_id: UUID
    target: RouteTarget
    action_type: ActionType
    parameters: dict[str, Any] = Field(default_factory=dict)


RoutedPayload = (
    InformationNeed
    | NavigationGoal
    | TransmissionRequest
    | OperatorEscalation
    | BeliefQueryRequest
    | ExecutiveDirective
    | RejectedAction
)


class RoutedAction(ConradModel):
    action_id: UUID
    target: RouteTarget | None
    payload: RoutedPayload


_TOOLS = {
    Domain.TECHNICAL: "query_model2t",
    Domain.ECOLOGICAL: "query_model2e",
    Domain.SPATIAL: "query_model2s",
}


class ExecutionRouter:
    def __init__(self, id_factory: IdFactory, config: DecisionConfig) -> None:
        self._ids = id_factory
        self.config = config

    def route(
        self, action: ActionProposal, decision: ConstraintDecision, ctx: DecisionContext
    ) -> RoutedAction:
        if decision.action_id != action.action_id:
            raise ValueError("constraint decision belongs to a different action")
        if not decision.accepted:
            rejected = RejectedAction(
                action_id=action.action_id,
                action_type=action.action_type,
                reason_codes=decision.reason_codes,
                trace_id=ctx.trace_id,
            )
            return RoutedAction(action_id=action.action_id, target=None, payload=rejected)
        target = ROUTE_TABLE[action.action_type]
        return RoutedAction(
            action_id=action.action_id, target=target, payload=self._payload(action, target, ctx)
        )

    def _payload(self, action: ActionProposal, target: RouteTarget, ctx: DecisionContext) -> RoutedPayload:
        p = action.parameters
        if target is RouteTarget.MCBR:
            question = QuestionType(str(p["question_type"]))
            cause = str(p.get("cause", ""))
            constraints: dict[str, Any] = {"cause": cause}
            for key in (
                "require_alternate_modality",
                "alternate_modalities",
                "cross_domain_disagreement",
                "calibration_check",
            ):
                if key in p:
                    constraints[key] = p[key]
            if action.target_region is not None:
                constraints["target_region"] = action.target_region.model_dump(mode="json")
            return InformationNeed(
                need_id=self._ids.new(),
                trace_id=ctx.trace_id,
                target_belief_ids=action.target_belief_ids,
                question_type=question,
                target_properties=tuple(str(x) for x in p.get("properties", ())),
                priority=action.priority,
                desired_uncertainty_reduction=dict(p.get("desired_uncertainty_reduction", {})),
                minimum_belief_revisions={
                    UUID(str(belief_id)): int(revision)
                    for belief_id, revision in dict(p.get("minimum_belief_revisions", {})).items()
                },
                minimum_independent_observation_counts={
                    UUID(str(belief_id)): int(count)
                    for belief_id, count in dict(p.get("minimum_independent_observation_counts", {})).items()
                },
                deadline_ns=p.get("deadline_ns"),
                constraints=constraints,
                originating_claim_ids=tuple(UUID(str(c)) for c in p.get("motivating_claims", ())),
            )
        if target is RouteTarget.NAVIGATION:
            return NavigationGoal(
                goal_id=self._ids.new(),
                trace_id=ctx.trace_id,
                target_region=action.target_region,
                position_tolerance_m=self.config.navigation_position_tolerance_m,
                orientation_tolerance_rad=self.config.navigation_orientation_tolerance_rad,
                risk_limit=self.config.constraints.risk_limit,
                source_action_id=action.action_id,
            )
        if target is RouteTarget.BAAC:
            return TransmissionRequest(
                request_id=self._ids.new(),
                trace_id=ctx.trace_id,
                belief_ids=action.target_belief_ids,
                intent=str(p.get("intent", "REPORT")),
                priority=action.priority,
                store_and_forward=action.action_type is ActionType.STORE_AND_FORWARD,
            )
        if target is RouteTarget.OPERATOR:
            return OperatorEscalation(
                escalation_id=self._ids.new(),
                trace_id=ctx.trace_id,
                reasons=tuple(str(r) for r in p.get("reason", ())),
                belief_ids=action.target_belief_ids,
                claim_ids=tuple(UUID(str(c)) for c in p.get("motivating_claims", ())),
                options=tuple(str(o) for o in p.get("options", ())),
                priority=action.priority,
            )
        if target is RouteTarget.BELIEF_BUS:
            domain = Domain(str(p["domain"]))
            return BeliefQueryRequest(
                request_id=self._ids.new(),
                trace_id=ctx.trace_id,
                tool=_TOOLS[domain],
                query=BeliefQuery(
                    domain=domain, belief_ids=action.target_belief_ids, include_provenance=True
                ),
            )
        return ExecutiveDirective(
            directive_id=self._ids.new(),
            trace_id=ctx.trace_id,
            target=target,
            action_type=action.action_type,
            parameters={k: v for k, v in p.items() if k != "motivating_claims"},
        )
