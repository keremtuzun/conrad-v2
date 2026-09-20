"""Decision Claim Graph (ch16 'Decision Claim Graph' / 'Grounding invariant', ch17 'Claim graph').

Grounding is mechanical, not a model output: a world-dependent claim is GROUNDED only if a path
Decision -> Claim -> Belief -> Evidence exists in the snapshot. Anything else is UNSUPPORTED and can
never be used as a fact (enforced again by the ConstraintEngine).

implementation_status: FROZEN_CONTRACT (node/edge types, invariant); assessment rules EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from uuid import UUID

from conrad.decision.config import DecisionConfig
from conrad.decision.context import DecisionContext, MissionRequirement
from conrad.decision.cross_domain import cross_domain_disagreement
from conrad.decision.graph import (  # re-exported: public names live in conrad.decision.graph
    ISSUE_CROSS_DOMAIN,
    ISSUE_DOMAIN_DEGRADED,
    ISSUE_DOMAIN_UNAVAILABLE,
    ISSUE_EVIDENCE_CONFLICT,
    ISSUE_MISSING,
    ISSUE_NO_EVIDENCE_PATH,
    ISSUE_STALE,
    ISSUE_UNCALIBRATED,
    ISSUE_UNKNOWN_STATUS,
    ISSUE_WRONG_ASSOCIATION,
    ClaimGraph,
    RequirementAssessment,
    _max_uncertainty,
    _prop,
    diagnose_causes,
)
from conrad.decision.route import OCCUPANCY_PROPERTY, route_blockers
from conrad.schemas.belief import Availability, BeliefMessage, KnowledgeStatus
from conrad.schemas.decision import (
    ActionProposal,
    ActionType,
    ClaimEdgeType,
    ClaimType,
    DecisionClaim,
    GroundingStatus,
    UncertaintyType,
)
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import NS_PER_S
from conrad.schemas.uncertainty import Uncertainty


class ClaimGraphBuilder:
    def __init__(self, id_factory: IdFactory, config: DecisionConfig) -> None:
        self._ids = id_factory
        self.config = config

    # ------------------------------------------------------------------ public
    def build(self, ctx: DecisionContext) -> ClaimGraph:
        graph = ClaimGraph(self.config.max_claim_nodes)
        self._state_nodes(ctx, graph)
        for requirement in sorted(ctx.requirements, key=lambda r: (-r.consequence, r.requirement_id.int)):
            self._requirement(ctx, requirement, graph)
        self._route(ctx, graph)
        for need in ctx.active_information_needs:
            graph.add(
                DecisionClaim(
                    claim_id=self._ids.new(),
                    claim_type=ClaimType.INFORMATION_NEED,
                    statement=f"active information need {need.question_type.value}",
                    structured_value={"need_id": str(need.need_id), "priority": need.priority},
                    timestamp=ctx.timestamp,
                    grounding=GroundingStatus.NOT_WORLD_DEPENDENT,
                )
            )
        return graph

    def attach_actions(self, graph: ClaimGraph, ctx: DecisionContext, actions: list[ActionProposal]) -> None:
        """CANDIDATE_ACTION / EXPECTED_OUTCOME nodes and their DEPENDS_ON / RESOLVES / AFFECTS edges."""
        resources = graph.of_type(ClaimType.RESOURCE_STATE)
        for action in actions:
            node = DecisionClaim(
                claim_id=action.action_id,
                claim_type=ClaimType.CANDIDATE_ACTION,
                statement=action.action_type.value,
                structured_value={"action_type": action.action_type.value},
                timestamp=ctx.timestamp,
                grounding=GroundingStatus.NOT_WORLD_DEPENDENT,
            )
            if not graph.add(node):
                continue
            for claim_id in action.supporting_claims:
                graph.connect(action.action_id, claim_id, ClaimEdgeType.DEPENDS_ON)
            if action.action_type is ActionType.CONTINUE_MISSION:
                for claim_id in graph.route_blocking_claim_ids:
                    graph.connect(claim_id, action.action_id, ClaimEdgeType.BLOCKS)
            for claim_id in action.parameters.get("motivating_claims", ()):
                graph.connect(action.action_id, UUID(str(claim_id)), ClaimEdgeType.RESOLVES)
            for res in resources:
                graph.connect(res.claim_id, action.action_id, ClaimEdgeType.AFFECTS)
            if action.expected_outcome:
                outcome = DecisionClaim(
                    claim_id=self._ids.new(),
                    claim_type=ClaimType.EXPECTED_OUTCOME,
                    statement=f"expected outcome of {action.action_type.value}",
                    structured_value=dict(action.expected_outcome),
                    timestamp=ctx.timestamp,
                    grounding=GroundingStatus.NOT_WORLD_DEPENDENT,
                )
                if graph.add(outcome):
                    graph.connect(outcome.claim_id, action.action_id, ClaimEdgeType.DEPENDS_ON)

    # ------------------------------------------------------------------ internals
    def _route(self, ctx: DecisionContext, graph: ClaimGraph) -> None:
        """Grounded obstacle claims on the planned route (ch17 'Replanning'). They BLOCK continuing."""
        for m in route_blockers(ctx, self.config):
            prop = _prop(m, OCCUPANCY_PROPERTY)
            node = DecisionClaim(
                claim_id=self._ids.new(),
                claim_type=ClaimType.BELIEF_CLAIM,
                statement=f"{m.domain.value} occupancy of belief {m.belief_id} on the planned route",
                structured_value={
                    "property": OCCUPANCY_PROPERTY,
                    "value": True,
                    "knowledge_status": m.knowledge_status.value,
                    "domain": m.domain.value,
                    "route_blocking": True,
                    "issues": [],
                    "stale": False,
                    "provenance_refs": [str(p) for p in m.provenance_refs],
                },
                source_belief_ids=(m.belief_id,),
                source_belief_revisions=(m.revision,),
                evidence_refs=m.evidence_support,
                uncertainty=prop.uncertainty if prop is not None else m.uncertainty,
                timestamp=m.timestamp,
                grounding=GroundingStatus.GROUNDED,  # route_blockers() only returns evidence-backed beliefs
            )
            if graph.add(node):
                graph.route_blocking_claim_ids.append(node.claim_id)

    def _state_nodes(self, ctx: DecisionContext, graph: ClaimGraph) -> None:
        if ctx.resource_state is not None:
            graph.add(
                DecisionClaim(
                    claim_id=self._ids.new(),
                    claim_type=ClaimType.RESOURCE_STATE,
                    statement="resource state",
                    structured_value=ctx.resource_state.model_dump(mode="json", exclude={"timestamp"}),
                    timestamp=ctx.resource_state.timestamp,
                    grounding=GroundingStatus.NOT_WORLD_DEPENDENT,
                )
            )
        if ctx.robot_state is not None:
            graph.add(
                DecisionClaim(
                    claim_id=self._ids.new(),
                    claim_type=ClaimType.ROBOT_STATE,
                    statement="estimated robot state",
                    structured_value={
                        "estimator_health": ctx.robot_state.estimator_health.value,
                        "position_m": list(ctx.robot_state.pose.position_m),
                        "frame_id": ctx.robot_state.pose.frame_id,
                    },
                    timestamp=ctx.robot_state.timestamp,
                    grounding=GroundingStatus.NOT_WORLD_DEPENDENT,
                )
            )
        constraints = {
            "battery_reserve_fraction": self.config.constraints.battery_reserve_fraction,
            "risk_limit": self.config.constraints.risk_limit,
            "motion_permitted": ctx.motion_permitted,
        }
        graph.add(
            DecisionClaim(
                claim_id=self._ids.new(),
                claim_type=ClaimType.CONSTRAINT,
                statement="hard constraints in force",
                structured_value=constraints,
                timestamp=ctx.timestamp,
                grounding=GroundingStatus.NOT_WORLD_DEPENDENT,
            )
        )

    def _requirement(self, ctx: DecisionContext, req: MissionRequirement, graph: ClaimGraph) -> None:
        req_node = DecisionClaim(
            claim_id=self._ids.new(),
            claim_type=ClaimType.MISSION_REQUIREMENT,
            statement=req.description,
            structured_value={
                "requirement_id": str(req.requirement_id),
                "domain": req.domain.value,
                "consequence": req.consequence,
                "properties": list(req.properties),
            },
            timestamp=ctx.timestamp,
            grounding=GroundingStatus.NOT_WORLD_DEPENDENT,
        )
        if not graph.add(req_node):
            return
        issues: list[str] = []
        availability = ctx.availability(req.domain)
        if availability is Availability.UNAVAILABLE:
            issues.append(ISSUE_DOMAIN_UNAVAILABLE)
        elif availability is Availability.DEGRADED:
            issues.append(ISSUE_DOMAIN_DEGRADED)

        primary = self._primary_messages(ctx, req)
        claim_ids: list[UUID] = []
        grounded: list[UUID] = []
        unsupported: list[UUID] = []
        uncertainties: list[Uncertainty] = []
        target_ids: list[UUID] = list(req.target_belief_ids)

        if not primary:
            issues.append(ISSUE_MISSING)
            claim = self._claim(
                ctx, req, None, None, GroundingStatus.UNSUPPORTED, (ISSUE_MISSING,), stale=False
            )
            if graph.add(claim):
                claim_ids.append(claim.claim_id)
                unsupported.append(claim.claim_id)
                graph.connect(req_node.claim_id, claim.claim_id, ClaimEdgeType.REQUIRES)

        for message in primary:
            if message.belief_id not in target_ids:
                target_ids.append(message.belief_id)
            m_issues = self._message_issues(ctx, req, message, availability)
            stale = ISSUE_STALE in m_issues
            hard = {ISSUE_WRONG_ASSOCIATION, ISSUE_UNKNOWN_STATUS, ISSUE_NO_EVIDENCE_PATH}
            names = req.properties or (None,)
            for name in names:
                prop = None if name is None else _prop(message, name)
                p_issues = list(m_issues)
                if name is not None and (prop is None or prop.status is KnowledgeStatus.UNKNOWN):
                    p_issues.append(ISSUE_UNKNOWN_STATUS)
                status = (
                    GroundingStatus.UNSUPPORTED if hard.intersection(p_issues) else GroundingStatus.GROUNDED
                )
                claim = self._claim(ctx, req, message, name, status, tuple(p_issues), stale=stale)
                if not graph.add(claim):
                    continue
                claim_ids.append(claim.claim_id)
                graph.connect(req_node.claim_id, claim.claim_id, ClaimEdgeType.REQUIRES)
                (grounded if status is GroundingStatus.GROUNDED else unsupported).append(claim.claim_id)
                issues.extend(i for i in p_issues if i not in issues)
                if status is GroundingStatus.GROUNDED:
                    uncertainties.append(prop.uncertainty if prop is not None else message.uncertainty)
                    uncertainties.append(message.uncertainty)
                elif not hard.intersection(p_issues) - {ISSUE_UNKNOWN_STATUS}:
                    # The belief exists and honestly says "not observed": its uncertainty (U_O) still
                    # diagnoses WHY the requirement is open, so an information need can be raised. The
                    # claim itself stays UNSUPPORTED and is never relied on as fact.
                    uncertainties.append(prop.uncertainty if prop is not None else message.uncertainty)

        disagreeing = self._cross_domain(ctx, req, primary, graph, claim_ids)
        if disagreeing:
            issues.append(ISSUE_CROSS_DOMAIN)

        raw = _max_uncertainty(uncertainties)
        confirmed = ISSUE_UNCALIBRATED in issues and _confirmed_after_request(ctx, req, primary)
        effective = None if raw is None else self._effective(raw, issues, req, confirmed)
        causes = () if effective is None else diagnose_causes(effective, self.config)
        # U_E over threshold only because of the uncalibrated-source floor: a calibration gap, not OOD evidence.
        # An independent confirming observation can close it, so it must not be read as "no autonomous path".
        calibration_only = (
            raw is not None
            and effective is not None
            and UncertaintyType.EPISTEMIC in causes
            and raw.epistemic < self.config.thresholds.epistemic
        )
        if ISSUE_CROSS_DOMAIN in issues and UncertaintyType.OBSERVATIONAL not in causes:
            causes = (*causes, UncertaintyType.OBSERVATIONAL)
        matters = req.consequence >= self.config.consequence_matters_above
        blocking = [i for i in issues if i not in (ISSUE_DOMAIN_DEGRADED, ISSUE_UNCALIBRATED)]
        satisfied = bool(grounded) and not unsupported and not blocking and not causes
        if satisfied:
            for claim_id in grounded:
                graph.connect(claim_id, req_node.claim_id, ClaimEdgeType.SUPPORTS)
        graph.assessments.append(
            RequirementAssessment(
                requirement_id=req.requirement_id,
                requirement_claim_id=req_node.claim_id,
                belief_claim_ids=tuple(claim_ids),
                grounded_claim_ids=tuple(grounded),
                unsupported_claim_ids=tuple(unsupported),
                target_belief_ids=tuple(target_ids),
                issues=tuple(issues),
                causes=causes,
                effective_uncertainty=effective,
                consequence=req.consequence,
                matters=matters,
                satisfied=satisfied,
                calibration_only_epistemic=calibration_only,
            )
        )

    def _primary_messages(self, ctx: DecisionContext, req: MissionRequirement) -> list[BeliefMessage]:
        out = []
        for m in ctx.beliefs(req.domain):
            by_belief = m.belief_id in req.target_belief_ids
            by_entity = m.world_entity_id is not None and m.world_entity_id in req.target_entity_ids
            untargeted = not req.target_belief_ids and not req.target_entity_ids
            if by_belief or by_entity or untargeted:
                out.append(m)
        return out

    def _message_issues(
        self, ctx: DecisionContext, req: MissionRequirement, m: BeliefMessage, availability: Availability
    ) -> list[str]:
        issues: list[str] = []
        max_age = req.max_age_s if req.max_age_s is not None else self.config.constraints.max_belief_age_s
        age_s = (ctx.timestamp.time_ns - m.timestamp.time_ns) / NS_PER_S
        listed_stale = str(m.belief_id) in ctx.snapshot.provenance.get("stale_belief_ids", [])
        if age_s > max_age or listed_stale or availability is Availability.STALE:
            issues.append(ISSUE_STALE)
        if (
            req.target_entity_ids
            and m.world_entity_id is not None
            and m.world_entity_id not in req.target_entity_ids
        ):
            issues.append(ISSUE_WRONG_ASSOCIATION)
        if m.knowledge_status is KnowledgeStatus.UNKNOWN:
            issues.append(ISSUE_UNKNOWN_STATUS)
        if m.knowledge_status is KnowledgeStatus.OBSERVED and not m.evidence_support:
            issues.append(ISSUE_NO_EVIDENCE_PATH)
        if m.evidence_conflicts:
            issues.append(ISSUE_EVIDENCE_CONFLICT)
        if not m.uncertainty.calibration_metadata.calibrated:
            issues.append(ISSUE_UNCALIBRATED)
        return issues

    def _claim(
        self,
        ctx: DecisionContext,
        req: MissionRequirement,
        m: BeliefMessage | None,
        prop_name: str | None,
        grounding: GroundingStatus,
        issues: tuple[str, ...],
        stale: bool,
    ) -> DecisionClaim:
        prop = None if (m is None or prop_name is None) else _prop(m, prop_name)
        value = None if prop is None else prop.value
        subject = "no belief available" if m is None else f"belief {m.belief_id}"
        return DecisionClaim(
            claim_id=self._ids.new(),
            claim_type=ClaimType.BELIEF_CLAIM,
            statement=f"{req.domain.value} {prop_name or 'state'} of {subject}",
            structured_value={
                "requirement_id": str(req.requirement_id),
                "property": prop_name,
                "value": value,
                "knowledge_status": None if m is None else m.knowledge_status.value,
                "domain": req.domain.value,
                "issues": list(issues),
                "stale": stale,
                "provenance_refs": [] if m is None else [str(p) for p in m.provenance_refs],
            },
            source_belief_ids=() if m is None else (m.belief_id,),
            source_belief_revisions=() if m is None else (m.revision,),
            evidence_refs=() if m is None else m.evidence_support,
            uncertainty=None if m is None else (prop.uncertainty if prop is not None else m.uncertainty),
            timestamp=ctx.timestamp if m is None else m.timestamp,
            grounding=grounding,
        )

    def _cross_domain(
        self,
        ctx: DecisionContext,
        req: MissionRequirement,
        primary: list[BeliefMessage],
        graph: ClaimGraph,
        primary_claim_ids: list[UUID],
    ) -> bool:
        return cross_domain_disagreement(self._ids, self.config, ctx, req, primary, graph, primary_claim_ids)

    def _effective(
        self, u: Uncertainty, issues: list[str], req: MissionRequirement, confirmed: bool = False
    ) -> Uncertainty:
        """Conservative reading of upstream output: uncalibrated critical beliefs get an epistemic floor.

        The floor is a calibration gap, not OOD evidence, and an independent confirming look closes it
        (``_confirmed_after_request``): after that the belief's own uncertainty is used."""
        epistemic = u.epistemic
        if (
            ISSUE_UNCALIBRATED in issues
            and not confirmed
            and req.consequence >= self.config.consequence_matters_above
        ):
            epistemic = max(epistemic, self.config.uncalibrated_epistemic_floor)
        contradiction = u.contradiction
        if ISSUE_EVIDENCE_CONFLICT in issues:
            contradiction = max(contradiction, self.config.thresholds.contradiction)
        return u.model_copy(update={"epistemic": epistemic, "contradiction": contradiction})


def _confirmed_after_request(
    ctx: DecisionContext, req: MissionRequirement, primary: list[BeliefMessage]
) -> bool:
    """An uncalibrated-source gap is closed by an independent confirming observation (ch16 L6825 design
    reading, M1-ACTION-E001): a belief of this requirement has a NEWER REVISION than the one an information
    request on it saw, the runtime carried that request out, and the new revision is DIRECT (every required
    property OBSERVED, evidence present, no evidence conflict). Deferred / dropped requests
    (``executed=False``) and history without recorded revisions never confirm anything.

    The closure does not expire with the decision-history window: ``DecisionContext.request_answered`` also
    reads the runtime's window-independent ledger. The belief-side conditions above ARE re-checked every
    cycle, so a later conflict, a lost observation or a stale property reopens the gap (I5 iteration 3).
    """
    for m in primary:
        if not m.evidence_support or m.evidence_conflicts:
            continue
        props = [_prop(m, n) for n in req.properties]
        if not props or any(p is None or p.status is not KnowledgeStatus.OBSERVED for p in props):
            continue
        if ctx.request_answered(m.belief_id, m.revision):
            return True
    return False


__all__ = [
    "ISSUE_CROSS_DOMAIN",
    "ISSUE_DOMAIN_DEGRADED",
    "ISSUE_DOMAIN_UNAVAILABLE",
    "ISSUE_EVIDENCE_CONFLICT",
    "ISSUE_MISSING",
    "ISSUE_NO_EVIDENCE_PATH",
    "ISSUE_STALE",
    "ISSUE_UNCALIBRATED",
    "ISSUE_UNKNOWN_STATUS",
    "ISSUE_WRONG_ASSOCIATION",
    "ClaimGraph",
    "ClaimGraphBuilder",
    "RequirementAssessment",
    "diagnose_causes",
]
