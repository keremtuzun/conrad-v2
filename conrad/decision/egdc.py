"""EGDC pipeline: context -> claim graph -> candidates -> consequence -> constraints -> selection -> router.

(ch16 'EGDC computational pipeline', ch28 'execute_proposal'.) The policy only ranks. Proposals are
checked in rank order by the deterministic ConstraintEngine; the first accepted proposal is chosen;
everything rejected is kept in the record with its reason codes. If nothing mission-advancing or
information-acquiring can be justified, the outcome is a recorded, justified abstention.

implementation_status: EXPERIMENTAL_CANDIDATE reasoning inside FROZEN_CONTRACT boundaries
"""

from __future__ import annotations

from uuid import UUID

from conrad.decision.actions import INFORMATION_ACTIONS, NON_ADVANCING_ACTIONS, CandidateActionGenerator
from conrad.decision.claims import ClaimGraph, ClaimGraphBuilder
from conrad.decision.config import DecisionConfig
from conrad.decision.consequence import ConsequenceConfig, ConsequenceEstimator
from conrad.decision.constraints import ConstraintEngine
from conrad.decision.context import DecisionContext
from conrad.decision.monitor import OutcomeMonitor
from conrad.decision.policy import DecisionPolicy, StructuredReasoningPolicy
from conrad.decision.router import ExecutionRouter, RejectedAction, RoutedAction
from conrad.schemas.base import ConradModel
from conrad.schemas.decision import ActionProposal, ConstraintDecision, DecisionRecord
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import ProvenanceRecord, SourceType


class DecisionOutcome(ConradModel):
    record: DecisionRecord
    provenance: ProvenanceRecord
    routed: RoutedAction | None
    rejected: tuple[RejectedAction, ...]


class EGDC:
    def __init__(
        self,
        id_factory: IdFactory,
        config: DecisionConfig | None = None,
        policy: DecisionPolicy | None = None,
        consequence_config: ConsequenceConfig | None = None,
        monitor: OutcomeMonitor | None = None,
    ) -> None:
        self._ids = id_factory
        self.config = config or DecisionConfig()
        self.policy: DecisionPolicy = policy or StructuredReasoningPolicy(self.config)
        self.builder = ClaimGraphBuilder(id_factory, self.config)
        self.generator = CandidateActionGenerator(id_factory, self.config)
        self.estimator = ConsequenceEstimator(self.config, consequence_config)
        self.constraints = ConstraintEngine(self.config)  # never injectable by a policy
        self.router = ExecutionRouter(id_factory, self.config)
        self.monitor = monitor or OutcomeMonitor()
        self.last_graph: ClaimGraph | None = None

    def decide(self, ctx: DecisionContext) -> DecisionOutcome:
        graph = self.builder.build(ctx)
        candidates = self.generator.generate(graph, ctx)
        self.builder.attach_actions(graph, ctx, candidates)
        consequences = [self.estimator.estimate(a, graph, ctx) for a in candidates]
        by_id = {a.action_id: c for a, c in zip(candidates, consequences, strict=True)}
        ranked = self.policy.rank(graph, candidates, consequences, ctx)
        if {a.action_id for a in ranked} != set(by_id):
            raise ValueError("policy must rank exactly the generated candidates (closed vocabulary)")

        decisions: list[ConstraintDecision] = []
        rejected: list[RejectedAction] = []
        chosen: ActionProposal | None = None
        routed: RoutedAction | None = None
        for action in ranked:
            consequence = by_id[action.action_id] if self.config.enforce_grounding else None
            verdict = self.constraints.check(action, graph, ctx, consequence)
            decisions.append(verdict)
            result = self.router.route(action, verdict, ctx)
            if verdict.accepted:
                chosen, routed = action, result
                break
            assert isinstance(result.payload, RejectedAction)
            rejected.append(result.payload)

        open_items = [a for a in graph.assessments if a.matters and not a.satisfied]
        abstained = chosen is None or (
            bool(open_items)
            and chosen.action_type in NON_ADVANCING_ACTIONS
            and chosen.action_type not in INFORMATION_ACTIONS
        )
        rationale = self._rationale(graph, chosen, rejected, abstained)
        used_beliefs = self._used_beliefs(graph)
        parents = tuple(
            dict.fromkeys(
                p for m in ctx.snapshot.messages if m.belief_id in used_beliefs for p in m.provenance_refs
            )
        )
        provenance = ProvenanceRecord(
            record_id=self._ids.new(),
            source_type=SourceType.DECISION,
            source_ids=(ctx.snapshot.snapshot_id, *sorted(used_beliefs, key=lambda u: u.int)),
            operation="egdc.decide",
            module="conrad.decision.egdc",
            model_version=f"{self.config.model_version}+{self.policy.name}",
            timestamp=ctx.timestamp,
            parent_records=parents,
        )
        record = DecisionRecord(
            decision_id=self._ids.new(),
            trace_id=ctx.trace_id,
            mission_id=ctx.mission.mission_id,
            timestamp=ctx.timestamp,
            belief_snapshot_id=ctx.snapshot.snapshot_id,
            claims=tuple(graph.claims),
            edges=tuple(graph.edges),
            candidates=tuple(ranked),
            chosen=chosen,
            constraint_decisions=tuple(decisions),
            unsupported_claim_ids=graph.unsupported_ids(),
            abstained=abstained,
            rationale=rationale,
            provenance_id=provenance.record_id,
            model_version=f"{self.config.model_version}+{self.policy.name}",
        )
        self.monitor.register(record)
        self.last_graph = graph
        return DecisionOutcome(record=record, provenance=provenance, routed=routed, rejected=tuple(rejected))

    @staticmethod
    def _used_beliefs(graph: ClaimGraph) -> set[UUID]:
        return {b for c in graph.world_claims() for b in c.source_belief_ids}

    @staticmethod
    def _rationale(
        graph: ClaimGraph, chosen: ActionProposal | None, rejected: list[RejectedAction], abstained: bool
    ) -> str:
        """Structured trace, not chain-of-thought text (ch17 'Explainability')."""
        parts = []
        for a in graph.assessments:
            state = "SATISFIED" if a.satisfied else ("OPEN" if a.matters else "OPEN_LOW_CONSEQUENCE")
            causes = ",".join(c.value for c in a.causes) or "-"
            issues = ",".join(a.issues) or "-"
            parts.append(f"req={a.requirement_id} {state} issues={issues} causes={causes}")
        parts.append(f"selected={'NONE' if chosen is None else chosen.action_type.value}")
        if rejected:
            parts.append(
                "rejected=" + ";".join(f"{r.action_type.value}[{','.join(r.reason_codes)}]" for r in rejected)
            )
        if abstained:
            parts.append("abstained=INSUFFICIENT_EVIDENCE_OR_NO_PERMITTED_ACTION")
        return " | ".join(parts)
