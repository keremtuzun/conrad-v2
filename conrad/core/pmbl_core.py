"""PMBL: persistent memory + belief lifecycle around the working graph (ch8, ch33).

Working memory (tensors) <-> persistent store (every committed revision) <-> immutable archive.
Late evidence is either an explicit ``late=True`` revision or an explicit rejection; it is never
silently treated as current (CC-02). ``reset_working_memory`` leaves persistence intact (CC-09).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from torch import Tensor

from conrad.core.belief_graph import WorkingBeliefGraph, WorkingNode
from conrad.core.buo_analytic import AnalyticBUO, AnalyticUpdateResult
from conrad.core.config import CoreConfig
from conrad.core.lifecycle import LifecycleEngine, is_terminal
from conrad.core.pmbl_store import CommitSpec, EvidenceArchive, PersistentBeliefStore
from conrad.core.state import (
    AnalyticBeliefState,
    ContradictionEntry,
    contradiction_refs_from_claims,
    estimates_from_claims,
    unknown_state,
)
from conrad.persistence.repository import Repository
from conrad.schemas.belief import BeliefCell, KnowledgeStatus, Lifecycle, Relationship, UpdateKind
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain


@dataclass(frozen=True)
class DirectUpdateOutcome:
    cell: BeliefCell
    result: AnalyticUpdateResult | None
    late_evidence_ids: tuple[UUID, ...] = ()
    rejected_late_ids: tuple[UUID, ...] = ()


class PMBLBase:
    def __init__(self, repo: Repository, run_id: UUID, cfg: CoreConfig, ids: IdFactory) -> None:
        self.cfg = cfg
        self.ids = ids
        self.store = PersistentBeliefStore(repo, run_id, cfg, ids)
        self.archive = EvidenceArchive(repo, run_id)
        self.lifecycle = LifecycleEngine(cfg.pmbl)
        self.buo = AnalyticBUO(cfg.analytic_buo)
        self.graph = WorkingBeliefGraph(cfg)

    # ------------------------------------------------------------------ memory levels
    def reset_working_memory(self) -> None:
        self.graph.clear()

    def rebuild_state(self, cell: BeliefCell) -> AnalyticBeliefState:
        """Reconstruct the interpretable state from persistent store + archive only."""
        evidence = self.archive.for_belief(cell.belief_id)
        by_id = {e.evidence_id: e for e in evidence}
        entries = tuple(
            ContradictionEntry(eid, prop, by_id[eid].measurements[prop], 0.0, by_id[eid].reliability)
            for eid, prop in contradiction_refs_from_claims(cell.claims)
            if eid in by_id and prop in by_id[eid].measurements
        )
        u = cell.uncertainty
        return AnalyticBeliefState(
            estimates=estimates_from_claims(cell.claims),
            ua=u.aleatoric,
            ue=u.epistemic,
            uc=u.contradiction,
            uo=u.observational,
            coverage=max(0.0, 1.0 - u.observational),
            independence_groups=frozenset(e.independence_group or str(e.evidence_id) for e in evidence),
            consumed_evidence_ids=frozenset(by_id),
            contradictions=entries,
        )

    def node(self, belief_id: UUID) -> WorkingNode:
        if belief_id in self.graph:
            return self.graph.nodes[belief_id]
        cell = self.store.repo.head(belief_id)
        if cell is None:
            raise KeyError(f"unknown belief {belief_id}")
        return self.graph.load(cell, self.rebuild_state(cell))

    def archive_evidence(self, evidence: Evidence, provenance: Sequence[ProvenanceRecord]) -> bool:
        return self.archive.put(evidence, provenance)

    def _require_archived(self, evidence: Sequence[Evidence]) -> None:
        missing = [e.evidence_id for e in evidence if self.archive.get(e.evidence_id) is None]
        if missing:
            raise ValueError(f"evidence must be archived before it updates a belief: {missing}")

    # ------------------------------------------------------------------ create / direct
    def create_candidate(
        self,
        evidence: Sequence[Evidence],
        domain: Domain,
        entity_type: str,
        registry_entity_id: UUID | None = None,
    ) -> BeliefCell:
        """Unmatched evidence creates a CANDIDATE with a belief-plane ID (never a simulator ID)."""
        if not evidence:
            raise ValueError("a candidate belief needs at least one evidence object")
        self._require_archived(evidence)
        belief_id = self.ids.new()
        result = self.buo.update(unknown_state(), evidence)
        latest = max(evidence, key=lambda e: e.timestamp.time_ns)
        rec = self.store.record(
            SourceType.DIRECT_OBSERVATION,
            result.consumed_evidence_ids,
            "pmbl.create_candidate",
            latest.timestamp,
            parents=[e.provenance_id for e in evidence],
            subject_id=belief_id,
        )
        count = result.state.independent_observation_count
        cell = self.store.commit(
            CommitSpec(
                belief_id=belief_id,
                domain=domain,
                entity_type=entity_type,
                lifecycle=self.lifecycle.initial(count),
                timestamp=latest.timestamp,
                state=result.state,
                update_kind=UpdateKind.CREATE,
                provenance_root=rec.record_id,
                provenance=(rec,),
                consumed_evidence_ids=result.consumed_evidence_ids,
                spatial_support=latest.spatial_support,
                registry_entity_id=registry_entity_id,
                fallback_status=KnowledgeStatus.OBSERVED,
            )
        )
        self.graph.load(cell, result.state)
        return cell

    def apply_direct(
        self, belief_id: UUID, evidence: Sequence[Evidence], z: Tensor | None = None
    ) -> DirectUpdateOutcome:
        node = self.node(belief_id)
        if is_terminal(node.cell.lifecycle):
            raise ValueError(f"belief {belief_id} is {node.cell.lifecycle.value} and cannot be updated")
        self._require_archived(evidence)
        head_ns = node.cell.timestamp.time_ns
        late = [e for e in evidence if e.timestamp.time_ns < head_ns]
        current = [e for e in evidence if e.timestamp.time_ns >= head_ns]
        rejected: tuple[UUID, ...] = ()
        outcome: AnalyticUpdateResult | None = None
        if late and self.cfg.pmbl.late_evidence_policy == "REJECT":
            rejected = tuple(e.evidence_id for e in late)
            late = []
        if late:
            outcome = self._commit_direct(node, [self._age_inflated(e, head_ns) for e in late], z, late=True)
        if current:
            outcome = self._commit_direct(node, current, z, late=False) or outcome
        return DirectUpdateOutcome(node.cell, outcome, tuple(e.evidence_id for e in late), rejected)

    def _age_inflated(self, ev: Evidence, head_ns: int) -> Evidence:
        """Old measurement of a possibly changed state: widen it by the declared process noise."""
        gap_s = (head_ns - ev.timestamp.time_ns) / 1e9
        extra = self.cfg.tbd.analytic_process_noise_per_s * gap_s
        return ev.model_copy(update={"aleatoric_uncertainty": ev.aleatoric_uncertainty + extra})

    def _commit_direct(
        self, node: WorkingNode, evidence: Sequence[Evidence], z: Tensor | None, late: bool
    ) -> AnalyticUpdateResult | None:
        base = node.predicted if node.predicted is not None else node.corrected
        result = self.buo.update(base, evidence)
        if not result.changed:
            return result  # duplicates only: nothing is committed (CC-01)
        cell = node.cell
        latest = max(evidence, key=lambda e: e.timestamp.time_ns)
        rec = self.store.record(
            SourceType.DIRECT_OBSERVATION,
            result.consumed_evidence_ids,
            "buo.direct_update.late" if late else "buo.direct_update",
            latest.timestamp,
            parents=[cell.provenance_root, *[e.provenance_id for e in evidence]],
            subject_id=cell.belief_id,
        )
        lifecycle = self.lifecycle.on_direct_evidence(
            cell.lifecycle, result.state.independent_observation_count
        )
        node.cell = self.store.commit(
            CommitSpec(
                belief_id=cell.belief_id,
                domain=cell.domain,
                entity_type=cell.entity_type,
                lifecycle=lifecycle,
                timestamp=cell.timestamp if late else latest.timestamp,
                measurement_time_ns=latest.timestamp.time_ns,
                late=late,
                state=result.state,
                update_kind=UpdateKind.DIRECT,
                provenance_root=rec.record_id,
                provenance=(rec,),
                consumed_evidence_ids=result.consumed_evidence_ids,
                z=node.z if z is None else z,
                temporal=node.temporal,
                history=node.history,
                registry_entity_id=cell.registry_entity_id,
            )
        )
        if z is not None:
            node.z = z.detach()
        node.corrected, node.predicted = result.state, None
        node.last_surprise = result.temporal_surprise
        return result

    # ------------------------------------------------------------------ predicted / relational / lifecycle
    def apply_prediction(self, belief_id: UUID, predicted: AnalyticBeliefState, now: TimeStamp) -> BeliefCell:
        """Persist B^- as a PREDICTED revision. The corrected state stays available on the node."""
        node = self.node(belief_id)
        rec = self.store.record(
            SourceType.TEMPORAL_PREDICTION,
            [belief_id],
            "tbd.predict",
            now,
            parents=[node.cell.provenance_root],
            subject_id=belief_id,
        )
        node.predicted = predicted
        node.cell = self._commit_derived(node, predicted, now, UpdateKind.PREDICTED, rec, ())
        return node.cell

    def apply_relational(
        self,
        belief_id: UUID,
        inferred: AnalyticBeliefState,
        source_belief_ids: Sequence[UUID],
        relationships: Sequence[Relationship],
        now: TimeStamp,
        z: Tensor | None = None,
    ) -> BeliefCell:
        node = self.node(belief_id)
        parents = [node.cell.provenance_root]
        parents += [self.node(s).cell.provenance_root for s in source_belief_ids]
        rec = self.store.record(
            SourceType.RELATIONAL_INFERENCE,
            [*source_belief_ids, *[r.relationship_id for r in relationships]],
            "rbp.propagate",
            now,
            parents=list(dict.fromkeys(parents)),
            subject_id=belief_id,
        )
        if z is not None:
            node.z = z.detach()
        node.cell = self._commit_derived(
            node, inferred, now, UpdateKind.RELATIONAL, rec, tuple(relationships)
        )
        node.corrected, node.predicted = inferred, None
        return node.cell

    def _commit_derived(
        self,
        node: WorkingNode,
        state: AnalyticBeliefState,
        now: TimeStamp,
        kind: UpdateKind,
        rec: ProvenanceRecord,
        relationships: tuple[Relationship, ...],
    ) -> BeliefCell:
        cell = node.cell
        fallback = KnowledgeStatus.PREDICTED if kind is UpdateKind.PREDICTED else KnowledgeStatus.INFERRED
        return self.store.commit(
            CommitSpec(
                belief_id=cell.belief_id,
                domain=cell.domain,
                entity_type=cell.entity_type,
                lifecycle=cell.lifecycle,
                timestamp=now,
                state=state,
                update_kind=kind,
                provenance_root=rec.record_id,
                provenance=(rec,),
                z=node.z,
                temporal=node.temporal,
                history=node.history,
                registry_entity_id=cell.registry_entity_id,
                fallback_status=fallback,
                relationships=relationships,
            )
        )

    def transition(self, belief_id: UUID, new: Lifecycle, now: TimeStamp) -> BeliefCell:
        """Logged lifecycle-only revision. Illegal transitions raise before any write."""
        node = self.node(belief_id)
        cell = node.cell
        state = node.predicted if node.predicted is not None else node.corrected
        node.cell = self.store.commit(
            CommitSpec(
                belief_id=belief_id,
                domain=cell.domain,
                entity_type=cell.entity_type,
                lifecycle=new,
                timestamp=self._same_domain(cell.timestamp, now),
                state=state,
                update_kind=UpdateKind.LIFECYCLE,
                provenance_root=cell.provenance_root,
                registry_entity_id=cell.registry_entity_id,
                fallback_status=cell.knowledge_status,
            )
        )
        return node.cell

    @staticmethod
    def _same_domain(old: TimeStamp, now: TimeStamp) -> TimeStamp:
        """Lifecycle bookkeeping keeps the belief's own time; it never invents a newer observation."""
        if old.clock_domain != now.clock_domain:
            raise ValueError("lifecycle transition time is in a different clock domain than the belief")
        return old

    def last_direct_time_ns(self, belief_id: UUID) -> int:
        direct = (UpdateKind.CREATE, UpdateKind.DIRECT)
        times = [
            r.measurement_time_ns for r in self.store.repo.revisions(belief_id) if r.update_kind in direct
        ]
        return max(times) if times else 0

    def sweep(self, now: TimeStamp) -> list[tuple[UUID, Lifecycle, Lifecycle]]:
        """Apply silence policy (reject stale candidates, dormancy, optional retirement)."""
        changed: list[tuple[UUID, Lifecycle, Lifecycle]] = []
        for cell in self.store.heads():
            if is_terminal(cell.lifecycle) or cell.timestamp.clock_domain != now.clock_domain:
                continue
            idle_s = max(0.0, (now.time_ns - self.last_direct_time_ns(cell.belief_id)) / 1e9)
            new = self.lifecycle.on_silence(cell.lifecycle, idle_s)
            if new is not cell.lifecycle:
                self.transition(cell.belief_id, new, now)
                changed.append((cell.belief_id, cell.lifecycle, new))
        return changed
