"""PMBL facade: merge, split and query on top of :class:`PMBLBase` (ch8).

Merge and split REPLAY the archived evidence of the predecessors through the analytic operator
instead of averaging latent vectors, so the successor's state, uncertainty and contradiction memory
are all re-derived from raw evidence, and every predecessor stays traceable (CC-04).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from conrad.core.lifecycle import check_transition, is_terminal
from conrad.core.pmbl_core import DirectUpdateOutcome, PMBLBase
from conrad.core.pmbl_store import CommitSpec
from conrad.core.state import unknown_state
from conrad.schemas.belief import (
    BeliefCell,
    BeliefQuery,
    KnowledgeStatus,
    Lifecycle,
    Relationship,
    UpdateKind,
)
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.observation import Evidence
from conrad.schemas.provenance import SourceType
from conrad.schemas.timebase import TimeStamp

__all__ = ["PMBL", "DirectUpdateOutcome"]


def _overlaps(a: SpatialSupport, b: SpatialSupport) -> bool:
    if a.frame_id != b.frame_id:
        return False
    return all(
        abs(ca - cb) <= ha + hb
        for ca, cb, ha, hb in zip(a.center_m, b.center_m, a.half_extent_m, b.half_extent_m, strict=True)
    )


class PMBL(PMBLBase):
    # ------------------------------------------------------------------ successor construction
    def _successor(
        self,
        template: BeliefCell,
        evidence: Sequence[Evidence],
        parents: tuple[UUID, ...],
        kind: str,
        now: TimeStamp,
        z_from: UUID,
    ) -> BeliefCell:
        belief_id = self.ids.new()
        state = unknown_state()
        for ev in sorted(evidence, key=lambda e: (e.timestamp.time_ns, e.evidence_id.int)):
            state = self.buo.update(state, [ev]).state
        roots = [self.node(p).cell.provenance_root for p in parents]
        rec = self.store.record(
            SourceType.RELATIONAL_INFERENCE,
            parents,
            f"pmbl.{kind.lower()}",
            now,
            parents=[*dict.fromkeys(roots), *dict.fromkeys(e.provenance_id for e in evidence)],
            subject_id=belief_id,
        )
        count = state.independent_observation_count
        confirmed = count >= self.cfg.pmbl.confirm_independent_observations
        latest = max(evidence, key=lambda e: e.timestamp.time_ns) if evidence else None
        cell = self.store.commit(
            CommitSpec(
                belief_id=belief_id,
                domain=template.domain,
                entity_type=template.entity_type,
                lifecycle=Lifecycle.CONFIRMED if confirmed else Lifecycle.CANDIDATE,
                timestamp=latest.timestamp if latest else now,
                state=state,
                update_kind=UpdateKind.CREATE,
                provenance_root=rec.record_id,
                provenance=(rec,),
                consumed_evidence_ids=tuple(e.evidence_id for e in evidence),
                z=self.node(z_from).z,
                spatial_support=latest.spatial_support
                if latest and latest.spatial_support
                else template.spatial_support,
                registry_entity_id=template.registry_entity_id,
                fallback_status=KnowledgeStatus.UNKNOWN,
                lineage_parents=parents,
                lineage_kind=kind,
            )
        )
        self.graph.load(cell, state)
        return cell

    def _repoint(self, old_ids: set[UUID], new_id: UUID, provenance_id: UUID) -> list[Relationship]:
        out: list[Relationship] = []
        for rel in list(self.graph.relationships.values()):
            if rel.source_belief_id not in old_ids and rel.target_belief_id not in old_ids:
                continue
            src = new_id if rel.source_belief_id in old_ids else rel.source_belief_id
            dst = new_id if rel.target_belief_id in old_ids else rel.target_belief_id
            if src == dst:
                continue
            out.append(
                rel.model_copy(
                    update={
                        "relationship_id": self.ids.new(),
                        "source_belief_id": src,
                        "target_belief_id": dst,
                        "provenance_id": provenance_id,
                    }
                )
            )
        return out

    # ------------------------------------------------------------------ merge / split
    def merge(self, a: UUID, b: UUID, now: TimeStamp) -> BeliefCell:
        """B_a + B_b -> B'. Predecessors become MERGED; lineage, evidence and provenance are preserved."""
        if a == b:
            raise ValueError("cannot merge a belief with itself")
        na, nb = self.node(a), self.node(b)
        for n in (na, nb):
            check_transition(n.cell.lifecycle, Lifecycle.MERGED)
        if na.cell.domain is not nb.cell.domain:
            raise ValueError("beliefs from different domains cannot be merged")
        evidence = {e.evidence_id: e for p in (a, b) for e in self.archive.for_belief(p)}
        richer = (
            a
            if na.corrected.independent_observation_count >= nb.corrected.independent_observation_count
            else b
        )
        template = self.node(richer).cell
        child = self._successor(template, list(evidence.values()), (a, b), "MERGE", now, richer)
        for rel in self._repoint({a, b}, child.belief_id, child.provenance_root):
            self.graph.add_relationship(rel)
        for parent in (a, b):
            self.transition(parent, Lifecycle.MERGED, now)
            self.graph.evict(parent)
        return child

    def split(
        self, belief_id: UUID, partitions: Sequence[Sequence[UUID]], now: TimeStamp
    ) -> list[BeliefCell]:
        """B -> B_1..B_n with the archived evidence redistributed exactly as given by ``partitions``."""
        node = self.node(belief_id)
        check_transition(node.cell.lifecycle, Lifecycle.SPLIT)
        if len(partitions) < 2:
            raise ValueError("a split needs at least two evidence partitions")
        evidence = {e.evidence_id: e for e in self.archive.for_belief(belief_id)}
        flat = [eid for part in partitions for eid in part]
        if len(flat) != len(set(flat)):
            raise ValueError("evidence appears in more than one split partition")
        if set(flat) != set(evidence):
            raise ValueError("split partitions must redistribute exactly the evidence of the parent belief")
        if any(len(part) == 0 for part in partitions):
            raise ValueError("every split successor needs at least one evidence object")
        children = [
            self._successor(node.cell, [evidence[e] for e in part], (belief_id,), "SPLIT", now, belief_id)
            for part in partitions
        ]
        self.transition(belief_id, Lifecycle.SPLIT, now)
        self.graph.evict(belief_id)
        return children

    # ------------------------------------------------------------------ query
    def query(self, query: BeliefQuery, include_terminal: bool = False) -> list[BeliefCell]:
        """Read from the PERSISTENT store, so results survive a working-memory reset."""
        out: list[BeliefCell] = []
        for cell in self.store.heads():
            if not include_terminal and is_terminal(cell.lifecycle):
                continue
            if query.domain is not None and cell.domain is not query.domain:
                continue
            if query.belief_ids and cell.belief_id not in query.belief_ids:
                continue
            if query.entity_ids and cell.registry_entity_id not in query.entity_ids:
                continue
            if query.region is not None and (
                cell.spatial_support is None or not _overlaps(query.region, cell.spatial_support)
            ):
                continue
            if query.time_range_ns is not None and not (
                query.time_range_ns[0] <= cell.timestamp.time_ns <= query.time_range_ns[1]
            ):
                continue
            if not query.include_predictions and cell.knowledge_status is KnowledgeStatus.PREDICTED:
                continue
            u = cell.uncertainty
            if (
                query.min_observational_uncertainty is not None
                and u.observational < query.min_observational_uncertainty
            ):
                continue
            if (
                query.min_contradiction_uncertainty is not None
                and u.contradiction < query.min_contradiction_uncertainty
            ):
                continue
            out.append(cell)
        out.sort(key=lambda c: c.belief_id.int)
        return out[: query.max_results]
