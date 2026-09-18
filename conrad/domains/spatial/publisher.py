"""Region/object belief publication and persistence through ``Repository.commit_update``. BELIEF PLANE.

Belief IDs are minted here (inference-owned) per global block or per registry asset; a registry identity
is attached only when it came from mission context. Each revision carries one provenance record:
DIRECT_OBSERVATION (source = consumed evidence), RELATIONAL_INFERENCE (source = supporting evidence of the
inferred cells) or TEMPORAL_PREDICTION (parent = previous revision record), chained to the previous one.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from conrad.domains.spatial import messages as msgs
from conrad.domains.spatial.config import SpatialConfig
from conrad.domains.spatial.hierarchy import HierarchicalMap
from conrad.domains.spatial.keys import IntArr, encode, region_indices
from conrad.persistence.repository import BeliefUpdate, Repository
from conrad.schemas.belief import Availability, BeliefCell, BeliefMessage, BeliefRevision, UpdateKind
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp

_SOURCE = {
    UpdateKind.DIRECT: SourceType.DIRECT_OBSERVATION,
    UpdateKind.RELATIONAL: SourceType.RELATIONAL_INFERENCE,
    UpdateKind.PREDICTED: SourceType.TEMPORAL_PREDICTION,
    UpdateKind.CONTEXT: SourceType.CROSS_DOMAIN_CONTEXT,
}


@dataclass(frozen=True)
class RegistryAsset:
    registry_entity_id: UUID
    center_m: tuple[float, float, float]
    half_extent_m: tuple[float, float, float]
    semantic_class: str | None = None


@dataclass(frozen=True)
class RegionChange:
    kind: UpdateKind
    evidence: tuple[UUID, ...]  # consumed direct evidence (DIRECT) or supporting evidence (RELATIONAL)
    measurement_ns: int
    summary: str


class RegionPublisher:
    def __init__(
        self, cfg: SpatialConfig, ids: IdFactory, repository: Repository | None, run_id: UUID | None
    ) -> None:
        self.cfg = cfg
        self.ids = ids
        self.repository = repository
        self.run_id = run_id
        self.block_ids: dict[int, UUID] = {}
        self.asset_ids: dict[UUID, UUID] = {}
        self.revision: dict[UUID, int] = {}
        self.head_ns: dict[UUID, int] = {}
        self.last_record: dict[UUID, UUID] = {}
        self.latest: dict[UUID, BeliefMessage] = {}
        self.commits = 0

    def _belief_id(self, table: dict[Any, UUID], key: Any) -> UUID:
        if key not in table:
            table[key] = self.ids.new()
        return table[key]

    def publish_changes(
        self,
        hmap: HierarchicalMap,
        block_ev: dict[int, list[UUID]],
        block_t: dict[int, int],
        relational: tuple[IntArr, dict[int, list[UUID]]],
        registry: Sequence[RegistryAsset],
        now: TimeStamp,
        avail: Availability,
    ) -> list[BeliefMessage]:
        """DIRECT revisions for blocks with new evidence, RELATIONAL ones for blocks whose inference changed,
        and one object revision per registry asset overlapping a changed block."""
        rel_codes, refs = relational
        rel_blocks: dict[int, list[UUID]] = {}
        if len(rel_codes):
            for code, b in zip(rel_codes.tolist(), hmap.block_code(rel_codes).tolist(), strict=True):
                rel_blocks.setdefault(b, []).extend(refs.get(code, []))
        changes: dict[int, RegionChange] = {}
        for b in sorted(set(block_ev) | set(rel_blocks)):
            if b in block_ev:
                n = len(block_ev[b])
                changes[b] = RegionChange(
                    UpdateKind.DIRECT, tuple(block_ev[b]), block_t[b], f"direct: {n} obs"
                )
            elif rel_blocks[b] or b in self.block_ids:  # a provenance record needs sources or a parent
                bid = self.block_ids.get(b)
                t = self.head_ns.get(bid, now.time_ns) if bid is not None else now.time_ns
                changes[b] = RegionChange(
                    UpdateKind.RELATIONAL, tuple(rel_blocks[b]), t, "relational gap inference"
                )
        out = [self.publish_block(hmap, b, ch, now, avail) for b, ch in changes.items()]
        for asset in registry:
            hit = sorted(self.asset_blocks(hmap, asset) & set(changes))
            if not hit:
                continue
            direct = [e for b in hit if changes[b].kind is UpdateKind.DIRECT for e in changes[b].evidence]
            ev = tuple(direct) if direct else tuple(e for b in hit for e in changes[b].evidence)
            if not ev and asset.registry_entity_id not in self.asset_ids:
                continue
            kind = UpdateKind.DIRECT if direct else UpdateKind.RELATIONAL
            t = max(changes[b].measurement_ns for b in hit)
            change = RegionChange(kind, ev, t, f"asset region: {len(hit)} blocks")
            out.append(self.publish_asset(hmap, asset, change, now, avail))
        return out

    def publish_block(
        self, hmap: HierarchicalMap, block: int, change: RegionChange, now: TimeStamp, avail: Availability
    ) -> BeliefMessage:
        lo, hi = hmap.block_bounds(block)
        rows = np.asarray(hmap.blocks.get(block, []), dtype=np.int64)
        n = self.cfg.grid.global_block_cells**3
        s = msgs.summarize(hmap.base, rows, n, 0.5 * (lo + hi), 0.5 * (hi - lo))
        return self._emit(self._belief_id(self.block_ids, block), None, s, change, now, avail)

    def publish_asset(
        self,
        hmap: HierarchicalMap,
        asset: RegistryAsset,
        change: RegionChange,
        now: TimeStamp,
        avail: Availability,
    ) -> BeliefMessage:
        idx = region_indices(np.asarray(asset.center_m), np.asarray(asset.half_extent_m), hmap.base.res)
        rows = hmap.base.rows(encode(idx))
        s = msgs.summarize(
            hmap.base, rows, len(idx), np.asarray(asset.center_m), np.asarray(asset.half_extent_m)
        )
        bid = self._belief_id(self.asset_ids, asset.registry_entity_id)
        return self._emit(bid, asset.registry_entity_id, s, change, now, avail)

    def asset_blocks(self, hmap: HierarchicalMap, asset: RegistryAsset) -> set[int]:
        idx = region_indices(np.asarray(asset.center_m), np.asarray(asset.half_extent_m), hmap.base.res)
        blocks = np.floor_divide(idx, self.cfg.grid.global_block_cells)
        return set(encode(np.unique(blocks, axis=0)).tolist())

    def _emit(
        self,
        bid: UUID,
        registry_id: UUID | None,
        s: msgs.RegionSummary,
        change: RegionChange,
        now: TimeStamp,
        avail: Availability,
    ) -> BeliefMessage:
        rev = self.revision.get(bid, -1) + 1
        kind = change.kind
        cap = self.cfg.messages.max_evidence_refs
        evidence = tuple(dict.fromkeys(change.evidence))  # every consumed id is committed; messages cap
        record = self._record(bid, kind, evidence, now)
        predicted = kind is UpdateKind.PREDICTED
        # a prediction is not a measurement: it keeps the head measurement time
        measured = self.head_ns.get(bid, change.measurement_ns) if predicted else change.measurement_ns
        spec = msgs.MessageSpec(
            belief_id=bid,
            revision=rev,
            registry_id=registry_id,
            provenance_id=record.record_id,
            evidence=evidence[-cap:] if kind is UpdateKind.DIRECT else (),
            change_summary=None if predicted else change.summary,
            prediction_summary=change.summary if predicted else None,
            availability=avail,
        )
        message, cell = msgs.build(
            s,
            spec,
            self.ids.new(),
            now,
            self.cfg.model_version,
            self.cfg.grid.base_voxel_m,
            self.cfg.messages.entity_type,
            self.cfg.grid.frame_id,
            predicted_only=predicted,
        )
        if self.repository is not None and self.run_id is not None:
            self._commit(message, cell, record, kind, evidence, measured)
        self.revision[bid] = rev
        self.head_ns[bid] = max(self.head_ns.get(bid, 0), measured)
        self.last_record[bid] = record.record_id
        self.latest[bid] = message
        return message

    def _record(
        self, bid: UUID, kind: UpdateKind, evidence: tuple[UUID, ...], now: TimeStamp
    ) -> ProvenanceRecord:
        parent = self.last_record.get(bid)
        sources = evidence if kind in (UpdateKind.DIRECT, UpdateKind.RELATIONAL) else (bid,)
        return ProvenanceRecord(
            record_id=self.ids.new(),
            source_type=_SOURCE[kind],
            source_ids=sources,
            operation=f"uahsm.{kind.value.lower()}",
            module="conrad.domains.spatial",
            model_version=self.cfg.model_version,
            timestamp=now,
            parent_records=(parent,) if parent is not None else (),
            subject_id=bid,
        )

    def _commit(
        self,
        message: BeliefMessage,
        cell: BeliefCell,
        record: ProvenanceRecord,
        kind: UpdateKind,
        evidence: tuple[UUID, ...],
        measurement_ns: int,
    ) -> None:
        assert self.repository is not None and self.run_id is not None
        bid, rev = message.belief_id, message.revision
        head = self.head_ns.get(bid)
        revision = BeliefRevision(
            belief_id=bid,
            revision=rev,
            predecessor_revision=rev - 1 if rev > 0 else None,
            measurement_time_ns=measurement_ns,
            consumed_evidence_ids=evidence if kind is UpdateKind.DIRECT else (),
            provenance_root=record.record_id,
            update_kind=kind,
            cell=cell,
            late=head is not None and measurement_ns < head,
        )
        self.repository.commit_update(
            BeliefUpdate(
                message_id=message.message_id,
                producer_version=self.cfg.model_version,
                run_id=self.run_id,
                revision=revision,
                provenance=(record,),
            )
        )
        self.commits += 1


def blocks_of(hmap: HierarchicalMap, base_codes: np.ndarray) -> np.ndarray:
    return np.unique(hmap.block_code(base_codes)) if len(base_codes) else np.zeros(0, dtype=np.int64)
