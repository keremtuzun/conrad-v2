"""Persistent belief store and evidence archive over the single-writer Repository (ch8, ch33 PMBL).

Every write goes through ``Repository.commit_update``; nothing here touches SQL directly.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from torch import Tensor

from conrad.core.config import CoreConfig
from conrad.core.lifecycle import check_transition
from conrad.core.state import AnalyticBeliefState, state_to_claims
from conrad.core.uncertainty import channels_to_uncertainty, uncalibrated_metadata
from conrad.persistence.repository import BeliefUpdate, CommitResult, Repository
from conrad.schemas.belief import (
    BeliefCell,
    BeliefRevision,
    KnowledgeStatus,
    Lifecycle,
    Relationship,
    UpdateKind,
    summarize_status,
)
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.uncertainty import UncertaintyRepresentation
from conrad.schemas.world import Domain


class EvidenceArchive:
    """Immutable evidence + its provenance. Idempotent on evidence_id."""

    def __init__(self, repo: Repository, run_id: UUID) -> None:
        self.repo = repo
        self.run_id = run_id

    def put(self, evidence: Evidence, provenance: Sequence[ProvenanceRecord] = ()) -> bool:
        if provenance:
            self.repo.put_provenance(self.run_id, provenance)
        return self.repo.put_evidence(evidence)

    def get(self, evidence_id: UUID) -> Evidence | None:
        return self.repo.evidence(evidence_id)

    def for_belief(self, belief_id: UUID) -> list[Evidence]:
        found = (
            self.repo.evidence(e)
            for e in sorted(self.repo.contributed_evidence(belief_id), key=lambda u: u.int)
        )
        return [e for e in found if e is not None]


@dataclass(frozen=True)
class CommitSpec:
    belief_id: UUID
    domain: Domain
    entity_type: str
    lifecycle: Lifecycle
    timestamp: TimeStamp
    state: AnalyticBeliefState
    update_kind: UpdateKind
    provenance_root: UUID
    provenance: tuple[ProvenanceRecord, ...] = ()
    consumed_evidence_ids: tuple[UUID, ...] = ()
    measurement_time_ns: int | None = None
    late: bool = False
    z: Tensor | None = None
    temporal: Tensor | None = None
    history: Tensor | None = None
    spatial_support: SpatialSupport | None = None
    registry_entity_id: UUID | None = None
    fallback_status: KnowledgeStatus = KnowledgeStatus.UNKNOWN
    relationships: tuple[Relationship, ...] = ()
    lineage_parents: tuple[UUID, ...] = ()
    lineage_kind: str | None = None


def _vec(t: Tensor | None, fallback: tuple[float, ...]) -> tuple[float, ...]:
    return fallback if t is None else tuple(float(v) for v in t.detach().cpu().reshape(-1).tolist())


class PersistentBeliefStore:
    def __init__(self, repo: Repository, run_id: UUID, cfg: CoreConfig, ids: IdFactory) -> None:
        self.repo = repo
        self.run_id = run_id
        self.cfg = cfg
        self.ids = ids
        self.last_result: CommitResult | None = None

    def record(
        self,
        source_type: SourceType,
        source_ids: Sequence[UUID],
        operation: str,
        timestamp: TimeStamp,
        parents: Sequence[UUID] = (),
        subject_id: UUID | None = None,
    ) -> ProvenanceRecord:
        return ProvenanceRecord(
            record_id=self.ids.new(),
            source_type=source_type,
            source_ids=tuple(source_ids),
            operation=operation,
            module="conrad.core",
            model_version=self.cfg.model_version,
            timestamp=timestamp,
            parent_records=tuple(parents),
            subject_id=subject_id,
        )

    def commit(self, spec: CommitSpec) -> BeliefCell:
        prev = self.repo.head(spec.belief_id)
        if prev is not None:
            check_transition(prev.lifecycle, spec.lifecycle)
        method = "analytic_buo_v0" if spec.update_kind is not UpdateKind.PREDICTED else "analytic_tbd_v0"
        uncertainty = channels_to_uncertainty(
            spec.state.channels, UncertaintyRepresentation.ANALYTIC, uncalibrated_metadata(method)
        )
        claims = state_to_claims(spec.state, uncertainty, spec.provenance_root)
        status = summarize_status(claims) if claims else spec.fallback_status
        revision = 0 if prev is None else prev.revision + 1
        dz = self.cfg.belief_dim
        cell = BeliefCell(
            belief_id=spec.belief_id,
            domain=spec.domain,
            entity_type=spec.entity_type,
            registry_entity_id=spec.registry_entity_id,
            lifecycle=spec.lifecycle,
            revision=revision,
            timestamp=spec.timestamp,
            state_embedding=_vec(spec.z, prev.state_embedding if prev else (0.0,) * dz),
            temporal_state=_vec(spec.temporal, prev.temporal_state if prev else ()),
            history_state=_vec(spec.history, prev.history_state if prev else ()),
            claims=claims,
            knowledge_status=status,
            uncertainty=uncertainty,
            spatial_support=spec.spatial_support
            if spec.spatial_support
            else (prev.spatial_support if prev else None),
            independent_observation_count=spec.state.independent_observation_count,
            lineage_parent_ids=spec.lineage_parents if prev is None else prev.lineage_parent_ids,
            provenance_root=spec.provenance_root,
            model_version=self.cfg.model_version,
        )
        rev = BeliefRevision(
            belief_id=spec.belief_id,
            revision=revision,
            predecessor_revision=None if prev is None else prev.revision,
            measurement_time_ns=spec.timestamp.time_ns
            if spec.measurement_time_ns is None
            else spec.measurement_time_ns,
            consumed_evidence_ids=spec.consumed_evidence_ids,
            provenance_root=spec.provenance_root,
            update_kind=spec.update_kind,
            cell=cell,
            late=spec.late,
        )
        self.last_result = self.repo.commit_update(
            BeliefUpdate(
                message_id=self.ids.new(),
                producer_version=self.cfg.model_version,
                run_id=self.run_id,
                revision=rev,
                provenance=spec.provenance,
                relationships=spec.relationships,
                lineage_parents=spec.lineage_parents,
                lineage_kind=spec.lineage_kind,
            )
        )
        stored = self.repo.head(spec.belief_id)
        assert stored is not None
        return stored

    def heads(self) -> list[BeliefCell]:
        return self.repo.heads(self.run_id)
