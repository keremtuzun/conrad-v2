"""Revision heads, provenance records and persistence for 2E beliefs.

Every revision goes through ``Repository.commit_update`` (when a repository is attached); consumed
evidence is archived first so the repository can count independent observations.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from conrad.persistence.repository import BeliefUpdate, Repository
from conrad.schemas.belief import (
    Availability,
    BeliefCell,
    BeliefMessage,
    BeliefRevision,
    EcologicalPayload,
    KnowledgeStatus,
    Lifecycle,
    PropertyClaim,
    UpdateKind,
    summarize_status,
)
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.uncertainty import Uncertainty
from conrad.schemas.world import Domain

MODULE = "conrad.domains.ecological"


@dataclass
class Head:
    revision: int
    root: UUID
    time_ns: int
    lifecycle: Lifecycle
    entity_type: str
    registry_id: UUID | None
    message: BeliefMessage | None = None


@dataclass(frozen=True)
class CommitSpec:
    belief_id: UUID
    entity_type: str
    registry_id: UUID | None
    lifecycle: Lifecycle
    timestamp: TimeStamp
    claims: tuple[PropertyClaim, ...]
    uncertainty: Uncertainty
    embedding: tuple[float, ...]
    support: SpatialSupport | None
    kind: UpdateKind
    payload: EcologicalPayload
    evidence: tuple[Evidence, ...] = ()
    records: tuple[ProvenanceRecord, ...] = ()
    change_summary: str | None = None


class BeliefLedger:
    def __init__(self, ids: IdFactory, repo: Repository | None, run_id: UUID, model_version: str) -> None:
        self.ids = ids
        self.repo = repo
        self.run_id = run_id
        self.model_version = model_version
        self.heads: dict[UUID, Head] = {}
        self.commits = 0

    def record(
        self,
        source_type: SourceType,
        source_ids: Sequence[UUID],
        subject: UUID,
        t: TimeStamp,
        parents: Sequence[UUID],
        operation: str,
    ) -> ProvenanceRecord:
        return ProvenanceRecord(
            record_id=self.ids.new(),
            source_type=source_type,
            source_ids=tuple(source_ids),
            operation=operation,
            module=MODULE,
            model_version=self.model_version,
            timestamp=t,
            parent_records=tuple(dict.fromkeys(parents)),
            subject_id=subject,
        )

    def root_of(self, belief_id: UUID) -> UUID | None:
        h = self.heads.get(belief_id)
        return None if h is None else h.root

    def store_records(self, records: Sequence[ProvenanceRecord]) -> None:
        if self.repo is not None and records:
            self.repo.put_provenance(self.run_id, records)

    def commit(self, spec: CommitSpec, availability: Availability) -> BeliefMessage:
        head = self.heads.get(spec.belief_id)
        revision = 0 if head is None else head.revision + 1
        root = spec.records[-1].record_id
        t = spec.timestamp
        late = head is not None and spec.kind is UpdateKind.DIRECT and t.time_ns < head.time_ns
        status = summarize_status(spec.claims) if spec.claims else KnowledgeStatus.UNKNOWN
        consumed = tuple(e.evidence_id for e in spec.evidence) if spec.kind is UpdateKind.DIRECT else ()
        prior_observations = (
            head.message.independent_observation_count if head is not None and head.message is not None else 0
        )
        observation_count = prior_observations + (1 if consumed else 0)
        cell = BeliefCell(
            belief_id=spec.belief_id,
            domain=Domain.ECOLOGICAL,
            entity_type=spec.entity_type,
            registry_entity_id=spec.registry_id,
            lifecycle=spec.lifecycle,
            revision=revision,
            timestamp=t,
            state_embedding=spec.embedding,
            claims=spec.claims,
            knowledge_status=status,
            uncertainty=spec.uncertainty,
            spatial_support=spec.support,
            independent_observation_count=observation_count,
            provenance_root=root,
            model_version=self.model_version,
        )
        rev = BeliefRevision(
            belief_id=spec.belief_id,
            revision=revision,
            predecessor_revision=None if revision == 0 else revision - 1,
            measurement_time_ns=t.time_ns,
            consumed_evidence_ids=consumed,
            provenance_root=root,
            update_kind=spec.kind,
            cell=cell,
            late=late,
        )
        message_id = self.ids.new()
        if self.repo is not None:
            for ev in spec.evidence:
                self.repo.put_evidence(ev)
            self.repo.commit_update(
                BeliefUpdate(
                    message_id=message_id,
                    producer_version=self.model_version,
                    run_id=self.run_id,
                    revision=rev,
                    provenance=spec.records,
                )
            )
        self.commits += 1
        msg = BeliefMessage(
            message_id=message_id,
            belief_id=spec.belief_id,
            revision=revision,
            independent_observation_count=observation_count,
            world_entity_id=spec.registry_id,
            domain=Domain.ECOLOGICAL,
            timestamp=t,
            state_summary=spec.claims,
            state_embedding=spec.embedding,
            knowledge_status=status,
            uncertainty=spec.uncertainty,
            evidence_support=consumed,
            change_summary=spec.change_summary,
            provenance_refs=(root,),
            spatial_support=spec.support,
            lifecycle=spec.lifecycle,
            model_version=self.model_version,
            publisher_availability=availability,
            ecological=spec.payload,
        )
        self.heads[spec.belief_id] = Head(
            revision=revision,
            root=root,
            time_ns=max(t.time_ns, head.time_ns if head else 0),
            lifecycle=spec.lifecycle,
            entity_type=spec.entity_type,
            registry_id=spec.registry_id,
            message=msg,
        )
        return msg


def next_lifecycle(old: Lifecycle, hits: int, registered: bool) -> Lifecycle:
    """CONFIRMED (registry-backed or 2+ hits) -> ACTIVE once evidence arrives."""
    if old is Lifecycle.CANDIDATE:
        return Lifecycle.CONFIRMED if (registered or hits >= 2) else Lifecycle.CANDIDATE
    if old in (Lifecycle.CONFIRMED, Lifecycle.DORMANT) and hits > 0:
        return Lifecycle.ACTIVE if old is Lifecycle.CONFIRMED else Lifecycle.REACTIVATED
    return old
