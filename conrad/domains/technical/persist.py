"""BeliefCell / BeliefMessage construction and Repository commits for Model2T."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from conrad.domains.technical.claims import build_claims, state_embedding, technical_payload
from conrad.domains.technical.config import Model2TConfig
from conrad.domains.technical.state import ComponentBelief
from conrad.persistence.repository import BeliefUpdate, CommitResult, Repository
from conrad.schemas.belief import (
    Availability,
    BeliefCell,
    BeliefMessage,
    BeliefRevision,
    KnowledgeStatus,
    Relationship,
    UpdateKind,
    summarize_status,
)
from conrad.schemas.provenance import ProvenanceRecord
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain


def build_cell(belief: ComponentBelief, cfg: Model2TConfig, now: TimeStamp) -> BeliefCell:
    claims = build_claims(belief, cfg.condition)
    assert belief.provenance_root is not None
    return BeliefCell(
        belief_id=belief.belief_id,
        domain=Domain.TECHNICAL,
        entity_type=belief.spec.component_type,
        registry_entity_id=belief.spec.registry_id,
        lifecycle=belief.lifecycle,
        revision=max(belief.revision, 0),
        timestamp=now,
        state_embedding=state_embedding(belief),
        claims=claims,
        knowledge_status=summarize_status(claims),
        uncertainty=belief.uncertainty(),
        independent_observation_count=len(belief.groups),
        provenance_root=belief.provenance_root,
        model_version=cfg.model_version,
    )


def build_message(
    belief: ComponentBelief,
    cfg: Model2TConfig,
    message_id: UUID,
    now: TimeStamp,
    *,
    evidence_support: Sequence[UUID] = (),
    change_summary: str | None = None,
    prediction_summary: str | None = None,
    availability: Availability = Availability.AVAILABLE,
) -> BeliefMessage:
    cell = build_cell(belief, cfg, now)
    status = cell.knowledge_status
    return BeliefMessage(
        message_id=message_id,
        belief_id=belief.belief_id,
        revision=max(belief.revision, 0),
        world_entity_id=belief.spec.registry_id,
        domain=Domain.TECHNICAL,
        timestamp=now,
        state_summary=cell.claims,
        state_embedding=cell.state_embedding,
        knowledge_status=status,
        uncertainty=cell.uncertainty,
        evidence_support=() if status is KnowledgeStatus.PREDICTED else tuple(evidence_support),
        evidence_conflicts=tuple(belief.conflicts[-16:]),
        change_summary=change_summary,
        prediction_summary=prediction_summary,
        relationships=belief.relationship_ids,
        provenance_refs=(cell.provenance_root,),
        lifecycle=belief.lifecycle,
        model_version=cfg.model_version,
        publisher_availability=availability,
        technical=technical_payload(belief, cfg.condition, cell.claims),
    )


def commit_revision(
    repo: Repository,
    run_id: UUID,
    belief: ComponentBelief,
    cfg: Model2TConfig,
    now: TimeStamp,
    *,
    kind: UpdateKind,
    message_id: UUID,
    provenance: Sequence[ProvenanceRecord],
    consumed: Sequence[UUID] = (),
    measurement_time_ns: int | None = None,
    relationships: Sequence[Relationship] = (),
) -> CommitResult:
    """Advance ``belief.revision`` and commit one revision atomically with its provenance."""
    rev = belief.revision + 1
    t_ns = now.time_ns if measurement_time_ns is None else measurement_time_ns
    late = t_ns < belief.head_time_ns
    belief.revision = rev
    cell = build_cell(belief, cfg, now)
    revision = BeliefRevision(
        belief_id=belief.belief_id,
        revision=rev,
        predecessor_revision=None if rev == 0 else rev - 1,
        measurement_time_ns=t_ns,
        consumed_evidence_ids=tuple(consumed) if kind in (UpdateKind.DIRECT, UpdateKind.CREATE) else (),
        provenance_root=cell.provenance_root,
        update_kind=kind,
        cell=cell,
        late=late,
    )
    result = repo.commit_update(
        BeliefUpdate(
            message_id=message_id,
            producer_version=cfg.model_version,
            run_id=run_id,
            revision=revision,
            provenance=tuple(provenance),
            relationships=tuple(relationships),
        )
    )
    belief.head_time_ns = max(belief.head_time_ns, t_ns)
    return result
