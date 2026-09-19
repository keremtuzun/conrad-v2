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

SURFACE_SUFFIX = "_READ_SURFACE"


def build_cell(
    belief: ComponentBelief, cfg: Model2TConfig, now: TimeStamp, *, surface: bool = False
) -> BeliefCell:
    """The component's cell, or (``surface=True``) the cell of its read-surface part: a sub-entity minted by
    Model2T, not a registry item, so its registry_entity_id is None."""
    claims = build_claims(belief, cfg.condition, surface=surface)
    assert belief.provenance_root is not None
    if surface:
        assert belief.surface_id is not None
    return BeliefCell(
        belief_id=belief.surface_id if surface and belief.surface_id else belief.belief_id,
        domain=Domain.TECHNICAL,
        entity_type=belief.spec.component_type + (SURFACE_SUFFIX if surface else ""),
        registry_entity_id=None if surface else belief.spec.registry_id,
        lifecycle=belief.lifecycle,
        revision=max(belief.surface_revision if surface else belief.revision, 0),
        timestamp=now,
        state_embedding=state_embedding(belief),
        claims=claims,
        knowledge_status=summarize_status(claims),
        uncertainty=belief.uncertainty(surface=surface),
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
    surface: bool = False,
) -> BeliefMessage:
    cell = build_cell(belief, cfg, now, surface=surface)
    status = cell.knowledge_status
    rels = (
        (belief.surface_relationship,)
        if surface and belief.surface_relationship is not None
        else belief.relationship_ids
    )
    return BeliefMessage(
        message_id=message_id,
        belief_id=cell.belief_id,
        revision=cell.revision,
        world_entity_id=cell.registry_entity_id,
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
        relationships=rels,
        provenance_refs=(cell.provenance_root,),
        lifecycle=belief.lifecycle,
        model_version=cfg.model_version,
        publisher_availability=availability,
        technical=technical_payload(belief, cfg.condition, cell.claims, surface=surface),
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
    surface: bool = False,
) -> CommitResult:
    """Advance the revision (of the component, or of its read-surface part) and commit it atomically with its
    provenance."""
    rev = (belief.surface_revision if surface else belief.revision) + 1
    t_ns = now.time_ns if measurement_time_ns is None else measurement_time_ns
    late = t_ns < belief.head_time_ns
    if surface:
        belief.surface_revision = rev
    else:
        belief.revision = rev
    cell = build_cell(belief, cfg, now, surface=surface)
    revision = BeliefRevision(
        belief_id=cell.belief_id,
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
