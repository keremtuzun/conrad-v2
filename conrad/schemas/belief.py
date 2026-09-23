"""Belief-plane contracts: cell, revision, message, query, relationship (ch2, ch3, ch8, ch28).

A belief ID is inference-owned. A known asset-registry identity may be associated when
justified; a simulator hidden entity ID must never become a belief ID (INV-ARCH-07).

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from conrad.schemas.base import VersionedModel
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.uncertainty import Uncertainty
from conrad.schemas.world import CrossDomainRelationType, Domain


class KnowledgeStatus(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    PREDICTED = "PREDICTED"
    UNKNOWN = "UNKNOWN"
    MIXED = "MIXED"


class Lifecycle(str, Enum):
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    REACTIVATED = "REACTIVATED"
    MERGED = "MERGED"
    SPLIT = "SPLIT"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


ALLOWED_LIFECYCLE_TRANSITIONS: dict[Lifecycle, frozenset[Lifecycle]] = {
    Lifecycle.CANDIDATE: frozenset({Lifecycle.CONFIRMED, Lifecycle.REJECTED, Lifecycle.MERGED}),
    Lifecycle.CONFIRMED: frozenset(
        {Lifecycle.ACTIVE, Lifecycle.DORMANT, Lifecycle.MERGED, Lifecycle.SPLIT, Lifecycle.RETIRED}
    ),
    Lifecycle.ACTIVE: frozenset({Lifecycle.DORMANT, Lifecycle.MERGED, Lifecycle.SPLIT, Lifecycle.RETIRED}),
    Lifecycle.DORMANT: frozenset({Lifecycle.REACTIVATED, Lifecycle.RETIRED, Lifecycle.MERGED}),
    Lifecycle.REACTIVATED: frozenset(
        {Lifecycle.ACTIVE, Lifecycle.DORMANT, Lifecycle.MERGED, Lifecycle.SPLIT, Lifecycle.RETIRED}
    ),
    Lifecycle.MERGED: frozenset(),
    Lifecycle.SPLIT: frozenset(),
    Lifecycle.RETIRED: frozenset(),
    Lifecycle.REJECTED: frozenset(),
}


class UpdateKind(str, Enum):
    CREATE = "CREATE"
    DIRECT = "DIRECT"
    RELATIONAL = "RELATIONAL"
    PREDICTED = "PREDICTED"
    CONTEXT = "CONTEXT"
    LIFECYCLE = "LIFECYCLE"


class Availability(str, Enum):
    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"


class Relationship(VersionedModel):
    relationship_id: UUID
    relation_type: str = Field(min_length=1)
    source_belief_id: UUID
    target_belief_id: UUID
    source_domain: Domain
    target_domain: Domain
    confidence: float = Field(ge=0, le=1)
    uncertainty: Uncertainty
    provenance_id: UUID
    attributes: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _cross_domain_typed(self) -> Relationship:
        if self.source_domain != self.target_domain:
            allowed = {t.value for t in CrossDomainRelationType}
            if self.relation_type not in allowed:
                raise ValueError(f"cross-domain relation type {self.relation_type!r} is not a justified type")
        if self.source_belief_id == self.target_belief_id:
            raise ValueError("self-relationship")
        return self


class PropertyClaim(VersionedModel):
    """One interpretable property with its own knowledge status (ch28: status is property-level)."""

    name: str
    value: float | str | bool | None
    units: str | None = None
    status: KnowledgeStatus
    uncertainty: Uncertainty
    provenance_id: UUID | None = None

    @model_validator(mode="after")
    def _unknown_has_no_value(self) -> PropertyClaim:
        if self.status is KnowledgeStatus.UNKNOWN and self.value is not None:
            raise ValueError("UNKNOWN is an epistemic status, not a value; value must be None")
        if self.status is not KnowledgeStatus.UNKNOWN and self.provenance_id is None:
            raise ValueError("a non-UNKNOWN claim requires provenance")
        return self


def summarize_status(claims: tuple[PropertyClaim, ...]) -> KnowledgeStatus:
    statuses = {c.status for c in claims}
    if not statuses:
        return KnowledgeStatus.UNKNOWN
    if len(statuses) == 1:
        return next(iter(statuses))
    return KnowledgeStatus.MIXED


class BeliefCell(VersionedModel):
    belief_id: UUID
    domain: Domain
    entity_type: str
    registry_entity_id: UUID | None = None
    lifecycle: Lifecycle
    revision: int = Field(ge=0)
    timestamp: TimeStamp
    state_embedding: tuple[float, ...]
    temporal_state: tuple[float, ...] = ()
    history_state: tuple[float, ...] = ()
    claims: tuple[PropertyClaim, ...] = ()
    knowledge_status: KnowledgeStatus
    uncertainty: Uncertainty
    spatial_support: SpatialSupport | None = None
    independent_observation_count: int = Field(default=0, ge=0)
    lineage_parent_ids: tuple[UUID, ...] = ()
    provenance_root: UUID
    model_version: str

    @model_validator(mode="after")
    def _status_matches_claims(self) -> BeliefCell:
        if self.claims and self.knowledge_status is not summarize_status(self.claims):
            raise ValueError("knowledge_status disagrees with property-level claim statuses")
        return self

    def claim(self, name: str) -> PropertyClaim | None:
        for c in self.claims:
            if c.name == name:
                return c
        return None


class BeliefRevision(VersionedModel):
    belief_id: UUID
    revision: int = Field(ge=0)
    predecessor_revision: int | None
    measurement_time_ns: int = Field(ge=0)
    consumed_evidence_ids: tuple[UUID, ...] = ()
    provenance_root: UUID
    update_kind: UpdateKind
    cell: BeliefCell
    late: bool = Field(
        default=False, description="evidence older than the belief head; handled by declared policy"
    )

    @model_validator(mode="after")
    def _consistent(self) -> BeliefRevision:
        if self.cell.belief_id != self.belief_id or self.cell.revision != self.revision:
            raise ValueError("revision header disagrees with embedded cell")
        if self.revision == 0 and self.predecessor_revision is not None:
            raise ValueError("revision 0 has no predecessor")
        if self.revision > 0 and self.predecessor_revision != self.revision - 1:
            raise ValueError("revision must reference its immediate predecessor")
        if self.update_kind is UpdateKind.DIRECT and not self.consumed_evidence_ids:
            raise ValueError("DIRECT update without consumed evidence")
        if self.update_kind in (UpdateKind.RELATIONAL, UpdateKind.PREDICTED) and self.consumed_evidence_ids:
            raise ValueError("relational/predicted updates must not claim direct evidence")
        return self


class TechnicalPayload(VersionedModel):
    condition: str | None = None
    degradation_type: str | None = None
    severity: float | None = Field(default=None, ge=0, le=1)
    corrosion_depth_m: float | None = Field(default=None, ge=0)
    crack_length_m: float | None = Field(default=None, ge=0)
    direct_support: float = Field(default=0.0, ge=0, le=1)
    propagated_support: float = Field(default=0.0, ge=0, le=1)


class EcologicalPayload(VersionedModel):
    kind: str = Field(description="ENTITY or FIELD; the two are different state types")
    quantities: dict[str, float] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)


class SpatialPayload(VersionedModel):
    occupancy_probability: float | None = Field(default=None, ge=0, le=1)
    coverage: float = Field(default=0.0, ge=0, le=1)
    resolution_m: float | None = Field(default=None, gt=0)
    semantic_class: str | None = None
    observation_count: int = Field(default=0, ge=0)


class BeliefMessage(VersionedModel):
    """Primary object on the Belief Bus. Model1 consumes these, never domain internals."""

    message_id: UUID
    belief_id: UUID
    revision: int = Field(ge=0)
    independent_observation_count: int = Field(
        default=0,
        ge=0,
        description="Cumulative independent DIRECT observations represented by this belief head.",
    )
    world_entity_id: UUID | None = Field(default=None, description="registry identity when known; else None")
    domain: Domain
    timestamp: TimeStamp
    state_summary: tuple[PropertyClaim, ...]
    state_embedding: tuple[float, ...]
    knowledge_status: KnowledgeStatus
    uncertainty: Uncertainty
    evidence_support: tuple[UUID, ...] = ()
    evidence_conflicts: tuple[UUID, ...] = ()
    change_summary: str | None = None
    prediction_summary: str | None = None
    relationships: tuple[UUID, ...] = ()
    provenance_refs: tuple[UUID, ...] = Field(min_length=1)
    spatial_support: SpatialSupport | None = None
    lifecycle: Lifecycle
    model_version: str
    publisher_availability: Availability = Availability.AVAILABLE
    technical: TechnicalPayload | None = None
    ecological: EcologicalPayload | None = None
    spatial: SpatialPayload | None = None

    @model_validator(mode="after")
    def _payload_matches_domain(self) -> BeliefMessage:
        expected = {Domain.TECHNICAL: "technical", Domain.ECOLOGICAL: "ecological", Domain.SPATIAL: "spatial"}
        for domain, attr in expected.items():
            if domain is not self.domain and getattr(self, attr) is not None:
                raise ValueError(f"{attr} payload on a {self.domain.value} belief message")
        if self.knowledge_status is KnowledgeStatus.PREDICTED and self.evidence_support:
            raise ValueError("a PREDICTED belief cannot list direct evidence support")
        return self


class BeliefQuery(VersionedModel):
    domain: Domain | None = None
    region: SpatialSupport | None = None
    entity_ids: tuple[UUID, ...] = ()
    belief_ids: tuple[UUID, ...] = ()
    time_range_ns: tuple[int, int] | None = None
    requested_fields: tuple[str, ...] = ()
    min_observational_uncertainty: float | None = Field(default=None, ge=0)
    min_contradiction_uncertainty: float | None = Field(default=None, ge=0)
    include_provenance: bool = False
    include_predictions: bool = True
    max_results: int = Field(default=256, gt=0)


class BeliefSnapshot(VersionedModel):
    """Coherent query reply. Carries revisions so consumers can detect mixed-time inputs (ch28)."""

    snapshot_id: UUID
    created_time_ns: int = Field(ge=0)
    messages: tuple[BeliefMessage, ...]
    domain_availability: dict[str, Availability]
    provenance: dict[str, Any] = Field(default_factory=dict)
