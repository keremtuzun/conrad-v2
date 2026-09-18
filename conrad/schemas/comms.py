"""Communication contracts for BAAC (ch16, ch19): link state, semantic deltas, information units.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import Field

from conrad.schemas.base import VersionedModel
from conrad.schemas.timebase import TimeStamp


class LinkStatus(str, Enum):
    UP = "UP"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"


class LinkState(VersionedModel):
    link_name: str
    timestamp: TimeStamp
    status: LinkStatus
    bandwidth_bps: float = Field(ge=0)
    latency_s: float = Field(ge=0)
    packet_loss: float = Field(ge=0, le=1)
    bit_error_rate: float = Field(ge=0, le=1)
    energy_per_bit_j: float | None = Field(default=None, ge=0)
    range_m: float | None = Field(default=None, ge=0)


class CommunicationState(VersionedModel):
    timestamp: TimeStamp
    links: tuple[LinkState, ...]
    queue_depth: int = Field(ge=0)
    queued_bits: int = Field(ge=0)
    last_contact_ns: int | None = None


class DeltaType(str, Enum):
    NEW_BELIEF = "NEW_BELIEF"
    STATE_CHANGED = "STATE_CHANGED"
    UNCERTAINTY_CHANGED = "UNCERTAINTY_CHANGED"
    CONTRADICTION_ADDED = "CONTRADICTION_ADDED"
    EVIDENCE_ADDED = "EVIDENCE_ADDED"
    RELATIONSHIP_CHANGED = "RELATIONSHIP_CHANGED"
    BELIEF_RETIRED = "BELIEF_RETIRED"


class SemanticDelta(VersionedModel):
    delta_type: DeltaType
    belief_id: UUID
    base_revision: int | None = Field(description="receiver-known revision the delta applies to")
    new_revision: int = Field(ge=0)
    changed_fields: dict[str, Any]


class Fidelity(int, Enum):
    F0_CRITICAL_ALERT = 0
    F1_STRUCTURED_BELIEF = 1
    F2_EVIDENCE_SUMMARY = 2
    F3_COMPRESSED_EVIDENCE = 3
    F4_RAW_EVIDENCE = 4


class FidelityOption(VersionedModel):
    fidelity: Fidelity
    size_bits: int = Field(gt=0)
    information_retained: float = Field(ge=0, le=1)


class InformationType(str, Enum):
    ALERT = "ALERT"
    BELIEF_DELTA = "BELIEF_DELTA"
    EVIDENCE = "EVIDENCE"
    DECISION = "DECISION"
    HEALTH = "HEALTH"


class InformationUnit(VersionedModel):
    unit_id: UUID
    trace_id: UUID
    content_type: InformationType
    belief_ids: tuple[UUID, ...] = ()
    evidence_ids: tuple[UUID, ...] = ()
    semantic_delta: SemanticDelta | None = None
    priority: float = Field(ge=0, le=1)
    mission_value: float = Field(ge=0)
    created_time_ns: int = Field(ge=0)
    deadline_ns: int | None = None
    fidelity_levels: tuple[FidelityOption, ...] = Field(min_length=1)
    dependencies: tuple[UUID, ...] = ()
    provenance_id: UUID


class Transmission(VersionedModel):
    transmission_id: UUID
    unit_id: UUID
    trace_id: UUID
    link_name: str
    fidelity: Fidelity
    bits: int = Field(gt=0)
    sent_time_ns: int = Field(ge=0)
    delivered: bool
    delivered_time_ns: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
