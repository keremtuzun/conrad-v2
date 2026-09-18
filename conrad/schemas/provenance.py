"""First-class provenance DAG (ch2 Provenance, ch28 Belief Updates and Provenance).

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from uuid import UUID

from pydantic import Field, model_validator

from conrad.schemas.base import ConradModel
from conrad.schemas.timebase import TimeStamp


class SourceType(str, Enum):
    DIRECT_OBSERVATION = "DIRECT_OBSERVATION"
    RELATIONAL_INFERENCE = "RELATIONAL_INFERENCE"
    TEMPORAL_PREDICTION = "TEMPORAL_PREDICTION"
    CROSS_DOMAIN_CONTEXT = "CROSS_DOMAIN_CONTEXT"
    PRIOR = "PRIOR"
    # Non-belief nodes used to extend the chain upward to decisions and commands (prompt s105).
    DECISION = "DECISION"
    PLAN = "PLAN"
    COMMAND = "COMMAND"
    SENSOR_ARTIFACT = "SENSOR_ARTIFACT"


class ProvenanceRecord(ConradModel):
    record_id: UUID
    source_type: SourceType
    source_ids: tuple[UUID, ...] = Field(description="IDs of the objects this record derives from")
    operation: str = Field(min_length=1)
    module: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    timestamp: TimeStamp
    parent_records: tuple[UUID, ...] = ()
    subject_id: UUID | None = Field(default=None, description="object this record describes")

    @model_validator(mode="after")
    def _no_self_parent(self) -> ProvenanceRecord:
        if self.record_id in self.parent_records:
            raise ProvenanceError("provenance record lists itself as a parent")
        if self.source_type not in (SourceType.PRIOR, SourceType.SENSOR_ARTIFACT) and not (
            self.source_ids or self.parent_records
        ):
            raise ProvenanceError(f"{self.source_type.value} record has neither sources nor parents")
        return self


class ProvenanceError(ValueError):
    """The provenance graph is cyclic, dangling or otherwise invalid."""


def validate_provenance_dag(records: Mapping[UUID, ProvenanceRecord], roots: Iterable[UUID]) -> set[UUID]:
    """Walk from ``roots``; every parent must resolve and no cycle may exist. Returns reachable IDs."""
    visiting: set[UUID] = set()
    done: set[UUID] = set()

    def visit(rid: UUID) -> None:
        if rid in done:
            return
        if rid in visiting:
            raise ProvenanceError(f"cycle through provenance record {rid}")
        if rid not in records:
            raise ProvenanceError(f"dangling provenance reference {rid}")
        visiting.add(rid)
        for parent in records[rid].parent_records:
            visit(parent)
        visiting.discard(rid)
        done.add(rid)

    for root in roots:
        visit(root)
    return done


def trace_to_sources(records: Mapping[UUID, ProvenanceRecord], root: UUID) -> list[ProvenanceRecord]:
    """Topologically ordered chain (root first) down to leaf records."""
    reachable = validate_provenance_dag(records, [root])
    order: list[ProvenanceRecord] = []
    seen: set[UUID] = set()
    stack = [root]
    while stack:
        rid = stack.pop()
        if rid in seen:
            continue
        seen.add(rid)
        order.append(records[rid])
        stack.extend(p for p in records[rid].parent_records if p in reachable)
    return order
