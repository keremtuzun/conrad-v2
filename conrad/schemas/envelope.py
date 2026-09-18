"""Message envelope (ch28). Distinguishes acquisition time from processing/publication time.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field, model_validator

from conrad.schemas.base import VersionedModel


class MessageEnvelope(VersionedModel):
    message_id: UUID
    scenario_id: UUID | None = None
    mission_id: UUID
    run_id: UUID
    measurement_time_ns: int = Field(ge=0)
    created_time_ns: int = Field(ge=0)
    clock_domain: str = Field(min_length=1)
    time_uncertainty_ns: int | None = Field(default=None, ge=0)
    producer_version: str = Field(min_length=1)
    correlation_id: UUID
    causation_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def _created_after_measurement(self) -> MessageEnvelope:
        # ch2 invariant "timestamp_evidence >= timestamp_observation" applies to creation time (ch28).
        if self.created_time_ns < self.measurement_time_ns:
            raise ValueError("created_time_ns precedes measurement_time_ns")
        return self

    @property
    def processing_latency_ns(self) -> int:
        return self.created_time_ns - self.measurement_time_ns
