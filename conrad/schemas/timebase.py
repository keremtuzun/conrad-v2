"""Common time semantics (ch2 Time, ch28 Time/Frames/Identity).

All times are UTC integer nanoseconds inside a declared clock domain. Acquisition (measurement)
time is distinct from creation/publication time.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from pydantic import Field, model_validator

from conrad.schemas.base import ConradModel

NS_PER_S = 1_000_000_000


class TimeStamp(ConradModel):
    time_ns: int = Field(ge=0, description="measurement/acquisition time, integer ns")
    clock_domain: str = Field(min_length=1)
    sequence_index: int = Field(default=0, ge=0)
    mission_time_ns: int | None = Field(default=None, ge=0)
    measurement_start_ns: int | None = Field(default=None, ge=0)
    measurement_end_ns: int | None = Field(default=None, ge=0)
    time_uncertainty_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _interval_ordered(self) -> TimeStamp:
        if (
            self.measurement_start_ns is not None
            and self.measurement_end_ns is not None
            and self.measurement_end_ns < self.measurement_start_ns
        ):
            raise ValueError("measurement_end_ns precedes measurement_start_ns")
        return self

    @property
    def seconds(self) -> float:
        return self.time_ns / NS_PER_S

    def delta_s(self, earlier: TimeStamp) -> float:
        """Physical elapsed seconds since ``earlier``. Clock domains must match."""
        if earlier.clock_domain != self.clock_domain:
            raise ClockDomainError(f"cannot subtract {earlier.clock_domain!r} from {self.clock_domain!r}")
        return (self.time_ns - earlier.time_ns) / NS_PER_S


class ClockDomainError(ValueError):
    """Two timestamps from different clock domains were combined without a declared mapping."""


def stamp(
    time_s: float, clock_domain: str, sequence_index: int = 0, mission_time_s: float | None = None
) -> TimeStamp:
    return TimeStamp(
        time_ns=round(time_s * NS_PER_S),
        clock_domain=clock_domain,
        sequence_index=sequence_index,
        mission_time_ns=None if mission_time_s is None else round(mission_time_s * NS_PER_S),
    )
