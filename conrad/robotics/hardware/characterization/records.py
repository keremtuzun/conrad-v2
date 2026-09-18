"""Hardware characterization records: one sourced value per record, with full provenance (ch22 handoff).

Every record states value, units, frame, acquisition time, uncertainty/range, method, provenance and
source kind. MEASURED values must name a raw log and carry its sha256 digest; OPEN carries no value.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from conrad.schemas.base import VersionedModel
from conrad.schemas.robot import SourceKind
from conrad.schemas.timebase import TimeStamp

_SHA256 = "^[0-9a-f]{64}$"


class Category(str, Enum):
    GEOMETRY = "geometry"
    MASS = "mass"
    INERTIA = "inertia"
    COM = "com"
    COB = "cob"
    DISPLACED_VOLUME = "displaced_volume"
    HYDRODYNAMICS = "hydrodynamics"
    THRUSTERS = "thrusters"
    SENSORS = "sensors"
    COMMUNICATIONS = "communications"
    POWER = "power"
    COMPUTE = "compute"
    SAFETY = "safety"
    HEALTH = "health"


class CharacterizationRecord(VersionedModel):
    record_id: str = Field(min_length=1)
    category: Category
    target: str = Field(
        min_length=1,
        description="RobotConfig path, e.g. mass_kg | thrusters[H1].thrust_coefficient | sensors[imu].noise_std;"
        " health records use health.<device>",
    )
    value: float | tuple[float, ...] | str | None
    units: str = Field(min_length=1)
    frame_id: str | None = None
    timestamp: TimeStamp | None = None
    uncertainty_1sigma: float | None = Field(default=None, ge=0)
    valid_range: tuple[float, float] | None = None
    source: SourceKind
    method: str = Field(min_length=1, description="how the value was obtained (procedure / fit / datasheet)")
    provenance: str | None = Field(default=None, description="report / citation / operator reference")
    raw_log: str | None = Field(default=None, description="path of the raw log, relative to the handoff root")
    raw_log_digest: str | None = Field(default=None, pattern=_SHA256)
    operator: str | None = None

    @model_validator(mode="after")
    def _source_rules(self) -> CharacterizationRecord:
        if self.source is SourceKind.OPEN:
            if self.value is not None:
                raise ValueError(f"{self.record_id}: an OPEN record cannot carry a value")
            return self
        if self.value is None:
            raise ValueError(f"{self.record_id}: {self.source.value} record requires a value")
        if self.source is SourceKind.MEASURED:
            if not self.raw_log or not self.raw_log_digest:
                raise ValueError(f"{self.record_id}: MEASURED requires raw_log and raw_log_digest")
            if self.timestamp is None:
                raise ValueError(f"{self.record_id}: MEASURED requires an acquisition timestamp")
        if self.source is SourceKind.IDENTIFIED and not (self.raw_log_digest or self.provenance):
            raise ValueError(
                f"{self.record_id}: IDENTIFIED requires an identification report or raw-log digest"
            )
        if self.source is SourceKind.LITERATURE_PRIOR and not self.provenance:
            raise ValueError(f"{self.record_id}: LITERATURE_PRIOR requires a citation in provenance")
        if self.valid_range is not None and self.valid_range[0] > self.valid_range[1]:
            raise ValueError(f"{self.record_id}: valid_range is inverted")
        return self

    def provenance_string(self) -> str:
        parts = [f"record:{self.record_id}", f"method:{self.method}"]
        if self.raw_log_digest:
            parts.append(f"raw-log:{self.raw_log}@sha256:{self.raw_log_digest}")
        if self.provenance:
            parts.append(f"ref:{self.provenance}")
        return ";".join(parts)


class CharacterizationBundle(VersionedModel):
    """Everything one handoff delivered. ``root`` is where raw logs were resolved and digested."""

    bundle_id: str = Field(min_length=1)
    robot_config_name: str
    root: str
    records: tuple[CharacterizationRecord, ...]

    @model_validator(mode="after")
    def _unique(self) -> CharacterizationBundle:
        ids = [r.record_id for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate record_id in bundle")
        targets = [r.target for r in self.records if not r.target.startswith("health.")]
        dup = {t for t in targets if targets.count(t) > 1}
        if dup:
            raise ValueError(f"several records target the same parameter: {sorted(dup)}")
        return self
