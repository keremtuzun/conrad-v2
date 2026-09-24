"""Raw observation and structured evidence (ch2, ch3, ch4). Evidence is not belief.

Large payloads (images, sonar arrays, point clouds) travel by content-addressed reference.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from conrad.schemas.base import VersionedModel
from conrad.schemas.frames import Pose, SpatialSupport
from conrad.schemas.structural_support import CapsuleSurfaceSupport
from conrad.schemas.timebase import TimeStamp

MAX_INLINE_VALUES = 4096
"""Inline scalar payloads above this many numbers must go through the object store."""


class Modality(str, Enum):
    RGB = "RGB"
    SONAR = "SONAR"
    DEPTH_RANGE = "DEPTH_RANGE"
    POINT_CLOUD = "POINT_CLOUD"
    IMU = "IMU"
    PRESSURE_DEPTH = "PRESSURE_DEPTH"
    ENVIRONMENTAL = "ENVIRONMENTAL"
    STRUCTURED = "STRUCTURED"
    POWER = "POWER"


class SensorHealth(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAULT = "FAULT"
    UNKNOWN = "UNKNOWN"


class PayloadRef(VersionedModel):
    """Reference to an immutable object. ``digest`` is sha256 of the stored bytes."""

    uri: str = Field(min_length=1)
    digest: str = Field(pattern="^[0-9a-f]{64}$")
    media_type: str
    shape: tuple[int, ...] = ()
    dtype: str | None = None
    byte_length: int = Field(ge=0)


class Observation(VersionedModel):
    observation_id: UUID
    mission_id: UUID
    run_id: UUID
    trace_id: UUID
    sensor_id: UUID
    modality: Modality
    timestamp: TimeStamp
    sensor_frame: str = Field(min_length=1)
    robot_pose_estimate: Pose | None = Field(
        description="ESTIMATED pose; simulator true pose is never placed here"
    )
    payload_ref: PayloadRef | None = None
    inline_values: tuple[float, ...] | None = None
    inline_units: str | None = None
    sensor_health: SensorHealth = SensorHealth.OK
    calibration_ref: str | None = None
    sensor_context: dict[str, Any] = Field(default_factory=dict)
    structural_support: CapsuleSurfaceSupport | None = None

    @model_validator(mode="after")
    def _payload(self) -> Observation:
        if self.payload_ref is None and self.inline_values is None:
            raise ValueError("observation carries neither payload_ref nor inline_values")
        if self.inline_values is not None:
            if len(self.inline_values) > MAX_INLINE_VALUES:
                raise ValueError("inline payload too large; use payload_ref")
            if self.inline_units is None:
                raise ValueError("inline_values require inline_units")
        return self


class EvidenceValidity(str, Enum):
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


class QualityContext(VersionedModel):
    """Measured quality factors. ``None`` means not measured; it is never filled with a nominal value."""

    blur: float | None = Field(default=None, ge=0, le=1)
    backscatter: float | None = Field(default=None, ge=0, le=1)
    lighting: float | None = Field(default=None, ge=0, le=1)
    occlusion: float | None = Field(default=None, ge=0, le=1)
    range_m: float | None = Field(default=None, ge=0)
    incidence_angle_rad: float | None = None
    snr_db: float | None = None
    multipath: float | None = Field(default=None, ge=0, le=1)
    sensor_health: SensorHealth = SensorHealth.UNKNOWN
    pose_sigma_m: float | None = Field(default=None, ge=0)
    ood_score: float | None = Field(default=None, ge=0, le=1)


class EntityCandidate(VersionedModel):
    """Inference-side association hint. Never a simulator hidden ID (INV-ARCH-07)."""

    belief_id: UUID | None = None
    registry_entity_id: UUID | None = Field(
        default=None, description="known asset-registry identity, when justified"
    )
    score: float = Field(ge=0, le=1)


class Evidence(VersionedModel):
    evidence_id: UUID
    source_observation_id: UUID
    mission_id: UUID
    run_id: UUID
    trace_id: UUID
    modality: Modality
    timestamp: TimeStamp = Field(description="measurement time of the source observation, unchanged")
    created_time_ns: int = Field(ge=0)
    embedding: tuple[float, ...]
    embedding_ref: PayloadRef | None = None
    spatial_support: SpatialSupport | None = None
    structural_support: CapsuleSurfaceSupport | None = None
    entity_candidates: tuple[EntityCandidate, ...] = ()
    reliability: float = Field(
        ge=0, le=1, description="sensing reliability; distinct from semantic confidence"
    )
    aleatoric_uncertainty: float = Field(ge=0)
    validity: EvidenceValidity = EvidenceValidity.VALID
    sensor_context: QualityContext = QualityContext()
    measurements: dict[str, float] = Field(
        default_factory=dict, description="interpretable measured quantities, SI"
    )
    measurement_units: dict[str, str] = Field(default_factory=dict)
    independence_group: str | None = Field(
        default=None, description="evidence sharing a group is NOT independent corroboration"
    )
    provenance_id: UUID
    encoder_version: str

    @field_validator("embedding")
    @classmethod
    def _nonempty(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if len(v) == 0:
            raise ValueError("evidence embedding is empty")
        return v

    @model_validator(mode="after")
    def _units(self) -> Evidence:
        missing = set(self.measurements) - set(self.measurement_units)
        if missing:
            raise ValueError(f"measurements without declared units: {sorted(missing)}")
        if self.created_time_ns < self.timestamp.time_ns:
            raise ValueError("evidence created before its observation was measured")
        return self
