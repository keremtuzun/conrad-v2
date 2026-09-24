"""Versioned measured support on a capsule surface, independent of Model2T cells.

Coordinates use the surveyed component's axial origin and angular basis.  A
support is one non-wrapping rectangle; a footprint crossing the angular seam
is represented by two observations sharing one independence group.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import Field, model_validator

from conrad.schemas.base import VersionedModel

STRUCTURAL_SUPPORT_VERSION = "structural-observation-v2"


class ParameterAuthority(str, Enum):
    SPECIFIED = "SPECIFIED"
    MEASURED = "MEASURED"
    ENGINEERING_ESTIMATE = "ENGINEERING_ESTIMATE"
    UNKNOWN = "UNKNOWN"


class CapsuleSurfaceSupport(VersionedModel):
    """Physical surface contributing to one measurement, not credited belief cells."""

    support_version: str = STRUCTURAL_SUPPORT_VERSION
    sensor_model_version: str = Field(min_length=1)
    sensor_config_digest: str = Field(pattern="^[0-9a-f]{64}$")
    frame_id: str = Field(min_length=1)
    axial_start_m: float = Field(ge=0)
    axial_end_m: float = Field(gt=0)
    angle_start_rad: float = Field(ge=0, lt=2 * math.pi)
    angle_end_rad: float = Field(gt=0, le=2 * math.pi)
    axial_uncertainty_m: float | None = Field(default=None, ge=0)
    angular_uncertainty_rad: float | None = Field(default=None, ge=0)
    occlusion_clipped: bool = False
    aggregation_kernel: str = "AREA_MEAN"

    @model_validator(mode="after")
    def _valid(self) -> CapsuleSurfaceSupport:
        if self.axial_end_m <= self.axial_start_m:
            raise ValueError("empty axial support")
        if self.angle_end_rad <= self.angle_start_rad:
            raise ValueError("angular seam must be split into separate supports")
        if self.aggregation_kernel not in ("AREA_MEAN", "LOCAL_MAX", "RESOLUTION_CELL_SAMPLES"):
            raise ValueError("unsupported structural aggregation kernel")
        return self
