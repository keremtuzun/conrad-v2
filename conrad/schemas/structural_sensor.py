"""Shared synthetic structural sensor contract; no default physical calibration."""

from __future__ import annotations

from pydantic import Field, model_validator

from conrad.schemas.base import ConradModel
from conrad.schemas.structural_support import ParameterAuthority


class StructuralSensorModel(ConradModel):
    version: str = "structural-sensor-v1"
    footprint_width_m: float = Field(gt=0)
    footprint_height_m: float = Field(gt=0)
    minimum_resolvable_corrosion_m: float = Field(gt=0)
    minimum_resolvable_crack_m: float = Field(gt=0)
    range_min_m: float = Field(ge=0)
    range_max_m: float = Field(gt=0)
    noise_sigma_m: float = Field(ge=0)
    position_uncertainty_m: float | None = Field(default=None, ge=0)
    orientation_uncertainty_rad: float | None = Field(default=None, ge=0)
    footprint_uncertainty_m: float | None = Field(default=None, ge=0)
    aggregation_kernel: str = "AREA_MEAN"
    authority: ParameterAuthority

    @model_validator(mode="after")
    def _valid(self) -> StructuralSensorModel:
        if self.range_max_m <= self.range_min_m:
            raise ValueError("range_max_m must exceed range_min_m")
        if self.aggregation_kernel not in ("AREA_MEAN", "LOCAL_MAX"):
            raise ValueError("unsupported structural aggregation kernel")
        return self

    @property
    def digest(self) -> str:
        return self.content_digest()
