"""Shared synthetic structural sensor contract; no default physical calibration."""

from __future__ import annotations

from pydantic import Field, model_validator

from conrad.schemas.base import ConradModel
from conrad.schemas.structural_support import ParameterAuthority

DETECTABILITY_VERSION = "spatial-synthetic-detectability-v1"


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
        if self.aggregation_kernel not in ("AREA_MEAN", "LOCAL_MAX", "RESOLUTION_CELL_SAMPLES"):
            raise ValueError("unsupported structural aggregation kernel")
        if self.aggregation_kernel == "RESOLUTION_CELL_SAMPLES" and not hasattr(self, "axial_resolution_m"):
            raise ValueError("resolution-cell response requires the v2 sensor model")
        return self

    @property
    def digest(self) -> str:
        return self.content_digest()


class StructuralSensorModelV2(StructuralSensorModel):
    """Synthetic spatial response; all dimensions are declared in surface metres.

    A resolution cell returns a detectable local peak. This is an explicit
    engineering hypothesis, not a measured response of a physical payload.
    """

    version: str = "structural-sensor-v2"
    aggregation_kernel: str = "RESOLUTION_CELL_SAMPLES"
    axial_resolution_m: float = Field(gt=0)
    lateral_resolution_m: float = Field(gt=0)

    @model_validator(mode="after")
    def _valid_v2(self) -> StructuralSensorModelV2:
        if self.aggregation_kernel != "RESOLUTION_CELL_SAMPLES":
            raise ValueError("v2 requires resolution-cell response")
        if self.authority is ParameterAuthority.UNKNOWN:
            raise ValueError("synthetic response requires declared parameter authority")
        return self
