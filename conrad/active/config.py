"""MCBR configuration. Defaults are ENGINEERING_ESTIMATE simulation values (none is a measured quantity)."""

from __future__ import annotations

from pydantic import Field

from conrad.schemas.base import ConradModel


class CostWeights(ConradModel):
    time_per_s: float = Field(default=0.002, ge=0)
    energy_per_j: float = Field(default=0.0002, ge=0)
    risk: float = Field(default=1.0, ge=0)
    travel_per_m: float = Field(default=0.01, ge=0)


class MCBRConfig(ConradModel):
    model_version: str = "mcbr-analytic-0.2"
    max_candidates: int = Field(default=128, gt=0, le=128)
    standoff_fractions: tuple[float, ...] = Field(
        default=(0.25, 0.5, 0.8), description="standoff = min_range + f * (max_range - min_range) per sensor"
    )
    n_azimuth: int = Field(default=12, gt=0)
    elevations_rad: tuple[float, ...] = (0.0, 0.5)
    min_visibility: float = Field(default=0.15, ge=0, le=1)
    risk_limit: float = Field(default=0.5, ge=0, le=1)
    cause_threshold: float = Field(default=0.4, ge=0, description="channel level that makes a cause active")
    default_target_uncertainty: float = Field(
        default=0.25, ge=0, description="need is satisfied when every targeted channel is at or below this"
    )
    off_target_weight: float = Field(default=0.25, ge=0, le=1)
    epistemic_resolvability: float = Field(default=0.3, ge=0, le=1)
    independent_modality_bonus: float = Field(default=0.25, ge=0)
    redundancy_sigma_m: float = Field(default=1.0, gt=0)
    min_expected_gain: float = Field(default=0.02, ge=0)
    n_alternatives: int = Field(default=3, ge=0)
    sensor_aware_visibility: bool = Field(
        default=True,
        description="when the request carries a predictive belief, candidate visibility = mission-weighted "
        "fraction of target elements that the candidate's SENSOR is predicted to measure (range, incidence, "
        "believed occluders); False = the legacy sensor-agnostic region visibility only",
    )
    cost: CostWeights = CostWeights()
