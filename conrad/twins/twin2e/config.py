"""Twin2E / MEIFE configuration. TRUTH PLANE. Every dimension, rate limit and switch lives here (ch33).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from conrad.schemas.base import ConradModel


class BoundaryCondition(str, Enum):
    CLOSED = "CLOSED"  # zero normal flux: tracer mass is conserved exactly
    OPEN = "OPEN"  # upwind outflow, inflow carries the boundary value
    PERIODIC = "PERIODIC"


class ObservationLevel(str, Enum):
    E0_ABSTRACT = "E0_ABSTRACT"
    E1_FEATURE = "E1_FEATURE"
    E2_SENSOR = "E2_SENSOR"  # owned by the renderer; Twin2E refuses instead of faking it


class GridConfig(ConradModel):
    frame_id: str = "WORLD"
    origin_m: tuple[float, float, float]
    spacing_m: tuple[float, float, float]
    shape: tuple[int, int, int]


class MeifeSwitches(ConradModel):
    """Ablation switches (ch13 'And ablate'). All True = full MEIFE."""

    field_dynamics: bool = True
    entity_dynamics: bool = True
    field_to_entity: bool = True
    entity_to_field: bool = True
    multi_scale: bool = True
    disturbances: bool = True
    recovery: bool = True
    stochasticity: bool = True
    random_ecology: bool = False  # baseline: independent-noise ecology, no mechanism


class LearnedResidualConfig(ConradModel):
    """ch33: no learned Twin2E component before calibration data exists."""

    enabled: bool = False
    calibration_data_ref: str | None = None


class SolverConfig(ConradModel):
    cfl_max: float = Field(default=0.8, gt=0, le=1.0)
    max_substeps: int = Field(default=4000, ge=1)
    boundary: BoundaryCondition = BoundaryCondition.OPEN
    nudging_time_s: float = Field(default=1800.0, gt=0, description="regional -> local relaxation")
    upward_nudging_time_s: float = Field(default=21600.0, gt=0, description="local -> regional")
    field_noise_fraction: float = Field(default=0.002, ge=0, description="eps_Phi per sqrt(hour)")


class EcologyConfig(ConradModel):
    time_scale: float = Field(
        default=1.0, gt=0, description="ecological seconds per simulated second (logged acceleration)"
    )
    max_cover_rate_per_day: float = Field(default=0.05, gt=0, description="rate limit on |d cover|")
    max_condition_rate_per_day: float = Field(default=0.10, gt=0)
    max_transition_rate_per_day: float = Field(default=6.0, gt=0, description="presence flips")
    growth_noise_sigma_per_sqrt_day: float = Field(default=0.05, ge=0)
    random_ecology_sigma_per_sqrt_day: float = Field(default=0.05, ge=0)


class ObservationConfig(ConradModel):
    level: ObservationLevel = ObservationLevel.E0_ABSTRACT
    survey_range_m: float = Field(default=6.0, gt=0)
    detection_p0: float = Field(default=0.95, ge=0, le=1)
    cover_sigma: float = Field(default=0.04, ge=0)
    range_sigma_m: float = Field(default=0.10, ge=0)
    beam_attenuation_per_m_per_ntu: float = Field(default=0.04, ge=0)
    beam_attenuation_clear_per_m: float = Field(default=0.05, ge=0)
    feature_dim: int = Field(default=16, ge=4)
    sensor_sigma: dict[str, float] = Field(
        default_factory=lambda: {"temperature": 0.02, "turbidity": 0.15, "current": 0.01, "light": 5.0}
    )


class Twin2EConfig(ConradModel):
    regional_grid: GridConfig = GridConfig(
        origin_m=(-1600.0, -1600.0, -80.0), spacing_m=(50.0, 50.0, 10.0), shape=(64, 64, 8)
    )
    local_grid: GridConfig = GridConfig(
        origin_m=(-80.0, -80.0, -80.0), spacing_m=(5.0, 5.0, 5.0), shape=(32, 32, 16)
    )
    switches: MeifeSwitches = MeifeSwitches()
    solver: SolverConfig = SolverConfig()
    ecology: EcologyConfig = EcologyConfig()
    observation: ObservationConfig = ObservationConfig()
    learned_residual: LearnedResidualConfig = LearnedResidualConfig()
    clock_domain: str = "SIM"
    allow_prior_sampling: bool = True


def load_config(path: str | Path) -> Twin2EConfig:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Twin2EConfig.model_validate(data.get("twin2e", data))
