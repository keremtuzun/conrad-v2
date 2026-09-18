"""Model2E configuration (ch12, ch33). Every constant is config; defaults are ENGINEERING_ESTIMATE.

BELIEF PLANE: no truth imports. Nothing here is calibrated against real data.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from conrad.schemas.base import ConradModel


class BeliefGridConfig(ConradModel):
    """Coarse belief grid (cell centres at origin + (i + 0.5) * spacing)."""

    frame_id: str = "WORLD"
    origin_m: tuple[float, float, float] = (-40.0, -40.0, -20.0)
    spacing_m: tuple[float, float, float] = (20.0, 20.0, 5.0)
    shape: tuple[int, int, int] = (4, 4, 4)


class FieldSpec(ConradModel):
    """One environmental field. ``components`` > 1 for vector fields (current)."""

    units: str
    components: int = Field(default=1, ge=1)
    prior_mean: float
    prior_sd: float = Field(gt=0, description="domain-wide level prior sd")
    local_sd: float = Field(gt=0, description="per-cell local deviation prior sd")
    length_scale_h_m: float = Field(gt=0, description="horizontal kernel correlation length")
    length_scale_v_m: float = Field(gt=0, description="vertical kernel correlation length")
    correlation_time_s: float = Field(gt=0, description="OU relaxation time of local deviations")
    level_correlation_time_s: float = Field(gt=0, description="OU relaxation time of the level")
    sensor_sd: float = Field(gt=0, description="datasheet sensor noise (not taken from the simulator)")
    nonnegative: bool = False


def _default_fields() -> dict[str, FieldSpec]:
    return {
        "temperature": FieldSpec(
            units="degC",
            prior_mean=17.0,
            prior_sd=6.0,
            local_sd=2.0,
            level_correlation_time_s=864000.0,
            length_scale_h_m=40.0,
            length_scale_v_m=6.0,
            correlation_time_s=86400.0,
            sensor_sd=0.05,
        ),
        "turbidity": FieldSpec(
            units="NTU",
            prior_mean=3.0,
            prior_sd=4.0,
            local_sd=4.0,
            level_correlation_time_s=86400.0,
            length_scale_h_m=30.0,
            length_scale_v_m=8.0,
            correlation_time_s=21600.0,
            sensor_sd=0.2,
            nonnegative=True,
        ),
        "current": FieldSpec(
            units="m s-1",
            components=3,
            prior_mean=0.0,
            prior_sd=0.3,
            local_sd=0.1,
            level_correlation_time_s=21600.0,
            length_scale_h_m=40.0,
            length_scale_v_m=10.0,
            correlation_time_s=3600.0,
            sensor_sd=0.02,
        ),
        "light": FieldSpec(
            units="W m-2",
            prior_mean=300.0,
            prior_sd=300.0,
            local_sd=150.0,
            level_correlation_time_s=21600.0,
            length_scale_h_m=60.0,
            length_scale_v_m=4.0,
            correlation_time_s=7200.0,
            sensor_sd=8.0,
            nonnegative=True,
        ),
    }


class EntityConfig(ConradModel):
    gate_radius_m: float = Field(default=4.0, gt=0, description="association gate around a registry asset")
    gate_sigma_multiplier: float = Field(default=3.0, gt=0)
    ambiguity_margin_m: float = Field(default=0.5, ge=0)
    cover_prior_mean: float = Field(default=0.3, ge=0, le=1)
    cover_prior_sd: float = Field(default=0.3, gt=0)
    cover_process_sd_per_sqrt_day: float = Field(default=0.05, ge=0)
    cover_meas_sd: float = Field(default=0.05, gt=0, description="clear-water survey cover noise")
    uncoupled_meas_sd: float = Field(default=0.12, gt=0, description="turbidity-blind survey noise")
    presence_prior: float = Field(default=0.5, gt=0, lt=1)
    presence_hit_rate: float = Field(default=0.9, gt=0, le=1)
    presence_false_alarm: float = Field(default=0.02, gt=0, lt=1)
    default_radius_m: float = Field(default=1.5, gt=0, description="footprint when the registry has none")
    presence_decay_time_s: float = Field(default=21600.0, gt=0)


class CouplingConfig(ConradModel):
    """Analytic CEFD constants (ENGINEERING_ESTIMATE)."""

    beam_attenuation_clear_per_m: float = Field(default=0.05, ge=0)
    beam_attenuation_per_m_per_ntu: float = Field(default=0.04, ge=0)
    min_visibility: float = Field(default=0.05, gt=0, le=1)
    stress_threshold_c: float = Field(default=26.0, description="used when the registry gives none")
    stress_process_inflation: float = Field(default=1.5, ge=0, description="cover process-noise gain")
    filtration_m3_s_per_m2: float = Field(default=3e-6, ge=0, description="entity->turbidity sink")
    filtration_mixing_volume_m3: float = Field(default=2000.0, gt=0)
    gate_prior: float = Field(default=0.5, ge=0, le=1, description="coupling gate before evidence")
    confident_probability: float = Field(default=0.9, gt=0.5, le=1)


class CefdSwitches(ConradModel):
    """Baselines differ only through these switches (one interface)."""

    entities: bool = True
    fields: bool = True
    field_to_entity: bool = True
    entity_to_field: bool = True
    field_dynamics: bool = True
    spatial_correlation: bool = True


class Model2EConfig(ConradModel):
    grid: BeliefGridConfig = BeliefGridConfig()
    fields: dict[str, FieldSpec] = Field(default_factory=_default_fields)
    entity: EntityConfig = EntityConfig()
    coupling: CouplingConfig = CouplingConfig()
    switches: CefdSwitches = CefdSwitches()
    variance_floor_fraction: float = Field(default=1e-4, gt=0)
    material_change_fraction: float = Field(default=0.01, ge=0, description="publish threshold")
    model_version: str = "model2e-cefd-analytic-0.1.0"
    clock_domain: str = "SIM"


def load_model2e_config(path: str | Path) -> Model2EConfig:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Model2EConfig.model_validate(data.get("model2e", data))
