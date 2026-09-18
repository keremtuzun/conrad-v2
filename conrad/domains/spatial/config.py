"""UAHSM configuration. Every number here is an ENGINEERING_ESTIMATE default (SYNTHETIC_ONLY), not a
calibrated physical constant; experiments override them from ``configs/eval/2s_*.yaml``.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from conrad.schemas.base import ConradModel


class GridConfig(ConradModel):
    base_voxel_m: float = Field(default=0.25, gt=0)
    refinement_voxels_m: tuple[float, ...] = (0.125, 0.0625)
    global_block_cells: int = Field(default=8, ge=1, description="global level block = N base cells a side")
    frame_id: str = "WORLD"

    @model_validator(mode="after")
    def _halving(self) -> GridConfig:
        prev = self.base_voxel_m
        for r in self.refinement_voxels_m:
            if abs(prev / r - 2.0) > 1e-9:
                raise ValueError("each refinement level must halve the previous voxel size")
            prev = r
        return self


class OccupancyConfig(ConradModel):
    l_hit: float = Field(default=1.2, gt=0)
    l_miss: float = Field(default=-0.7, lt=0)
    l_min: float = -4.0
    l_max: float = 4.0
    per_evidence_mass_cap: bool = Field(
        default=True, description="one observation contributes at most unit mass to a cell (no ray counting)"
    )
    p_occupied: float = Field(default=0.65, gt=0.5, lt=1)
    p_free: float = Field(default=0.35, gt=0, lt=0.5)
    observed_min_weight: float = Field(default=1.0, ge=0, description="direct mass needed for OBSERVED")
    coverage_weight_scale: float = Field(
        default=0.6, gt=0, description="coverage = 1 - exp(-w / scale); one full-mass view -> 0.81"
    )
    knowledge_status: bool = Field(default=True, description="False: every touched cell reports OBSERVED")
    use_evidence_reliability: bool = True
    degraded_health_weight: float = Field(default=0.5, ge=0, le=1)


class PoseConfig(ConradModel):
    use_pose_covariance: bool = True
    unknown_pose_sigma_m: float = Field(default=0.5, gt=0, description="used when covariance is None")
    unknown_pose_sigma_rad: float = Field(default=0.1, gt=0)
    splat_sigma_k: float = Field(default=2.0, gt=0)
    max_splat_radius_cells: int = Field(default=2, ge=0)
    confidence_cap: bool = Field(
        default=True, description="occupied-side logodds <= l_max / (1 + (sigma_pose/res)^2)"
    )


class SensorModelConfig(ConradModel):
    ray_step_fraction: float = Field(default=0.5, gt=0, le=1)
    free_space_on_no_return: bool = Field(
        default=False, description="NaN is ambiguous (dropout vs no return): carve nothing by default"
    )
    free_margin_sigma_k: float = Field(default=2.0, ge=0)
    range_noise_growth_per_m: float = Field(default=0.05, ge=0)
    default_range_noise_sigma_m: float = Field(default=0.05, gt=0)
    dedupe_point_cloud_with_depth: bool = True
    sonar_detect_threshold: float = Field(default=0.05, ge=0)
    sonar_cfar_k: float = Field(default=4.0, ge=0)
    sonar_mass_scale: float = Field(default=0.6, gt=0, le=1)
    sonar_free_mass_scale: float = Field(default=0.4, ge=0, le=1)
    sonar_free_space: bool = True
    sonar_elevation_min_samples: int = Field(default=3, ge=1)


class RelationalConfig(ConradModel):
    enabled: bool = True
    max_gap_cells: int = Field(default=2, ge=1)
    min_support_probability: float = Field(default=0.7, gt=0.5, le=1)
    inferred_occupancy: float = Field(default=0.7, gt=0.5, lt=1)
    inferred_observational_uncertainty: float = Field(default=0.7, ge=0, le=1)
    max_free_mass: float = Field(default=0.3, ge=0, description="cells with more free mass are never filled")
    max_support_refs: int = Field(default=8, ge=1)


class TemporalConfig(ConradModel):
    change_alpha: float = Field(default=0.5, gt=0, le=1)
    contradiction_min_confidence: float = Field(default=0.4, ge=0, le=1)
    contradiction_min_mass: float = Field(default=0.5, ge=0)
    dynamic_change_threshold: float = Field(default=0.35, ge=0, le=1)
    dynamic_tau_s: float = Field(default=5.0, gt=0)
    dynamic_uncertainty_rate_per_s: float = Field(default=0.1, ge=0)
    stale_after_s: float = Field(default=60.0, gt=0)


class RefinementConfig(ConradModel):
    enabled: bool = True
    min_coverage: float = Field(default=0.3, ge=0, le=1)
    uncertainty_gradient_threshold: float = Field(default=0.35, ge=0)
    discontinuity_threshold: float = Field(default=0.3, ge=0)
    local_radius_m: float = Field(default=4.0, gt=0)
    max_refined_cells_per_level: int = Field(default=4000, ge=0)
    fine_free_tail_m: float = Field(default=1.0, ge=0)


class MessageConfig(ConradModel):
    max_evidence_refs: int = Field(default=64, ge=1)
    max_cell_provenance: int = Field(default=16, ge=1)
    entity_type: str = "spatial_region"


class QueryConfig(ConradModel):
    unknown_block_probability: float = Field(default=0.5, ge=0, le=1)
    max_visibility_targets: int = Field(default=512, ge=1)


class SpatialConfig(ConradModel):
    model_version: str = "uahsm-v1-analytic"
    grid: GridConfig = GridConfig()
    occupancy: OccupancyConfig = OccupancyConfig()
    pose: PoseConfig = PoseConfig()
    sensor: SensorModelConfig = SensorModelConfig()
    relational: RelationalConfig = RelationalConfig()
    temporal: TemporalConfig = TemporalConfig()
    refinement: RefinementConfig = RefinementConfig()
    messages: MessageConfig = MessageConfig()
    query: QueryConfig = QueryConfig()


def _merge(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        out[k] = _merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), dict) else v
    return out


def spatial_config(
    overrides: Mapping[str, Any] | None = None, base: SpatialConfig | None = None
) -> SpatialConfig:
    """Deep-merge ``overrides`` into ``base`` (default :class:`SpatialConfig`) and validate."""
    root = (base or SpatialConfig()).model_dump()
    return SpatialConfig.model_validate(_merge(root, overrides or {}))


def load_spatial_config(path: str | Path, key: str = "model") -> SpatialConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    return spatial_config(data.get(key, {}))


def plain_grid_config(base: SpatialConfig | None = None) -> SpatialConfig:
    """S-B0 classical occupancy grid: per-ray counting, OctoMap-style clamps, no status, no pose covariance."""
    return spatial_config(
        {
            "model_version": "baseline-plain-occupancy-v1",
            "occupancy": {
                "l_hit": 0.85,
                "l_miss": -0.4,
                "l_min": -2.0,
                "l_max": 3.5,
                "per_evidence_mass_cap": False,
                "observed_min_weight": 0.0,
                "knowledge_status": False,
                "use_evidence_reliability": False,
            },
            "pose": {"use_pose_covariance": False, "confidence_cap": False},
            "relational": {"enabled": False},
            "refinement": {"enabled": False},
        },
        base,
    )


def ignore_pose_config(base: SpatialConfig | None = None) -> SpatialConfig:
    """Ablation: full UAHSM but pose covariance is ignored (every return lands at a precise cell)."""
    return spatial_config(
        {
            "model_version": "uahsm-v1-no-pose-cov",
            "pose": {"use_pose_covariance": False, "confidence_cap": False},
        },
        base,
    )
