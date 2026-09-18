"""Twin2S configuration. Every number is a SIMULATION DEFAULT (ch33), never a physical-world claim.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class OctreeConfig:
    base_voxel_m: float = 0.25
    refinement_voxels_m: tuple[float, ...] = (0.125, 0.0625)
    max_leaf_cells: int = 2_000_000


@dataclass(frozen=True)
class RaycastConfig:
    max_steps: int = 96
    hit_epsilon_m: float = 0.004
    min_step_m: float = 0.002
    start_offset_m: float = 0.02
    refine_iterations: int = 6


@dataclass(frozen=True)
class ObservedCriteria:
    """Sensor-specific conditions under which a visible surface point counts as OBSERVED (ch14)."""

    min_quality: float = 0.35
    min_incidence_cos: float = 0.17
    diversity_angle_rad: float = 0.35
    visibility_tolerance_m: float = 0.05


@dataclass(frozen=True)
class Twin2SConfig:
    octree: OctreeConfig = field(default_factory=OctreeConfig)
    raycast: RaycastConfig = field(default_factory=RaycastConfig)
    observed: ObservedCriteria = field(default_factory=ObservedCriteria)
    water_attenuation_per_m: float = 0.12
    clock_domain: str = "SIM"
    source_kind: str = "SYNTHETIC_ONLY"


def _build(cls: type[T], data: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown {cls.__name__} config key {key!r}")
        factory = known[key].default_factory
        default = None if factory is MISSING else factory()
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build(type(default), value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_twin2s_config(path: str | Path | None = None) -> Twin2SConfig:
    """Load ``configs/sim/twin2s_default.yaml``-style file (key ``twin2s``); ``None`` -> defaults."""
    if path is None:
        return Twin2SConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _build(Twin2SConfig, raw.get("twin2s", {}))
