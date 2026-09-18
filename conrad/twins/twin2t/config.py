"""Twin2T configuration (YAML: ``configs/sim/twin2t_*.yaml``). Unknown keys fail.

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from conrad.twins.twin2t.mechanisms import CorrosionModel


@dataclass(frozen=True)
class MCDEConfig:
    """Ablation switches (ch11 MCDE ablations) and coupling constants."""

    mechanism_coupling: bool = True
    topology_coupling: bool = True
    stochastic: bool = True
    events_enabled: bool = True
    interventions_enabled: bool = True
    environment_conditioning: bool = True
    loading_conditioning: bool = True
    corrosion_model: CorrosionModel = CorrosionModel.POWER_LAW
    max_stress_concentration: float = 5.0
    """Cap on net-section stress amplification t0/(t0-d) (ENGINEERING_ESTIMATE)."""
    support_load_redistribution: float = 0.5
    """dsigma multiplier slope per unit support wall-loss fraction (ENGINEERING_ESTIMATE)."""
    crack_coating_length_scale_m: float = 0.05
    """Coating over a crack is considered breached over crack_length/scale of the region (ENGINEERING_ESTIMATE)."""
    max_crack_length_m: float = 1.0
    reference_stress_range_pa: float = 50.0e6
    """Used only when loading_conditioning is ablated (ENGINEERING_ESTIMATE)."""
    reference_cycles_per_s: float = 0.05


@dataclass(frozen=True)
class ObservationConfig:
    visibility_threshold: float = 0.3
    default_fidelity: str = "T0"
    feature_dim: int = 16
    wall_loss_sigma_m: float = 2.0e-4
    anomaly_sigma: float = 0.05
    crack_sigma_m: float = 1.0e-3
    crack_detection_limit_m: float = 2.0e-3
    turbidity_noise_gain: float = 2.0
    biofouling_noise_gain: float = 1.5
    corruption_noise_gain: float = 4.0
    contradiction_magnitude: float = 0.6


@dataclass(frozen=True)
class Twin2TConfig:
    mcde: MCDEConfig = field(default_factory=MCDEConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    prior_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    generator: str = "CONFIGURED"
    """GeneratorKind name (baselines.py); CONFIGURED = MCDE with the given switches."""
    clock_domain: str = "sim"
    generator_version: str = "twin2t-mcde-0.1.0"


def _build(cls: type[Any], data: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def config_from_dict(data: dict[str, Any]) -> Twin2TConfig:
    data = dict(data)
    unknown = set(data) - {f.name for f in fields(Twin2TConfig)}
    if unknown:
        raise ValueError(f"unknown twin2t keys: {sorted(unknown)}")
    mcde_raw = dict(data.pop("mcde", {}) or {})
    if "corrosion_model" in mcde_raw:
        mcde_raw["corrosion_model"] = CorrosionModel(mcde_raw["corrosion_model"])
    return Twin2TConfig(
        mcde=_build(MCDEConfig, mcde_raw),
        observation=_build(ObservationConfig, dict(data.pop("observation", {}) or {})),
        **data,
    )


def load_config(path: str | Path) -> Twin2TConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "twin2t" not in raw:
        raise ValueError(f"{path}: expected top-level 'twin2t' section")
    return config_from_dict(raw["twin2t"])
