"""Model2T configuration. Every number is the BELIEF model's own ENGINEERING_ESTIMATE guess.

None of these values is read from Twin2T; they are deliberately independent (ch10 "Important truth/model
separation": the model gets imperfect prior models and must learn from observations).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

YEAR_S = 365.25 * 86400.0


class PropagationMode(str, Enum):
    NONE = "NONE"
    """Independent-component model: no relational inference at all."""
    TCDP = "TCDP"
    """Mechanism-aware, reliability-gated, attenuated propagation (candidate)."""
    GENERIC = "GENERIC"
    """Mechanism-agnostic propagation over every edge (contamination-prone baseline B9)."""


@dataclass(frozen=True)
class DirectConfig:
    sigma_m: dict[str, float] = field(
        default_factory=lambda: {
            "corrosion_depth_m": 3.0e-4,
            "crack_length_m": 1.5e-3,
            "surface_anomaly": 0.08,
        }
    )
    """Nominal 1-sigma measurement noise per quantity at reliability 1 (own guess, not Twin2T's)."""
    min_reliability: float = 0.05
    conflict_sigma: float = 3.5
    """Normalised innovation above which reliable evidence is a contradiction."""
    conflict_min_reliability: float = 0.6
    uc_gain: float = 0.35
    uc_resolve_factor: float = 0.7
    ua_smoothing: float = 0.5


@dataclass(frozen=True)
class DynamicsConfig:
    """Engineering prior rates. Corrosion ~0.1 mm/yr is a generic immersed-steel rule of thumb;
    crack growth is a lumped Paris-style exponential rate. Both are uncalibrated guesses."""

    corrosion_rate_m_per_yr: float = 1.0e-4
    corrosion_rate_sd_m_per_yr: float = 1.0e-4
    crack_rate_m_per_yr: float = 5.0e-4
    crack_rate_sd_m_per_yr: float = 1.0e-3
    level_process_noise: dict[str, float] = field(
        default_factory=lambda: {
            "corrosion_depth_m": 2.0e-5,
            "crack_length_m": 2.0e-4,
            "surface_anomaly": 0.05,
        }
    )
    """Level random-walk sd per sqrt(year)."""
    rate_process_noise: dict[str, float] = field(
        default_factory=lambda: {
            "corrosion_depth_m": 2.0e-5,
            "crack_length_m": 2.0e-4,
            "surface_anomaly": 0.0,
        }
    )
    support_timescale_s: float = 180 * 86400.0
    """Direct support decays with this e-folding time while a component is not re-observed."""
    epistemic_growth_per_yr: float = 0.1


@dataclass(frozen=True)
class PriorConfig:
    """Population prior for a never-observed component (reported as UNKNOWN; used only internally)."""

    mean: dict[str, float] = field(
        default_factory=lambda: {
            "corrosion_depth_m": 5.0e-4,
            "crack_length_m": 1.5e-3,
            "surface_anomaly": 0.2,
        }
    )
    sd: dict[str, float] = field(
        default_factory=lambda: {
            "corrosion_depth_m": 5.0e-4,
            "crack_length_m": 1.5e-3,
            "surface_anomaly": 0.2,
        }
    )
    base_epistemic: float = 0.2
    unknown_material_epistemic: float = 0.4


@dataclass(frozen=True)
class TCDPConfig:
    mode: PropagationMode = PropagationMode.TCDP
    mechanism_relations: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "CORROSION": ("CONNECTED_TO", "ATTACHED_TO", "CONTACTS", "EXPOSED_TO"),
            "FATIGUE": ("SUPPORTED_BY", "LOAD_TRANSFER"),
        }
    )
    """Only these relation types may carry a message for the mechanism (shared environment / load path)."""
    correlation: dict[str, float] = field(
        default_factory=lambda: {
            "CONNECTED_TO": 0.4,
            "ATTACHED_TO": 0.5,
            "CONTACTS": 0.3,
            "EXPOSED_TO": 0.3,
            "SUPPORTED_BY": 0.3,
            "LOAD_TRANSFER": 0.3,
        }
    )
    """Assumed state correlation rho across an edge: the message is the Gaussian conditional (attenuation)."""
    min_source_support: float = 0.3
    """Reliability gate: a source with less direct support than this sends nothing."""
    min_gate: float = 0.05
    max_shift_sd: float = 3.0
    """Bound on a message's mean shift in target-prior sd (0 disables; GENERIC always unbounded)."""
    generic_correlation: float = 0.6
    generic_iterations: int = 2


@dataclass(frozen=True)
class ContextConfig:
    biofouling_keys: tuple[str, ...] = ("biofouling_cover", "biofouling_fraction", "biofouling")
    turbidity_keys: tuple[str, ...] = ("turbidity",)
    visibility_keys: tuple[str, ...] = ("visibility",)
    uo_gain: float = 0.6
    ua_gain: float = 0.4
    surface_variance_gain: float = 4.0
    """Biofouling hides surface appearance: surface-anomaly measurement variance is inflated."""


@dataclass(frozen=True)
class ConditionConfig:
    nominal_wall_m: float = 0.015
    crack_critical_m: float = 0.05
    bands: tuple[float, float, float] = (0.05, 0.3, 0.8)
    """severity -> DEGRADED / SEVERE / FAILED thresholds (own bands)."""
    change_sigma: float = 2.0


@dataclass(frozen=True)
class LearnedTCDPConfig:
    """ch33 defaults: z 256, r 64, mechanism 128, message 512->256->256, gate 512->128->1, 3 layers."""

    d_node_in: int = 16
    d_z: int = 256
    d_r: int = 64
    d_mech: int = 128
    d_hidden: int = 512
    d_gate: int = 128
    d_head: int = 128
    n_layers: int = 3
    n_mechanisms: int = 6
    partitions: tuple[int, ...] = (64, 64, 48, 48, 32)
    """z_T = [material, degradation, load, geometry, history] (ch33)."""
    mutable_partitions: tuple[int, ...] = (1, 2, 4)
    """TCDP may rewrite degradation, load and history only; material and geometry are frozen."""
    dropout: float = 0.10


@dataclass(frozen=True)
class Model2TConfig:
    direct: DirectConfig = field(default_factory=DirectConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    tcdp: TCDPConfig = field(default_factory=TCDPConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    condition: ConditionConfig = field(default_factory=ConditionConfig)
    learned_tcdp: LearnedTCDPConfig = field(default_factory=LearnedTCDPConfig)
    clock_domain: str = "sim"
    model_version: str = "model2t-analytic-0.1.0"


def _build(cls: type[Any], data: dict[str, Any]) -> Any:
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def model2t_config_from_dict(data: dict[str, Any]) -> Model2TConfig:
    data = dict(data)
    sub = {
        "direct": DirectConfig,
        "dynamics": DynamicsConfig,
        "prior": PriorConfig,
        "tcdp": TCDPConfig,
        "context": ContextConfig,
        "condition": ConditionConfig,
        "learned_tcdp": LearnedTCDPConfig,
    }
    unknown = set(data) - {f.name for f in fields(Model2TConfig)}
    if unknown:
        raise ValueError(f"unknown model2t keys: {sorted(unknown)}")
    built: dict[str, Any] = {}
    for key, cls in sub.items():
        raw = dict(data.pop(key, {}) or {})
        if key == "tcdp" and "mode" in raw:
            raw["mode"] = PropagationMode(raw["mode"])
        for k, v in list(raw.items()):
            if isinstance(v, list):
                raw[k] = tuple(v)
        built[key] = _build(cls, raw)
    return Model2TConfig(**built, **data)
