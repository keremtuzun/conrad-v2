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
    """Nominal 1-sigma measurement noise per quantity at reliability 1 (own guess, not Twin2T's).

    Used for the surface anomaly score always, and for wall loss / crack length only under the legacy
    ``ABSOLUTE_GAUSSIAN`` measurement model."""
    measurement_model: str = "SENSOR_CHARACTERISED"
    """``SENSOR_CHARACTERISED`` (default, docs/audits/MODEL2T_REPAIR.md): wall loss and crack length use the
    declared :class:`SensorCharacteristics` (relative error, persistent bias, POD, partial view).
    ``ABSOLUTE_GAUSSIAN``: the pre-repair fixed absolute-sigma Kalman update (kept as an ablation)."""
    min_reliability: float = 0.05
    conflict_sigma: float = 3.5
    """Normalised innovation above which reliable evidence is a contradiction."""
    conflict_min_reliability: float = 0.6
    uc_gain: float = 0.35
    uc_resolve_factor: float = 0.7
    ua_smoothing: float = 0.5

    def __post_init__(self) -> None:
        if self.measurement_model not in ("SENSOR_CHARACTERISED", "ABSOLUTE_GAUSSIAN"):
            raise ValueError(f"unknown direct.measurement_model {self.measurement_model!r}")


@dataclass(frozen=True)
class SensorCharacteristics:
    """Datasheet-style characterisation of the STRUCTURED inspection sensor type, as Model2T ASSUMES it.

    ENGINEERING_ESTIMATE (uncalibrated): a generic close-range visual / structured-light inspection payload.
    These are Model2T's own declared numbers. They are NOT read from Twin2T or its config; where they match
    or differ from the simulated sensor is documented in docs/audits/MODEL2T_REPAIR.md.

    Model (per reading, per quantity; ``f`` = unknown fraction of the defect that is in view):
      * crack: the reading is a *call* only if it is at least ``crack_call_threshold_m``; below that it is a
        non-detection (censored, weak negative evidence). P(detect) is log-logistic in the in-view length
        ``f * L`` with a50 ``crack_pod_a50_m`` (raised by reported degradation via ``pod_ua_gain``).
        A detection is log-normal around ``crack_sizing_median_factor * f * L`` with relative scatter
        ``crack_rel_sigma`` plus a persistent per-sensor bias ``crack_systematic_rel_sigma``, plus an absolute
        floor ``crack_abs_sigma_m``. An undetected crack still produces an indication from the noise floor
        (half-normal with sd ``crack_abs_sigma_m``), which can exceed the call threshold (a false call).
      * wall loss: always measured; log-normal around ``wall_sizing_median_factor * L`` (partial view:
        ``f * L``) with ``wall_rel_sigma`` + persistent ``wall_systematic_rel_sigma`` + ``wall_abs_sigma_m``.
      * partial view: the sensor never reports how much of a component it saw. Every reading is therefore a
        mixture: with probability ``full_view_prob`` (scaled by reading reliability) the whole defect is in
        view (f = 1), otherwise f ~ Uniform(``*_partial_view_min_fraction``, 1): a low reading is a lower
        bound, not a contradiction.
      * correlation: readings from one sensor on one component share the persistent bias, so n of them carry
        no more certainty than the bias allows: the level variance is floored at (bias * level)^2 / #sensors.
    """

    crack_call_threshold_m: float = 2.5e-3
    crack_pod_a50_m: float = 8.0e-3
    crack_pod_log_width: float = 0.6
    pod_ua_gain: float = 2.0
    """a50 multiplier per unit of evidence aleatoric level (turbid / fouled views detect less)."""
    crack_sizing_median_factor: float = 0.9
    crack_rel_sigma: float = 0.30
    crack_systematic_rel_sigma: float = 0.25
    crack_abs_sigma_m: float = 1.0e-3
    noise_ua_gain: float = 3.0
    """Every scatter / noise-floor sigma is multiplied by (1 + noise_ua_gain * evidence aleatoric level)."""
    wall_sizing_median_factor: float = 1.0
    wall_rel_sigma: float = 0.12
    wall_systematic_rel_sigma: float = 0.06
    wall_abs_sigma_m: float = 3.0e-4
    full_view_prob: float = 0.8
    crack_partial_view_min_fraction: float = 0.2
    """Smallest in-view fraction of a crack's length in a partial view."""
    wall_partial_view_min_fraction: float = 0.5
    """Smallest ratio of the viewed region's wall loss to the component worst case in a partial view."""
    partial_view_nodes: int = 9
    """Quadrature nodes over the partial-view fraction."""
    defect_locality_m: float = 0.3
    """An indication is a local feature. A reading whose measured surface point (Evidence.spatial_support, from
    association) lies further than this plus 2 sigma from where the component's worst indication was seen
    views another region: it only bounds the worst case from below, and a miss there is uninformative.
    Readings without a measured surface point are treated as views of the whole component."""
    grid_points: int = 400
    """Grid resolution of the per-reading Bayesian update (moment-matched back to the Gaussian belief)."""
    plausible_likelihood_ratio: float = 0.01
    """A reading's plausible set: states whose likelihood is at least this fraction of the maximum."""
    contradiction_prior_mass: float = 1.0e-3
    """Credible disagreement: the belief puts less than this probability on the reading's plausible set."""


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
    relative_level_noise: dict[str, float] = field(
        default_factory=lambda: {"corrosion_depth_m": 0.0, "crack_length_m": 0.0}
    )
    """Level random-walk sd per sqrt(year) PROPORTIONAL to the current level (Paris-type growth: a long crack
    can grow by a large amount). Default 0 (DEV runs with 1.0 and 2.0 for cracks made small, near-static
    cracks drift upward more than they helped run-away cracks; docs/audits/MODEL2T_REPAIR.md)."""
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
    tail_weight: dict[str, float] = field(
        default_factory=lambda: {"corrosion_depth_m": 0.1, "crack_length_m": 0.05}
    )
    """Heavy tail of the population prior (SENSOR_CHARACTERISED model only): the probability that a component
    carries a real defect outside the Gaussian core. Own ENGINEERING_ESTIMATE."""
    tail_scale_m: dict[str, float] = field(
        default_factory=lambda: {"corrosion_depth_m": 3.0e-3, "crack_length_m": 1.0e-2}
    )
    """Exponential scale (m) of that tail."""
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
    sensor: SensorCharacteristics = field(default_factory=SensorCharacteristics)
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
        "sensor": SensorCharacteristics,
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
