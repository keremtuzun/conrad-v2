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


class MessageModel(str, Enum):
    """How a TCDP message turns a neighbour's posterior into a prior for the receiver."""

    GAUSSIAN_CONDITIONAL = "GAUSSIAN_CONDITIONAL"
    """Iteration-3 arm: linear Gaussian conditional under an ASSUMED per-relation correlation."""
    MEASURED_EXPOSURE = "MEASURED_EXPOSURE"
    """Iteration-4 candidate: shared-exposure mixture whose edge correlation is MEASURED on the pairs
    whose two ends both carry direct evidence (empirical Bayes, shrunk toward the assumed correlation)."""


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
    crack_false_call_weight: float = 0.05
    """Fraction of the crack noise floor that is NOT Gaussian (iteration 4). A structured-light / visual
    payload reports crack-like indications from weld toes, scratches, marine growth and registration error,
    and those are much heavier tailed than the sizing noise. Without this tail a single indication a few
    sigma above the floor is read as a real crack, which is what made a PRISTINE surface report DEGRADED
    (docs/audits/MODEL2T_REPAIR.md iteration 4). ENGINEERING_ESTIMATE, selected on DEVELOPMENT worlds."""
    crack_false_call_scale_m: float = 3.0e-3
    """Exponential scale of that false-indication tail (scaled by the sensor-health noise factor)."""
    noise_ua_gain: float = 5.0
    """Every scatter / noise-floor sigma is multiplied by 1 + noise_ua_gain * max(0, aleatoric -
    noise_ua_reference) (evidence aleatoric level from reported sensor health)."""
    noise_ua_reference: float = 0.1
    """Aleatoric level of a healthy sensor, for which the declared sigmas hold (iteration 2 used 0 with gain 3;
    DEV iteration 3: healthy readings were over-covered, 0.997, docs/audits/MODEL2T_REPAIR.md)."""
    wall_sizing_median_factor: float = 1.0
    wall_rel_sigma: float = 0.12
    wall_systematic_rel_sigma: float = 0.06
    wall_abs_sigma_m: float = 2.5e-4
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
class CrackGrowthConfig:
    """Crack temporal model (SENSOR_CHARACTERISED measurement model only). Own ENGINEERING_ESTIMATE.

    ``REGIME_MIXTURE`` (default, docs/audits/MODEL2T_REPAIR.md iteration 3): the crack belief is a discrete
    joint over log length and a growth regime. Regime 0 is STABLE (below the growth threshold: no growth);
    the other regimes are RUN-AWAY exponential growth d ln L / dt = g with g on a log-spaced grid. Regimes
    switch with hazards (initiation, arrest, rate drift), and log length diffuses, so predictive uncertainty
    grows with PHYSICAL elapsed time and the predictive distribution is heavy-tailed (a minority of cracks run
    away). ``CONSTANT_RATE``: the iteration-2 Gaussian [level, rate] Kalman model (kept as an ablation)."""

    model: str = "REGIME_MIXTURE"
    min_length_m: float = 1.0e-4
    """Bottom of the log-length grid (a smaller crack is indistinguishable from none)."""
    max_length_m: float = 3.0
    """Top of the grid: a generic component-scale bound on crack length (growth past it is refuted)."""
    grid_points: int = 240
    stable_prob: float = 0.85
    """Prior probability that a never-seen crack is in the STABLE regime."""
    runaway_rates_per_yr: tuple[float, ...] = (0.2, 0.45, 1.0, 2.2, 5.0, 11.0)
    """Run-away regimes: exponential growth rates of the length (1/yr), equally likely a priori."""
    initiation_per_yr: float = 0.02
    """Hazard of a STABLE crack starting to grow (to any run-away regime)."""
    arrest_per_yr: float = 0.3
    """Hazard of a run-away crack arresting (e.g. load shedding, or reaching a geometric limit)."""
    rate_drift_per_yr: float = 0.5
    """Hazard of moving to a neighbouring run-away rate (the growth rate is itself uncertain over time)."""
    log_diffusion_per_sqrt_yr: float = 0.1
    """Random-walk sd of ln(length) per sqrt(year) in every regime."""
    surprise_mix: float = 0.02
    """On a reading the belief cannot explain (see ``contradiction_prior_mass``), this fraction of the belief is
    replaced by a log-uniform length before the update: a genuine change is followed, not smoothed away."""
    min_propagation_s: float = 3600.0
    """Elapsed physical time is accumulated and applied to the grid once it exceeds this (a mission tick of
    0.1 s does not need a grid convolution; the pending time is always applied before a reading)."""
    max_substep_s: float = 30 * 86400.0
    """A long elapsed time is applied in sub-steps of at most this (operator-splitting accuracy)."""
    region_prior_power: float = 0.5
    """Prior of ONE surface region, as a power of the component population prior CDF (iteration 4).

    The component prior states the worst crack on a WHOLE component. Giving every surface region that same
    prior and reporting the worst region inflates a pristine component towards the DEGRADED band. The exact
    correction is the n-th root of the CDF (``power = 1 / n_cells``), but on DEVELOPMENT worlds that prior is
    so strong that a genuine 10 mm crack is explained away as a false indication, so the power is a declared
    ENGINEERING_ESTIMATE selected on DEVELOPMENT worlds. 1.0 = the iteration-3 component prior per region."""
    point_estimate: str = "MEDIAN"
    """Reported crack length: posterior ``MEDIAN`` (default; the skewed run-away tail pulls the mean up) or
    ``MEAN``. The variance is always the posterior's."""

    def __post_init__(self) -> None:
        if self.model not in ("REGIME_MIXTURE", "CONSTANT_RATE"):
            raise ValueError(f"unknown crack_growth.model {self.model!r}")
        if self.point_estimate not in ("MEAN", "MEDIAN"):
            raise ValueError(f"unknown crack_growth.point_estimate {self.point_estimate!r}")
        if not 0.0 < self.region_prior_power <= 1.0:
            raise ValueError("crack_growth.region_prior_power must be in (0, 1]")


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
    mode: PropagationMode = PropagationMode.NONE
    """Production default NONE (ADR-0009): gate 2T-TCDP FAILED, so no condition is propagated to neighbours.
    TCDP / GENERIC are EXPERIMENTAL arms: pass ``mode=`` explicitly (experiments, tests) or, on the mission
    runtime, set ``experimental_enabled``."""
    experimental_enabled: bool = False
    """Explicit opt-in that lets the mission runtime honour a relational ``model2t_mode`` (ADR-0009)."""
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
    message_model: MessageModel = MessageModel.GAUSSIAN_CONDITIONAL
    """TCDP message form (GENERIC always uses the Gaussian conditional, so baseline B9 is unchanged).
    MEASURED_EXPOSURE was the iteration-4 redesign; it was measured on DEV, lost to GAUSSIAN_CONDITIONAL on
    both benefit and contamination, and is kept only as an ablation (docs/audits/MODEL2T_REPAIR.md)."""
    elevated_sd: float = 2.0
    """A component "carries a defect" when its level is this many core-prior sd above the core-prior mean."""
    elevated_prior: dict[str, float] = field(
        default_factory=lambda: {"corrosion_depth_m": 0.1, "crack_length_m": 0.05}
    )
    """Prior rate of defect-carrying components. Model2T's own ENGINEERING_ESTIMATE; it matches
    ``PriorConfig.tail_weight`` so that an uninformative message reproduces the population prior exactly."""
    base_rate_pseudo_n: float = 4.0
    """Pseudo-components shrinking the MEASURED defect rate toward ``elevated_prior``."""
    correlation_pseudo_pairs: float = 4.0
    """Pseudo-pairs shrinking the MEASURED edge correlation toward ``correlation`` (empirical Bayes)."""
    max_measured_correlation: float = 0.9
    severity_pseudo_weight: float = 1.0
    """Pseudo-weight shrinking the measured shared-exposure severity toward the population tail level."""


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
class CoverageConfig:
    """Surface coverage (``coverage.py``). Used only for components whose surveyed design geometry is in the
    mission context (``design_geometry``); otherwise every reading is treated as a whole-component view."""

    enabled: bool = True
    cell_m: float = 0.5
    """Axial length of one coverage cell of a capsule (pipe segment)."""
    sectors: int = 8
    """Circumferential sectors of a capsule."""
    complete_fraction: float = 0.8
    """Component-level condition is OBSERVED only once this fraction of the cells has been read (or once the
    observed part alone already puts the component in the worst condition band). Below it the condition is
    UNKNOWN, U_O >= 1 - seen fraction, and the observed part is reported as ``observed_region_condition``."""
    view_credit: str = "FOOTPRINT"
    """``FOOTPRINT`` (default, iteration 4): a reading credits every cell in the DECLARED footprint of the
    payload around its measured surface point. ``MEASURED_CELL``: the iteration-3 behaviour, one cell per
    reading (kept as an ablation)."""
    footprint_half_angle_deg: float = 50.0
    """Half angle of the declared reading footprint, measured between outward design normals. A cell whose
    normal is further than this from the measured point's normal is NOT credited, so a near-side look never
    credits the far side. ENGINEERING_ESTIMATE; the same value the frozen MCBR predictive model declares
    (``SurfacePredictiveConfig.footprint_half_angle_deg``, configs/active/mcbr_frozen_v2.yaml)."""
    footprint_axial_m: float = 1.0
    """Axial reach of the declared reading footprint (same source as the half angle)."""
    probe_offset_m: float = 0.4
    """Water-side stand-off of the probe point handed to the deployment's occlusion test. It must be larger
    than the Model2S voxel (0.25 m), or the surface's own voxel would report every cell as occluded."""
    reveal_pod_floor: float = 0.02
    """Floor of the declared reveal probability (below it a look tells almost nothing about the worst case)."""

    def __post_init__(self) -> None:
        if self.view_credit not in ("FOOTPRINT", "MEASURED_CELL"):
            raise ValueError(f"unknown coverage.view_credit {self.view_credit!r}")


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
    crack_growth: CrackGrowthConfig = field(default_factory=CrackGrowthConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    tcdp: TCDPConfig = field(default_factory=TCDPConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    condition: ConditionConfig = field(default_factory=ConditionConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    learned_tcdp: LearnedTCDPConfig = field(default_factory=LearnedTCDPConfig)
    clock_domain: str = "sim"
    model_version: str = "model2t-analytic-0.1.0"


def production_propagation_mode(requested: str | PropagationMode, config: Model2TConfig) -> PropagationMode:
    """Propagation mode of the PRODUCTION mission runtime (ADR-0009). Gate 2T-TCDP FAILED, so a relational
    mode is honoured only with the explicit opt-in ``tcdp.experimental_enabled``; otherwise NONE."""
    mode = PropagationMode(requested)
    return mode if config.tcdp.experimental_enabled else PropagationMode.NONE


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
        "crack_growth": CrackGrowthConfig,
        "prior": PriorConfig,
        "tcdp": TCDPConfig,
        "context": ContextConfig,
        "condition": ConditionConfig,
        "coverage": CoverageConfig,
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
