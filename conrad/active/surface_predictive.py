"""Belief-side surface-cell predictive model for MCBR in the integrated mission (gate I4 repair, 2026-09-19).

The quantities a structural inspection resolves are component WORST CASES over a surface. The belief side
knows (i) which surface cells of the SURVEYED design geometry its structural readings already covered
(Model2T coverage), (ii) where the worst indication was measured (Model2T locus), (iii) the current posterior
of each quantity and the population prior of the unread surface (Model2T), (iv) the declared sensor
datasheet (Model2T ``SensorCharacteristics``), and (v) which cells a candidate pose would see (Model2S
ray-cast occlusion, sensor range/FOV, incidence). Nothing here reads Twin truth.

For each quantity q two scalars are exposed through the ``PredictiveBelief`` protocol:

    ``q:unread``  worst case of the not-yet-read surface. Prior = Model2T population prior. A view observes it
                  only if the worst indication lies in a cell the view sees: an ERASURE channel with success
                  probability phi(v) = sum_c P(defect in c) * w_c(v) (w_c = predicted clear-ray probability x
                  incidence cosine) times the declared probability of detection.
    ``q:read``    the read-surface estimate (only once q was detected). A view observes it with quality w of
                  the locus cell; the persistent per-sensor bias floor caps what repeated looks can achieve.

The expected entropy reduction of an erasure channel is phi * 0.5 ln(var / var_post). It is handed to the
generic rankers as an INFORMATION-EQUIVALENT noise std (the Gaussian noise whose entropy reduction equals the
expected one), so ``predictive.expected_values(...)['entropy']`` is exactly this conventional EIG.

Every candidate always reports every scalar (a view that sees nothing gets an effectively infinite std), so
``predicted_coverage`` is 1 and the shared feasibility filter is identical for every planner in a comparison:
the model changes the RANKING of planners that read it, never the feasible set.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from pydantic import Field

from conrad.active.candidates import SensorOption
from conrad.active.predictive import HypothesisBelief, PredictedOutcome, ScalarBelief
from conrad.schemas.base import ConradModel
from conrad.schemas.frames import Pose, SpatialSupport

# (pose, sensor) -> per-cell observation weight in [0, 1] (clear-ray probability x incidence), shape (n_cells,)
CellWeights = Callable[[Pose, SensorOption], np.ndarray]
NO_INFORMATION_STD_FACTOR = 1.0e6


class SurfacePredictiveConfig(ConradModel):
    """Parameters of the mission's belief-side predictive model. ENGINEERING_ESTIMATE. Defaults = the DEVELOPMENT
    choice frozen in configs/active/mcbr_frozen_v2.yaml (docs/audits/MCBR_REEVALUATION.md, I4 repair)."""

    enabled: bool = True
    footprint_discount: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="P(defect in cell) multiplier for cells inside the declared footprint of a reading that did "
        "not close the need (Model2T credits only the measured cell; the payload looked at more)",
    )
    track_discount: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="P(defect in cell) multiplier 1 - d * w for the best observation weight w the cell had from any "
        "of the robot's own ESTIMATED past positions (sensor range, incidence, Model2S clear ray)",
    )
    track_max_positions: int = Field(
        default=16, ge=1, description="past positions evaluated (thinned evenly)"
    )
    footprint_half_angle_deg: float = Field(default=50.0, ge=0, le=180)
    footprint_axial_m: float = Field(default=1.0, ge=0)
    seen_surface_discount: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="P(defect in cell) multiplier for a cell whose water-side probe Model2S already OBSERVED "
        "(the side-looking payload most likely looked at it) but that no structural reading covered",
    )
    probe_offset_m: float = Field(
        default=0.15, gt=0, description="water-side probe distance from the surface"
    )
    nominal_reliability: float = Field(
        default=1.0, ge=0, le=1, description="reliability of a planned reading"
    )
    incidence_power: float = Field(
        default=1.0, ge=0, description="cell weight = clear-ray probability x cos(incidence)^power"
    )
    read_min_band_probability: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="the read-surface scalar is offered only when the belief gives at least this probability that "
        "the read estimate lies in the component's worst condition band (only then can refining it decide the "
        "component condition; while the condition is open the component-level claims stay UNKNOWN)",
    )
    read_channel: bool = Field(
        default=False,
        description="include the read-surface (locus refinement) scalars (off: DEVELOPMENT choice)",
    )


@dataclass(frozen=True)
class QuantityChannel:
    """Belief-side channel of one quantity (all numbers from Model2T belief + declared datasheet)."""

    quantity: str
    unread_mean: float
    unread_var: float
    look_std_unread: float
    """std of ONE reading of a worst case at the unread prior level (declared relative + absolute noise,
    partial-view scatter and persistent bias)."""
    detection_probability: float = 1.0
    """declared probability that a reading of a worst case at the unread prior level is a detection"""
    read_mean: float | None = None
    read_var: float | None = None
    look_std_read: float | None = None
    read_var_floor: float = 0.0
    """persistent-bias floor of the read estimate (repeated same-sensor looks cannot go below it)"""
    locus_cell: int | None = None


def information_equivalent_std(var: float, info_nats: float) -> float:
    """Std of a Gaussian measurement whose entropy reduction of N(., var) is ``info_nats``."""
    if info_nats <= 1e-12 or var <= 0.0:
        return math.sqrt(max(var, 1e-24)) * NO_INFORMATION_STD_FACTOR
    post = var * math.exp(-2.0 * info_nats)
    prec = 1.0 / post - 1.0 / var
    return 1.0 / math.sqrt(max(prec, 1e-300))


def gaussian_info(var: float, look_std: float) -> float:
    return 0.5 * math.log1p(max(var, 0.0) / max(look_std, 1e-12) ** 2)


@dataclass
class SurfaceCellPredictive:
    """``PredictiveBelief`` over one target component's surface cells."""

    cell_prior: np.ndarray
    """P(the unread worst case lies in cell c); zero on cells already covered by readings; sums to 1"""
    channels: tuple[QuantityChannel, ...]
    cell_weights: CellWeights
    candidate_regions: tuple[SpatialSupport, ...] = ()
    epistemic: float = 0.0
    used_modalities: frozenset[str] = frozenset()
    hypotheses: tuple[HypothesisBelief, ...] = ()
    scalars: tuple[ScalarBelief, ...] = field(init=False)
    _cache: dict[tuple[tuple[float, ...], str], PredictedOutcome] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        out: list[ScalarBelief] = []
        for ch in self.channels:
            out.append(ScalarBelief(key=f"{ch.quantity}:unread", mean=ch.unread_mean, var=ch.unread_var))
            if ch.read_var is not None and ch.read_mean is not None:
                out.append(ScalarBelief(key=f"{ch.quantity}:read", mean=ch.read_mean, var=ch.read_var))
        self.scalars = tuple(out)

    def phi(self, weights: np.ndarray) -> float:
        return float(np.clip(np.sum(self.cell_prior * weights), 0.0, 1.0))

    def expected_information(self, pose: Pose, sensor: SensorOption) -> Mapping[str, float]:
        w = np.clip(np.asarray(self.cell_weights(pose, sensor), dtype=np.float64), 0.0, 1.0)
        phi = self.phi(w)
        info: dict[str, float] = {}
        for ch in self.channels:
            info[f"{ch.quantity}:unread"] = (
                phi * ch.detection_probability * gaussian_info(ch.unread_var, ch.look_std_unread)
            )
            if ch.read_var is not None and ch.look_std_read is not None:
                wl = float(w[ch.locus_cell]) if ch.locus_cell is not None else 0.0
                full = gaussian_info(ch.read_var, ch.look_std_read)
                cap = (
                    0.5 * math.log(ch.read_var / ch.read_var_floor)
                    if ch.read_var_floor > 0.0 and ch.read_var > ch.read_var_floor
                    else (full if ch.read_var_floor <= 0.0 else 0.0)
                )
                info[f"{ch.quantity}:read"] = wl * min(full, cap)
        return info

    def predict(self, pose: Pose, sensor: SensorOption) -> PredictedOutcome:
        key = (tuple(round(x, 6) for x in (*pose.position_m, *pose.orientation_wxyz)), str(sensor.sensor_id))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        info = self.expected_information(pose, sensor)
        var = {s.key: s.var for s in self.scalars}
        out = PredictedOutcome(
            noise_std={k: information_equivalent_std(var[k], v) for k, v in info.items()},
            hypothesis_accuracy={},
        )
        self._cache[key] = out
        return out
