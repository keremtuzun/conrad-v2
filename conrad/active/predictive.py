"""Belief-side predictive model for value ranking: what would an observation do to the BELIEF? (ch16, ch18).

A ``PredictiveBelief`` exposes the mission-relevant part of a belief as

    scalars     Gaussian beliefs (mean, variance) over mission-relevant quantities, each with an optional
                ACTION THRESHOLD (e.g. "defect severity above the repair threshold")
    hypotheses  binary hypothesis beliefs P(H true)

and a sensor model ``predict(pose, sensor)`` that says which scalars a candidate would measure (with what
predicted noise) and whether it would return a hypothesis vote (with what believed accuracy). Everything is
computed by the caller from BELIEF (believed map, nominal sensor specification, current posterior), never
from twin truth. The functions below are standard Bayesian preposterior quantities:

    entropy reduction          0.5 ln(var / var_post) per scalar; mutual information of a binary symmetric
                               channel per hypothesis                        (standard EIG)
    expected |error| reduction sqrt(2/pi) (sd - sd_post) per scalar; 2p(1-p) - E[2p'(1-p')] per hypothesis
                               (the belief's own prediction of the mission-relevant error metric)
    misclassification          min(q, 1-q) with q = P(x > threshold), and min(p, 1-p) per hypothesis; the
                               expected post-observation value uses Gauss-Hermite quadrature over the
                               predictive distribution of the posterior mean   (hypothesis discrimination)

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from conrad.active.candidates import SensorOption
from conrad.schemas.frames import Pose

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(24)  # probabilists' Hermite: weight exp(-x^2/2)
_GH_W = _GH_W / _GH_W.sum()
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


@dataclass(frozen=True)
class ScalarBelief:
    key: str
    mean: float
    var: float
    threshold: float | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class HypothesisBelief:
    key: str
    p_true: float
    weight: float = 1.0


@dataclass(frozen=True)
class PredictedOutcome:
    """Predicted measurement of one candidate: scalar key -> noise std, hypothesis key -> vote accuracy."""

    noise_std: Mapping[str, float] = field(default_factory=dict)
    hypothesis_accuracy: Mapping[str, float] = field(default_factory=dict)


class PredictiveBelief(Protocol):
    scalars: tuple[ScalarBelief, ...]
    hypotheses: tuple[HypothesisBelief, ...]
    epistemic: float
    used_modalities: frozenset[str]  # modalities already applied to this belief

    def predict(self, pose: Pose, sensor: SensorOption) -> PredictedOutcome: ...


# ---------------------------------------------------------------------------------------------- primitives
def posterior_var(var: float, std: float) -> float:
    return 1.0 / (1.0 / max(var, 1e-12) + 1.0 / max(std, 1e-9) ** 2)


def _phi(z: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0))), dtype=np.float64)


def _h(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def _vote_posteriors(p: float, acc: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """((P(vote=H), p' | vote=H), (P(vote=not H), p' | vote=not H)) for a binary symmetric channel."""
    pv = p * acc + (1 - p) * (1 - acc)
    pn = 1.0 - pv
    post_v = p * acc / pv if pv > 0 else p
    post_n = p * (1 - acc) / pn if pn > 0 else p
    return (pv, post_v), (pn, post_n)


def scalar_entropy_reduction(var: float, std: float) -> float:
    return 0.5 * math.log(max(var, 1e-12) / posterior_var(var, std))


def hypothesis_information(p: float, acc: float) -> float:
    (pv, a), (pn, b) = _vote_posteriors(p, acc)
    return _h(p) - (pv * _h(a) + pn * _h(b))


def scalar_abs_error_reduction(var: float, std: float) -> float:
    return SQRT_2_OVER_PI * (math.sqrt(max(var, 0.0)) - math.sqrt(posterior_var(var, std)))


def hypothesis_abs_error_reduction(p: float, acc: float) -> float:
    (pv, a), (pn, b) = _vote_posteriors(p, acc)
    return 2 * p * (1 - p) - (pv * 2 * a * (1 - a) + pn * 2 * b * (1 - b))


def scalar_misclassification(mean: float, var: float, threshold: float) -> float:
    q = float(_phi(np.asarray([(mean - threshold) / math.sqrt(max(var, 1e-12))]))[0])
    return min(q, 1.0 - q)


def scalar_misclassification_reduction(mean: float, var: float, threshold: float, std: float) -> float:
    vp = posterior_var(var, std)
    spread = math.sqrt(max(var - vp, 0.0))  # sd of the predictive distribution of the posterior mean
    means = mean + spread * _GH_X
    q = _phi((means - threshold) / math.sqrt(vp))
    after = float(np.sum(_GH_W * np.minimum(q, 1.0 - q)))
    return scalar_misclassification(mean, var, threshold) - after


def hypothesis_misclassification_reduction(p: float, acc: float) -> float:
    (pv, a), (pn, b) = _vote_posteriors(p, acc)
    return min(p, 1 - p) - (pv * min(a, 1 - a) + pn * min(b, 1 - b))


# ---------------------------------------------------------------------------------------------- aggregates
def expected_values(model: PredictiveBelief, outcome: PredictedOutcome) -> dict[str, float]:
    """All three preposterior values of one predicted outcome.

    ``entropy`` is GENERIC (every scalar and hypothesis counts once, whatever its mission weight: standard
    EIG). ``abs_error`` and ``misclassification`` are MISSION-CONDITIONED (``weight`` = mission relevance,
    0 = irrelevant to the need).
    """
    ent = err = mis = 0.0
    for s in model.scalars:
        std = outcome.noise_std.get(s.key)
        if std is None:
            continue
        ent += scalar_entropy_reduction(s.var, std)
        err += s.weight * scalar_abs_error_reduction(s.var, std)
        if s.threshold is not None:
            mis += s.weight * scalar_misclassification_reduction(s.mean, s.var, s.threshold, std)
    for h in model.hypotheses:
        acc = outcome.hypothesis_accuracy.get(h.key)
        if acc is None:
            continue
        ent += hypothesis_information(h.p_true, acc)
        err += h.weight * hypothesis_abs_error_reduction(h.p_true, acc)
        mis += h.weight * hypothesis_misclassification_reduction(h.p_true, acc)
    return {"entropy": ent, "abs_error": err, "misclassification": mis}


def predicted_coverage(model: PredictiveBelief, pose: Pose, sensor: SensorOption) -> float:
    """Mission-weighted fraction of target scalars this (pose, sensor) is predicted to measure."""
    out = model.predict(pose, sensor)
    relevant = [s for s in model.scalars if s.weight > 0] or list(model.scalars)
    total = sum(s.weight for s in relevant) or float(len(relevant))
    if not relevant:
        return 1.0 if out.hypothesis_accuracy else 0.0
    got = sum((s.weight if total else 1.0) for s in relevant if s.key in out.noise_std)
    return float(got / total) if total else 0.0
