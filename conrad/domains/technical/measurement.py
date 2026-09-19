"""Sensor-characterised measurement model for wall loss and crack length (BELIEF PLANE).

Model2T's reported ``corrosion_depth_m`` / ``crack_length_m`` are component worst-case values. A reading of
either comes from a sensor with the declared :class:`SensorCharacteristics` (an ENGINEERING_ESTIMATE, not the
simulator's sensor): relative sizing error, a persistent per-sensor bias, a probability of detection that
grows with crack length, and an unknown in-view fraction of the defect. The update is exact Bayes on a 1-D
grid for the level, moment-matched back to the Gaussian [level, rate] belief through an equivalent linear
pseudo-measurement, so the rate still learns from repeated inspections.

* Non-detection (crack reading below the call threshold) is censored evidence: its likelihood is
  P(no call | L), which falls only slowly with L because a partial view or a tight crack can be missed.
  It lowers the probability of large cracks; it never sets the length to about zero.
* Partial view: every reading is a mixture of a full view (f = 1) and a partial view f ~ U(f_min, 1), so
  a reading below the belief is a plausible lower bound, not a contradiction.
* Correlated readings: readings of one sensor on one component share the persistent bias. A reading's
  likelihood carries only its independent scatter; the caller floors the posterior at the bias
  (``direct._characterised``), so repeated looks never average the bias away.
* Contradiction: the reading's plausible set (likelihood within a factor of its maximum) holds less than
  ``contradiction_prior_mass`` of the current belief.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from conrad.domains.technical.config import SensorCharacteristics
from conrad.domains.technical.registry import CORROSION_DEPTH, CRACK_LENGTH


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoid rule (numpy renamed trapz -> trapezoid in 2.0; this works on both)."""
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) * 0.5)


MODELLED = frozenset({CORROSION_DEPTH, CRACK_LENGTH})


@dataclass(frozen=True)
class Reading:
    quantity: str
    value: float
    reliability: float
    aleatoric: float
    weight: float = 1.0
    """Tempering exponent in (0, 1] (1 = an ordinary likelihood)."""
    elsewhere: bool = False
    """The reading's measured surface point is away from where this component's worst indication was seen:
    it views another region, so it bounds the worst case from below and a miss says nothing about it."""

    def detected(self, sc: SensorCharacteristics) -> bool:
        return self.quantity != CRACK_LENGTH or self.value >= sc.crack_call_threshold_m


@dataclass(frozen=True)
class GridPosterior:
    mean: float
    var: float
    surprise: bool
    """The belief put less than ``contradiction_prior_mass`` on the reading's plausible set."""
    mode: float
    """Most likely level under the reading alone (0 for a non-detection)."""


def noise_scale(aleatoric: float, sc: SensorCharacteristics) -> float:
    """Multiplier on every declared scatter / noise-floor sigma: 1 for a healthy sensor (reported aleatoric level
    at or below ``noise_ua_reference``), growing with the excess degradation."""
    return 1.0 + sc.noise_ua_gain * max(0.0, aleatoric - sc.noise_ua_reference)


def scatter_sigmas(q: str, value: float, aleatoric: float, sc: SensorCharacteristics) -> tuple[float, float]:
    """(independent per-reading relative sd, persistent per-sensor relative sd) for a reading of size value."""
    if q == CRACK_LENGTH:
        rel, sys_, absolute = sc.crack_rel_sigma, sc.crack_systematic_rel_sigma, sc.crack_abs_sigma_m
    else:
        rel, sys_, absolute = sc.wall_rel_sigma, sc.wall_systematic_rel_sigma, sc.wall_abs_sigma_m
    scale = noise_scale(aleatoric, sc)
    ind = math.hypot(rel * scale, absolute * scale / max(abs(value), absolute))
    return ind, sys_


def min_fraction(q: str, sc: SensorCharacteristics) -> float:
    return sc.crack_partial_view_min_fraction if q == CRACK_LENGTH else sc.wall_partial_view_min_fraction


def view_mixture(q: str, reliability: float, sc: SensorCharacteristics) -> tuple[np.ndarray, np.ndarray]:
    """In-view fraction nodes and weights. A less reliable (degraded) view is somewhat less likely to have
    covered everything: P(full) = full_view_prob * (0.5 + 0.5 * reliability)."""
    p_full = min(1.0, max(0.0, sc.full_view_prob * (0.5 + 0.5 * reliability)))
    n = max(1, sc.partial_view_nodes)
    lo = min_fraction(q, sc)
    partial = lo + (np.arange(n) + 0.5) * (1.0 - lo) / n
    f = np.concatenate([[1.0], partial])
    w = np.concatenate([[p_full], np.full(n, (1.0 - p_full) / n)])
    return f, w


def _pod(a: np.ndarray, aleatoric: float, sc: SensorCharacteristics) -> np.ndarray:
    a50 = sc.crack_pod_a50_m * (1.0 + sc.pod_ua_gain * aleatoric)
    z = (np.log(np.maximum(a, 1e-9)) - math.log(a50)) / sc.crack_pod_log_width
    return np.where(a > 0.0, 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0))), 0.0)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return np.asarray(ndtr(x), dtype=np.float64)


def log_likelihood(r: Reading, grid: np.ndarray, sc: SensorCharacteristics) -> np.ndarray:
    """log p(reading | worst-case level L) on a grid of L >= 0 (untempered)."""
    if r.elsewhere:
        return _elsewhere_log_likelihood(r, grid, sc)
    f, w = view_mixture(r.quantity, r.reliability, sc)
    a = np.maximum(grid, 0.0)[:, None] * f[None, :]  # in-view size per (L, f)
    scale = noise_scale(r.aleatoric, sc)
    if r.quantity == CORROSION_DEPTH:
        mu = sc.wall_sizing_median_factor * a
        sd = np.sqrt((sc.wall_rel_sigma * scale * mu) ** 2 + (sc.wall_abs_sigma_m * scale) ** 2)
        dens = np.exp(-0.5 * ((r.value - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
        lik = dens @ w
        return np.log(np.maximum(lik, 1e-300))
    pod = _pod(a, r.aleatoric, sc)
    med = np.maximum(sc.crack_sizing_median_factor * a, 1e-9)
    floor = sc.crack_abs_sigma_m * scale
    s = np.minimum(np.sqrt((sc.crack_rel_sigma * scale) ** 2 + (floor / med) ** 2), 3.0)
    call = sc.crack_call_threshold_m
    if r.detected(sc):
        z = max(r.value, 1e-9)
        dens = np.exp(-0.5 * ((math.log(z) - np.log(med)) / s) ** 2) / (z * s * math.sqrt(2 * math.pi))
        noise = 2.0 * math.exp(-0.5 * (z / floor) ** 2) / (floor * math.sqrt(2 * math.pi))
        lik = (pod * dens + (1.0 - pod) * noise) @ w
    else:
        below = _norm_cdf((math.log(call) - np.log(med)) / s)
        noise_below = 2.0 * float(_norm_cdf(np.asarray(call / floor))) - 1.0
        lik = ((1.0 - pod) * noise_below + pod * below) @ w
    return np.log(np.maximum(lik, 1e-300))


def _elsewhere_log_likelihood(r: Reading, grid: np.ndarray, sc: SensorCharacteristics) -> np.ndarray:
    """Another region's reading: P(region value <= worst case), a soft lower bound; a miss is uninformative."""
    if not r.detected(sc):
        return np.zeros_like(grid)
    scale = noise_scale(r.aleatoric, sc)
    if r.quantity == CORROSION_DEPTH:
        mu = sc.wall_sizing_median_factor * np.maximum(grid, 0.0)
        sd = np.sqrt((sc.wall_rel_sigma * scale * mu) ** 2 + (sc.wall_abs_sigma_m * scale) ** 2)
        cdf = _norm_cdf((mu - r.value) / sd)
    else:
        med = np.maximum(sc.crack_sizing_median_factor * np.maximum(grid, 0.0), 1e-9)
        s = math.hypot(sc.crack_rel_sigma * scale, sc.crack_systematic_rel_sigma)
        cdf = _norm_cdf((np.log(med) - math.log(max(r.value, 1e-9))) / s)
    return np.log(np.maximum(cdf, 1e-300))


def _grid(
    mean: float, var: float, r: Reading, sc: SensorCharacteristics, tail_scale: float = 0.0
) -> np.ndarray:
    sd = math.sqrt(max(var, 1e-18))
    lo_fine, hi_fine = max(0.0, mean - 8.0 * sd), max(mean + 8.0 * sd, 1e-6)
    top = max(hi_fine, 4.0 * sc.crack_call_threshold_m, 1e-3, tail_scale)
    if r.quantity == CRACK_LENGTH:
        top = max(top, 4.0 * sc.crack_pod_a50_m * (1.0 + sc.pod_ua_gain * r.aleatoric))
        if r.detected(sc):
            top = max(top, 4.0 * r.value / (sc.crack_sizing_median_factor * min_fraction(r.quantity, sc)))
    else:
        top = max(top, 4.0 * r.value / (sc.wall_sizing_median_factor * min_fraction(r.quantity, sc)))
    n = max(sc.grid_points // 2, 50)
    coarse = np.concatenate([[0.0], np.geomspace(top * 1e-4, top, n)])
    fine = np.linspace(lo_fine, hi_fine, n)
    return np.unique(np.concatenate([coarse, fine]))


def _moments(grid: np.ndarray, logp: np.ndarray) -> tuple[float, float] | None:
    if not np.any(np.isfinite(logp)):
        return None
    p = np.exp(logp - np.max(logp))
    mass = _trapz(p, grid)
    if not mass > 0.0:
        return None
    m = float(_trapz(p * grid, grid) / mass)
    v = float(_trapz(p * (grid - m) ** 2, grid) / mass)
    return m, max(v, 1e-18)


def tail_moments(mean: float, var: float, weight: float, scale: float) -> tuple[float, float]:
    """Mean and variance of the heavy-tailed population prior (1 - w) N(mean, var) + w Exp(scale)."""
    m = (1.0 - weight) * mean + weight * scale
    second = (1.0 - weight) * (var + mean * mean) + weight * 2.0 * scale * scale
    return m, max(second - m * m, var)


def _log_prior(grid: np.ndarray, mean: float, var: float, tail: tuple[float, float] | None) -> np.ndarray:
    g = np.exp(-0.5 * (grid - mean) ** 2 / max(var, 1e-18)) / math.sqrt(2 * math.pi * max(var, 1e-18))
    if tail is not None and tail[0] > 0.0:
        w, scale = tail
        g = (1.0 - w) * g + w * np.exp(-grid / scale) / scale
    return np.log(np.maximum(g, 1e-300))


def grid_update(
    mean: float,
    var: float,
    r: Reading,
    sc: SensorCharacteristics,
    *,
    inflate_on_surprise: bool,
    prior_tail: tuple[float, float] | None = None,
) -> GridPosterior:
    """Posterior moments of the worst-case level after one (tempered) reading.

    ``prior_tail`` = (weight, exponential scale): a heavy-tailed population prior for a component with no
    direct evidence yet (a minority of components carry real defects far outside the Gaussian core)."""
    grid = _grid(mean, var, r, sc, 8.0 * prior_tail[1] if prior_tail else 0.0)
    ll = log_likelihood(r, grid, sc)
    mode = float(grid[int(np.argmax(ll))])
    plausible = ll >= ll.max() + math.log(sc.plausible_likelihood_ratio)
    log_prior = _log_prior(grid, mean, var, prior_tail)
    prior = np.exp(log_prior - log_prior.max())
    total = _trapz(prior, grid)
    in_set = _trapz(np.where(plausible, prior, 0.0), grid)
    surprise = not (total > 0.0 and in_set / total >= sc.contradiction_prior_mass)
    if surprise and inflate_on_surprise:
        # Innovation-matched inflation: a genuine change is followed instead of being smoothed away.
        var = var + (mode - mean) ** 2
        grid = _grid(mean, var, r, sc, 8.0 * prior_tail[1] if prior_tail else 0.0)
        ll = log_likelihood(r, grid, sc)
        log_prior = _log_prior(grid, mean, var, prior_tail)
    post = _moments(grid, log_prior + r.weight * ll)
    if post is None:
        post = _moments(grid, r.weight * ll) or (mean, var)
    return GridPosterior(post[0], post[1], surprise, mode)
