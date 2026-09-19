"""Regime-mixture crack filter (BELIEF PLANE): a discrete joint over log crack length and growth regime.

Why (docs/audits/MODEL2T_REPAIR.md iteration 3): a constant-rate Gaussian [level, rate] model cannot
represent that most cracks do not grow at all while a minority run away (Paris-type growth). Between
inspections its prediction under-covers run-away cracks, and its rate is learned from sizing noise.

State: p[j, k] = P(regime j, ln L in cell k). Regime 0 = STABLE (no growth); regime j > 0 = RUN-AWAY with
d ln L / dt = g_j. Prediction over PHYSICAL delta_t (seconds, never a step index):
  1. each run-away row is shifted by g_j * dt in ln L; mass pushed past ``max_length_m`` leaves the grid (a
     crack cannot outgrow the component-scale bound, so that hypothesis is refuted, not piled up at the top);
  2. every row diffuses in ln L with sd ``log_diffusion_per_sqrt_yr * sqrt(dt)``;
  3. regimes mix through expm(Q dt) (initiation, arrest, drift between neighbouring rates).
So uncertainty grows with elapsed time and the predictive length is heavy-tailed.
A reading multiplies the joint by the sensor-characterised likelihood (``measurement.log_likelihood``), which
depends on the length only; the regime posterior is learned from how the length changes between readings.
Estimates handed to the rest of Model2T are posterior moments (mean or median, variance, dL/dt moments).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from scipy.ndimage import gaussian_filter1d

from conrad.domains.technical.config import YEAR_S, CrackGrowthConfig, PriorConfig, SensorCharacteristics
from conrad.domains.technical.measurement import Reading, log_likelihood
from conrad.domains.technical.registry import CRACK_LENGTH


@dataclass
class CrackGrid:
    p: np.ndarray
    """(n_regimes, n_cells) joint probability, sums to 1."""
    pending_s: float = 0.0
    """Elapsed physical time not yet applied to ``p``."""

    def copy(self) -> CrackGrid:
        return CrackGrid(self.p.copy(), self.pending_s)


@dataclass(frozen=True)
class CrackMoments:
    level: float
    level_var: float
    rate: float
    rate_var: float
    cov: float


@lru_cache(maxsize=8)
def _axis(lo: float, hi: float, n: int) -> tuple[np.ndarray, float]:
    ln = np.linspace(math.log(lo), math.log(hi), n)
    return ln, float(ln[1] - ln[0])


def lengths(cfg: CrackGrowthConfig) -> np.ndarray:
    ln, _ = _axis(cfg.min_length_m, cfg.max_length_m, cfg.grid_points)
    return np.exp(ln)


def rates(cfg: CrackGrowthConfig) -> np.ndarray:
    return np.concatenate([[0.0], np.asarray(cfg.runaway_rates_per_yr, dtype=np.float64)])


def _generator(cfg: CrackGrowthConfig) -> np.ndarray:
    n = len(cfg.runaway_rates_per_yr)
    q = np.zeros((n + 1, n + 1))
    if n:
        q[0, 1:] = cfg.initiation_per_yr / n
        for j in range(1, n + 1):
            q[j, 0] = cfg.arrest_per_yr
            for nb in (j - 1, j + 1):
                if 1 <= nb <= n:
                    q[j, nb] = cfg.rate_drift_per_yr
    np.fill_diagonal(q, -q.sum(axis=1))
    return q


@lru_cache(maxsize=64)
def _transition(cfg: CrackGrowthConfig, dt_yr: float) -> np.ndarray:
    return np.asarray(expm(_generator(cfg) * dt_yr))


def population_prior(cfg: CrackGrowthConfig, prior: PriorConfig) -> CrackGrid:
    """Never-seen crack: the Model2T population prior (Gaussian core + exponential tail) over length, and
    ``stable_prob`` in the STABLE regime (the rest spread evenly over the run-away rates)."""
    x = lengths(cfg)
    mean, sd = prior.mean[CRACK_LENGTH], prior.sd[CRACK_LENGTH]
    w, scale = prior.tail_weight.get(CRACK_LENGTH, 0.0), prior.tail_scale_m.get(CRACK_LENGTH, 1.0)
    dens = (1.0 - w) * np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    dens = dens + w * np.exp(-x / scale) / scale
    cell = dens * x  # density per unit ln L
    cell = cell / cell.sum()
    n = len(cfg.runaway_rates_per_yr)
    regime = np.concatenate([[cfg.stable_prob], np.full(n, (1.0 - cfg.stable_prob) / max(n, 1))])
    if n == 0:
        regime = np.ones(1)
    return CrackGrid(regime[:, None] * cell[None, :])


def _shift(row: np.ndarray, cells: float) -> np.ndarray:
    """Move a row's mass up by ``cells`` (fractional) cells. Mass pushed past the top leaves the grid: a crack
    cannot outgrow the component-scale bound, so that growth hypothesis is refuted (the caller renormalises)."""
    if cells <= 0.0 or not row.any():
        return row
    n = row.size
    edges = np.arange(n + 1, dtype=np.float64)
    cdf = np.concatenate([[0.0], np.cumsum(row)])
    moved = np.interp(edges - cells, edges, cdf, left=0.0, right=float(cdf[-1]))
    return np.maximum(np.diff(moved), 0.0)


def propagate(grid: CrackGrid, dt_s: float, cfg: CrackGrowthConfig, *, force: bool = False) -> bool:
    """Accumulate ``dt_s`` physical seconds; apply them once they exceed ``min_propagation_s`` (or ``force``).
    Returns True if the grid changed."""
    if dt_s < 0.0:
        raise ValueError("delta_t must be non-negative physical seconds")
    grid.pending_s += dt_s
    if grid.pending_s <= 0.0 or (not force and grid.pending_s < cfg.min_propagation_s):
        return False
    total_yr = grid.pending_s / YEAR_S
    grid.pending_s = 0.0
    n = max(1, math.ceil(total_yr * YEAR_S / cfg.max_substep_s - 1e-9))
    for _ in range(n):
        _step(grid, total_yr / n, cfg)
    return True


def _step(grid: CrackGrid, dt_yr: float, cfg: CrackGrowthConfig) -> None:
    """One operator-split sub-step (shift, diffusion, regime mixing); sub-steps keep the split accurate when a
    long gap is applied at once (a crack that initiates mid-gap still grows for the rest of it)."""
    _, h = _axis(cfg.min_length_m, cfg.max_length_m, cfg.grid_points)
    g = rates(cfg)
    p = grid.p.copy()
    for j in range(1, p.shape[0]):
        p[j] = _shift(p[j], g[j] * dt_yr / h)
    if not p.sum() > 0.0:
        p = grid.p.copy()  # every hypothesis outgrew the bound: keep the belief rather than divide by zero
    sd_cells = cfg.log_diffusion_per_sqrt_yr * math.sqrt(dt_yr) / h
    if sd_cells > 1e-3:
        mass = p.sum(axis=1, keepdims=True)
        p = gaussian_filter1d(p, sd_cells, axis=1, mode="nearest")
        p = np.maximum(p, 0.0)
        p *= mass / np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
    p = _transition(cfg, round(dt_yr, 9)).T @ p
    grid.p = p / p.sum()


@dataclass(frozen=True)
class CrackUpdate:
    surprise: bool
    mode: float
    """Most likely length under the reading alone (0 for a non-detection)."""


def update(
    grid: CrackGrid, reading: Reading, sc: SensorCharacteristics, cfg: CrackGrowthConfig
) -> CrackUpdate:
    """Bayes update of the joint with one reading (pending time must already be applied)."""
    x = lengths(cfg)
    ll = log_likelihood(reading, x, sc)
    detected = reading.detected(sc)
    mode = float(x[int(np.argmax(ll))]) if detected else 0.0
    plausible = ll >= ll.max() + math.log(sc.plausible_likelihood_ratio)
    marginal = grid.p.sum(axis=0)
    surprise = float(marginal[plausible].sum()) < sc.contradiction_prior_mass
    p = grid.p
    if surprise and cfg.surprise_mix > 0.0:
        regime = p.sum(axis=1, keepdims=True)
        p = (1.0 - cfg.surprise_mix) * p + cfg.surprise_mix * regime / p.shape[1]
    lik = np.exp(ll - ll.max())
    post = p * lik[None, :]
    total = float(post.sum())
    if total > 0.0 and math.isfinite(total):
        grid.p = post / total
    return CrackUpdate(surprise, mode)


def floor_log_sd(grid: CrackGrid, log_sd: float, cfg: CrackGrowthConfig) -> None:
    """Persistent sizing bias: the length posterior's ln L spread never drops below ``log_sd``."""
    ln, h = _axis(cfg.min_length_m, cfg.max_length_m, cfg.grid_points)
    marginal = grid.p.sum(axis=0)
    m = float(marginal @ ln)
    var = float(marginal @ (ln - m) ** 2)
    extra = log_sd * log_sd - var
    if extra <= 0.0:
        return
    mass = grid.p.sum(axis=1, keepdims=True)
    p = np.maximum(gaussian_filter1d(grid.p, math.sqrt(extra) / h, axis=1, mode="nearest"), 0.0)
    p *= mass / np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
    grid.p = p / p.sum()


def moments(grid: CrackGrid, cfg: CrackGrowthConfig) -> CrackMoments:
    x = lengths(cfg)
    g = rates(cfg)
    marginal = grid.p.sum(axis=0)
    mean = float(marginal @ x)
    var = max(float(marginal @ (x * x)) - mean * mean, 0.0)
    dldt = g[:, None] * x[None, :]  # m / yr
    rate = float((grid.p * dldt).sum())
    rate_var = max(float((grid.p * dldt * dldt).sum()) - rate * rate, 0.0)
    cov = float((grid.p * dldt * x[None, :]).sum()) - rate * mean
    level = mean
    if cfg.point_estimate == "MEDIAN":
        cdf = np.cumsum(marginal)
        level = float(np.interp(0.5, cdf, x))
    return CrackMoments(level, var, rate, rate_var, cov)


def runaway_probability(grid: CrackGrid) -> float:
    return float(grid.p[1:].sum())
