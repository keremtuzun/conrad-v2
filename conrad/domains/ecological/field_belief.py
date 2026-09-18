"""Field beliefs: coarse grid of per-cell mean + variance per environmental field (ch12 field stream).

Two scales (ch12 field hierarchy, "regional context -> prior -> local field"):
  value(c) = level + residual(c)
* level: one domain-wide value per component, updated by every reading (regional context);
* residual: per-cell local deviation with a GP-lite rank-1 kernel update. The cross-covariance
  between cell c and sensor location x is k(c, x) * sqrt(v_c * v_x) with an anisotropic squared-
  exponential k. Only marginal variances are stored (no full covariance): an approximation.
Residual variance shrinks near sensors and stays at the local prior far from them.
Prediction is Ornstein-Uhlenbeck over PHYSICAL delta_t (level -> prior mean, residual -> 0), so
variance grows with elapsed time. Each field keeps its own clock: streams are never aligned.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from conrad.domains.ecological.config import BeliefGridConfig, FieldSpec, Model2EConfig
from conrad.schemas.timebase import NS_PER_S

_EMA = 0.2


@dataclass
class FieldBelief:
    name: str
    spec: FieldSpec
    level: np.ndarray  # (components,)
    level_var: np.ndarray  # (components,)
    resid: np.ndarray  # (components, cells)
    resid_var: np.ndarray  # (components, cells)
    time_ns: int | None = None
    n_obs: int = 0
    nis_ema: float = 1.0
    ood_ema: float = 0.0
    meas_var_ema: float = 0.0
    late_count: int = 0

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.level[:, None] + self.resid)

    @property
    def var(self) -> np.ndarray:
        return np.asarray(self.level_var[:, None] + self.resid_var)

    @property
    def local_var(self) -> float:
        return self.spec.local_sd**2

    @property
    def prior_var(self) -> float:
        return self.spec.prior_sd**2 + self.spec.local_sd**2


@dataclass(frozen=True)
class PointUpdate:
    innovation: float
    innovation_var: float
    late: bool
    lag_s: float


class BeliefGrid:
    def __init__(self, cfg: BeliefGridConfig) -> None:
        self.cfg = cfg
        o, s, n = np.asarray(cfg.origin_m), np.asarray(cfg.spacing_m), cfg.shape
        idx = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), axis=-1).reshape(-1, 3)
        self.shape = n
        self.centers = o + (idx + 0.5) * s
        self.origin, self.spacing = o, s

    @property
    def n_cells(self) -> int:
        return int(self.centers.shape[0])

    def cell_of(self, p: np.ndarray) -> int:
        i = np.clip(np.floor((p - self.origin) / self.spacing).astype(int), 0, np.asarray(self.shape) - 1)
        return int(np.ravel_multi_index(tuple(i), self.shape))

    def interp_weights(self, p: np.ndarray) -> list[tuple[int, float]]:
        """Trilinear weights over cell centres (clamped at the grid edge)."""
        f = (np.asarray(p, dtype=np.float64) - self.origin) / self.spacing - 0.5
        n = np.asarray(self.shape)
        f = np.clip(f, 0.0, n - 1)
        i0 = np.minimum(np.floor(f).astype(int), np.maximum(n - 2, 0))
        t = f - i0
        out: list[tuple[int, float]] = []
        for corner in range(8):
            bits = np.array([(corner >> k) & 1 for k in range(3)])
            ii = np.minimum(i0 + bits, n - 1)
            w = float(np.prod(np.where(bits == 1, t, 1.0 - t)))
            if w > 0:
                out.append((int(np.ravel_multi_index(tuple(ii), self.shape)), w))
        return out

    def kernel(self, p: np.ndarray, spec: FieldSpec) -> np.ndarray:
        d = self.centers - np.asarray(p, dtype=np.float64)
        q = (d[:, 0] ** 2 + d[:, 1] ** 2) / spec.length_scale_h_m**2 + d[:, 2] ** 2 / spec.length_scale_v_m**2
        return np.asarray(np.exp(-0.5 * q))


Moments = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class FieldBeliefGrid:
    """All field beliefs over one coarse grid. Pure numpy; no identity or persistence here."""

    def __init__(self, config: Model2EConfig) -> None:
        self.cfg = config
        self.grid = BeliefGrid(config.grid)
        n = self.grid.n_cells
        self.fields: dict[str, FieldBelief] = {
            name: FieldBelief(
                name=name,
                spec=spec,
                level=np.full(spec.components, spec.prior_mean, dtype=np.float64),
                level_var=np.full(spec.components, spec.prior_sd**2, dtype=np.float64),
                resid=np.zeros((spec.components, n), dtype=np.float64),
                resid_var=np.full((spec.components, n), spec.local_sd**2, dtype=np.float64),
            )
            for name, spec in config.fields.items()
        }

    # ------------------------------------------------------------------ temporal
    def _components_at(self, fb: FieldBelief, to_ns: int) -> Moments:
        if fb.time_ns is None or to_ns <= fb.time_ns or not self.cfg.switches.field_dynamics:
            return fb.level.copy(), fb.level_var.copy(), fb.resid.copy(), fb.resid_var.copy()
        dt = (to_ns - fb.time_ns) / NS_PER_S
        a = math.exp(-dt / fb.spec.level_correlation_time_s)
        b = math.exp(-dt / fb.spec.correlation_time_s)
        s = fb.spec
        return (
            s.prior_mean + a * (fb.level - s.prior_mean),
            a * a * fb.level_var + (1.0 - a * a) * s.prior_sd**2,
            b * fb.resid,
            b * b * fb.resid_var + (1.0 - b * b) * s.local_sd**2,
        )

    def moments_at_time(self, name: str, to_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """Predicted per-cell (mean, var) at ``to_ns`` WITHOUT mutating the belief."""
        lv, lvar, r, rvar = self._components_at(self.fields[name], to_ns)
        return lv[:, None] + r, lvar[:, None] + rvar

    def predicted_uo(self, name: str, to_ns: int, cells: np.ndarray | None = None) -> float:
        """U_O at ``to_ns``: mean predicted LOCAL variance over the local prior (1 = never observed)."""
        fb = self.fields[name]
        if fb.n_obs == 0:
            return 1.0
        rvar = self._components_at(fb, to_ns)[3]
        v = rvar if cells is None else rvar[:, cells]
        return float(np.clip(np.mean(v) / fb.local_var, 0.0, 1.0)) if v.size else 1.0

    def advance(self, name: str, to_ns: int) -> None:
        fb = self.fields[name]
        if fb.time_ns is not None and to_ns <= fb.time_ns:
            return
        fb.level, fb.level_var, fb.resid, fb.resid_var = self._components_at(fb, to_ns)
        fb.time_ns = to_ns

    def apply_sink(
        self, name: str, positions: list[np.ndarray], rates_per_s: list[float], dt_s: float
    ) -> None:
        """Entity->field message: first-order local removal (e.g. filtration of turbidity)."""
        fb = self.fields[name]
        if dt_s <= 0 or not positions:
            return
        total = np.zeros(self.grid.n_cells)
        for p, r in zip(positions, rates_per_s, strict=True):
            total += r * self.grid.kernel(p, fb.spec)
        mean = fb.mean
        fb.resid = fb.resid + mean * (np.exp(-dt_s * total)[None, :] - 1.0)

    # ------------------------------------------------------------------ measurement
    def sample(self, name: str, p: np.ndarray, component: int = 0) -> tuple[float, float]:
        fb = self.fields[name]
        w = self.grid.interp_weights(p)
        r = sum(wi * fb.resid[component, c] for c, wi in w)
        rv = sum(wi * fb.resid_var[component, c] for c, wi in w)
        return float(fb.level[component] + r), float(fb.level_var[component] + rv)

    def update_point(
        self, name: str, component: int, p: np.ndarray, value: float, meas_var: float, t_ns: int
    ) -> PointUpdate:
        fb = self.fields[name]
        late = fb.time_ns is not None and t_ns < fb.time_ns
        lag_s = 0.0
        if late:
            assert fb.time_ns is not None
            lag_s = (fb.time_ns - t_ns) / NS_PER_S
            b = math.exp(-lag_s / fb.spec.correlation_time_s) if self.cfg.switches.field_dynamics else 1.0
            meas_var = meas_var + (1.0 - b * b) * fb.prior_var  # old news says less about now
            fb.late_count += 1
        else:
            self.advance(name, t_ns)
        m_x, v_x = self.sample(name, p, component)
        rv_x = v_x - fb.level_var[component]
        s = v_x + meas_var
        innov = value - m_x
        lv = fb.level_var[component]
        fb.level[component] += lv / s * innov
        fb.level_var[component] = max(
            lv - lv * lv / s, self.cfg.variance_floor_fraction * fb.spec.prior_sd**2
        )
        if self.cfg.switches.spatial_correlation:  # static-field baseline: level only, no local structure
            k = self.grid.kernel(p, fb.spec)
            cov = k * np.sqrt(fb.resid_var[component] * max(rv_x, 0.0))
            fb.resid[component] = fb.resid[component] + cov / s * innov
            floor = self.cfg.variance_floor_fraction * fb.local_var
            fb.resid_var[component] = np.maximum(fb.resid_var[component] - cov * cov / s, floor)
        if fb.spec.nonnegative:
            fb.resid[component] = np.maximum(fb.resid[component], -fb.level[component])
        fb.n_obs += 1
        nis = innov * innov / s
        fb.nis_ema = (1 - _EMA) * fb.nis_ema + _EMA * nis
        ood = abs(value - fb.spec.prior_mean) > 4.0 * fb.spec.prior_sd
        fb.ood_ema = (1 - _EMA) * fb.ood_ema + _EMA * float(ood)
        fb.meas_var_ema = (1 - _EMA) * fb.meas_var_ema + _EMA * meas_var if fb.n_obs > 1 else meas_var
        return PointUpdate(innov, s, late, lag_s)

    # ------------------------------------------------------------------ summaries
    def region_cells(self, center: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
        d = np.abs(self.grid.centers - center)
        return np.asarray(np.all(d <= np.maximum(half_extent, 0.5 * self.grid.spacing), axis=1))

    def coverage(self, name: str, cells: np.ndarray | None = None) -> float:
        """Fraction of cells whose LOCAL variance a sensor has at least halved."""
        fb = self.fields[name]
        v = fb.resid_var if cells is None else fb.resid_var[:, cells]
        return float(np.mean(v < 0.5 * fb.local_var)) if v.size else 0.0

    def uncertainty_channels(
        self, name: str, cells: np.ndarray | None = None
    ) -> tuple[float, float, float, float]:
        """(UA, UE, UC, UO), driven by inputs: sensor noise, OOD readings, innovation excess, local coverage."""
        fb = self.fields[name]
        v = fb.resid_var if cells is None else fb.resid_var[:, cells]
        ua = min(1.0, fb.meas_var_ema / fb.local_var) if fb.n_obs else 0.0
        ue = float(fb.ood_ema)
        excess = max(0.0, fb.nis_ema - 1.0)
        uc = excess / (1.0 + excess) if fb.n_obs else 0.0
        uo = float(np.clip(np.mean(v) / fb.local_var, 0.0, 1.0)) if v.size else 1.0
        if fb.n_obs == 0:
            uo = 1.0
        return ua, ue, uc, uo
