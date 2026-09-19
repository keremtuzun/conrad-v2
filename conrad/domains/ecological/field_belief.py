"""Field beliefs: hierarchical empirical-Bayes field model over a coarse grid (ch12 field stream).

value(x, t) = level(t) + deviation(x, t)       (ch12 field hierarchy: regional context -> prior -> local field)

* STATIONS. Point readings are pooled by sensor location (a new reading joins a station within
  ``station_merge_radius_m``). Each station runs a local-level filter: its value is a random walk with drift
  rate q (variance per second), read through the sensor noise rho * r (r = stated noise / reliability).
  q and rho are learned from the data by a temporal variogram of the readings at each station: the slope
  (lag-2 minus lag-1 squared differences) gives q independently of the noise, the lag-1 intercept gives rho.
  q is shrunk toward the prior 2 local_sd^2 / correlation_time_s (``drift_prior_exposure`` correlation
  times of evidence), rho toward 1 (``noise_prior_pairs``). Station filters are re-run when they change.
* KRIGING WITH AN UNKNOWN LEVEL. Station values m_s(t) with variance v_s(t) = P_s + q (t - t_s) are combined as
  m ~ N(mu0 1, P0 1 1^T + tau^2 K + diag(v)); level ~ N(mu0, P0) is the domain level (estimated from ALL
  stations), the deviation is a GP with an anisotropic squared-exponential kernel K and variance tau^2. Cell
  means are the exact Gaussian posterior: deviations are shrunk toward the level according to their evidence.
* EMPIRICAL BAYES tau^2. tau^2 is re-estimated after each update by maximum a-posteriori marginal likelihood
  on a log grid, with a weak log-normal prior centred on the configured local_sd^2. With one station the prior
  decides; with several, the between-station spread decides.

Baselines through the same switches: ``spatial_correlation=False`` fixes tau^2 = 0 (one uniform level);
``field_dynamics=False`` fixes q = 0 (static: every reading is averaged forever, no variance growth).
Only marginal cell variances are exposed (``resid_var`` absorbs the level/deviation posterior covariance so
that level_var + resid_var is the exact marginal variance of the cell value).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from conrad.domains.ecological.config import BeliefGridConfig, FieldSpec, Model2EConfig
from conrad.schemas.timebase import NS_PER_S

_EMA = 0.2


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
        return kernel_between(self.centers, np.asarray(p, dtype=np.float64)[None], spec)[:, 0]


def kernel_between(
    a: np.ndarray, b: np.ndarray, spec: FieldSpec, mult: tuple[float, float] = (1.0, 1.0)
) -> np.ndarray:
    """Anisotropic squared-exponential correlation between point sets a (N,3) and b (M,3).

    ``mult`` scales the configured (horizontal, vertical) length scales (empirical-Bayes choice)."""
    d = a[:, None, :] - b[None, :, :]
    lh, lv = spec.length_scale_h_m * mult[0], spec.length_scale_v_m * mult[1]
    q = (d[..., 0] ** 2 + d[..., 1] ** 2) / lh**2 + d[..., 2] ** 2 / lv**2
    return np.asarray(np.exp(-0.5 * q))


@dataclass
class Station:
    """One sensor location; per component a local-level (random-walk) filter and a time-sorted buffer."""

    pos: np.ndarray
    n_pos: int
    m: np.ndarray  # (C,) filtered value at t_ns[c]
    P: np.ndarray  # (C,) its variance; inf = no reading yet
    t_ns: list[int | None]
    buf: list[list[tuple[int, float, float]]]  # per component: (t_ns, value, meas_var)
    # per component: change points found by the last filter run, as (t_ns, intervention variance)
    changes: list[list[tuple[int, float]]] = field(default_factory=list)
    qa: np.ndarray = field(default_factory=lambda: np.zeros(1))  # (C,) innovation-driven extra drift rate


@dataclass
class Fit:
    level: np.ndarray  # (C,)
    level_var: np.ndarray  # (C,)
    resid: np.ndarray  # (C, cells)
    resid_var: np.ndarray  # (C, cells)


@dataclass
class FieldBelief:
    name: str
    spec: FieldSpec
    tau2: np.ndarray  # (C,) empirical-Bayes local-deviation variance
    q: np.ndarray  # (C,) learned station drift rate, variance per second
    ls_mult: np.ndarray = field(default_factory=lambda: np.ones((1, 2)))  # (C, 2) EB length-scale multipliers
    stations: list[Station] = field(default_factory=list)
    sink: np.ndarray | None = None  # (C, cells) entity->field removal not yet seen by a sensor
    sink_ns: int | None = None
    time_ns: int | None = None
    n_obs: int = 0
    nis_ema: float = 1.0
    ood_ema: float = 0.0
    meas_var_ema: float = 0.0
    late_count: int = 0
    noise_scale: np.ndarray = field(default_factory=lambda: np.ones(1))  # (C,) EB factor on stated noise
    # temporal variogram sufficient statistics per component (see _update_temporal)
    vstats: np.ndarray = field(default_factory=lambda: np.zeros((1, 6)))
    max_local_var: float = 0.0
    hyper_dirty: bool = False
    fitter: Callable[[FieldBelief, int], Fit] | None = None
    _cache: tuple[int, Fit] | None = None

    def invalidate(self) -> None:
        self._cache = None

    def fit(self, t_ns: int | None = None) -> Fit:
        t = self.time_ns if t_ns is None else t_ns
        t = 0 if t is None else t
        if self._cache is not None and self._cache[0] == t:
            return self._cache[1]
        assert self.fitter is not None
        f = self.fitter(self, t)
        self._cache = (t, f)
        return f

    @property
    def level(self) -> np.ndarray:
        return self.fit().level

    @property
    def level_var(self) -> np.ndarray:
        return self.fit().level_var

    @property
    def resid(self) -> np.ndarray:
        return self.fit().resid

    @property
    def resid_var(self) -> np.ndarray:
        return self.fit().resid_var

    @property
    def mean(self) -> np.ndarray:
        f = self.fit()
        return np.asarray(f.level[:, None] + f.resid)

    @property
    def var(self) -> np.ndarray:
        f = self.fit()
        return np.asarray(f.level_var[:, None] + f.resid_var)

    @property
    def local_var(self) -> float:
        """Current (empirical-Bayes) local-deviation variance; the UO / coverage reference scale."""
        return float(max(self.tau2[0], 1e-12))

    @property
    def prior_var(self) -> float:
        """Upper bound of any cell variance: level prior + the largest admissible tau^2."""
        return self.spec.prior_sd**2 + self.max_local_var


@dataclass(frozen=True)
class PointUpdate:
    innovation: float
    innovation_var: float
    late: bool
    lag_s: float


Moments = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class FieldBeliefGrid:
    """All field beliefs over one coarse grid. Pure numpy; no identity or persistence here."""

    def __init__(self, config: Model2EConfig) -> None:
        self.cfg = config
        self.fm = config.field_model
        self.grid = BeliefGrid(config.grid)
        lo, hi = self.fm.tau2_grid_decades
        self._decades = np.arange(round(lo * 10), round(hi * 10) + 1) / 10.0
        self._z_ref = float(np.mean(self.grid.centers[:, 2]))
        max_dz2 = float(np.max((self.grid.centers[:, 2] - self._z_ref) ** 2))
        self.fields: dict[str, FieldBelief] = {}
        for name, spec in config.fields.items():
            c = spec.components
            tau0 = spec.local_sd**2 if config.switches.spatial_correlation else 0.0
            fb = FieldBelief(
                name=name,
                spec=spec,
                tau2=np.full(c, tau0),
                q=np.full(c, self._q_prior(spec)),
                ls_mult=np.ones((c, 2)),
                noise_scale=np.ones(c),
                vstats=np.zeros((c, 6)),
                max_local_var=spec.local_sd**2 * 10.0**hi + self._trend_var(spec) * max_dz2,
                fitter=self._fit,
            )
            self.fields[name] = fb

    # ------------------------------------------------------------------ level + depth trend (the mean)
    def _trend_var(self, spec: FieldSpec) -> float:
        return spec.depth_trend_sd**2 if self.cfg.switches.spatial_correlation else 0.0

    def _basis(self, spec: FieldSpec, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean basis H (N,B) = [1, z - z_ref] and its prior covariance diag(prior_sd^2, depth_trend_sd^2).

        The level is the field value at the grid's mean depth; the depth trend is dropped when its prior sd is 0
        or in the static (spatially uniform) baseline."""
        ones = np.ones((pts.shape[0], 1))
        tv = self._trend_var(spec)
        if tv <= 0:
            return ones, np.array([[spec.prior_sd**2]])
        return np.hstack([ones, pts[:, 2:3] - self._z_ref]), np.diag([spec.prior_sd**2, tv])

    # ------------------------------------------------------------------ hyper-parameters
    def _q_prior(self, spec: FieldSpec) -> float:
        if not self.cfg.switches.field_dynamics:
            return 0.0
        return 2.0 * spec.local_sd**2 / spec.correlation_time_s

    def _accumulate(
        self,
        fb: FieldBelief,
        c: int,
        buf: list[tuple[int, float, float]],
        t_ns: int,
        y: float,
        r: float,
        slope: bool = True,
    ) -> None:
        """Temporal variogram statistics from an in-order reading and its station's previous two.

        lag-1: E[(y_i - y_{i-1})^2] = q dt1 + rho (r_i + r_{i-1});
        slope: E[(y_i - y_{i-2})^2 - (y_i - y_{i-1})^2] = q (t_{i-1} - t_{i-2}) + rho (r_{i-2} - r_{i-1}).
        The slope does not depend on the sensor noise, so q is learned even when the stated noise is wrong.
        ``slope=False`` when the previous reading was a change point (its lag-2 pair spans the jump)."""
        if not buf or t_ns <= buf[-1][0]:
            return
        t1, y1, r1 = buf[-1]
        vs = fb.vstats[c]
        vs[0] += (y - y1) ** 2
        vs[1] += (t_ns - t1) / NS_PER_S
        vs[2] += r + r1
        vs[3] += 1.0
        if slope and len(buf) >= 2 and buf[-2][0] < t1:
            t2, y2, r2 = buf[-2]
            vs[4] += (y - y2) ** 2 - (y - y1) ** 2 - (r2 - r1)
            vs[5] += (t1 - t2) / NS_PER_S

    def _update_temporal(self, fb: FieldBelief) -> None:
        """EB drift rate q (variogram slope, shrunk to the prior) and sensor-noise scale rho (intercept)."""
        q0 = self._q_prior(fb.spec)
        t0 = self.fm.drift_prior_exposure * fb.spec.correlation_time_s
        n0 = self.fm.noise_prior_pairs
        for c in range(fb.spec.components):
            d2, dt1, rs, n1, sa, sb = fb.vstats[c]
            if self.cfg.switches.field_dynamics:
                # the prior counts as `exposure` correlation times of station evidence at rate q0
                fb.q[c] = max(0.0, (q0 * t0 + sa) / (t0 + sb))
            else:
                fb.q[c] = 0.0
            if n1 > 0 and rs > 0:
                rbar = rs / n1
                rho = (n0 * rbar + d2 - fb.q[c] * dt1) / (n0 * rbar + rs)
                lo, hi = self.fm.noise_scale_bounds
                fb.noise_scale[c] = float(np.clip(rho, lo, hi))
        for st in fb.stations:
            for c in range(fb.spec.components):
                self._refilter(st, c, float(fb.q[c]), float(fb.noise_scale[c]))

    def _station_arrays(
        self, fb: FieldBelief, c: int, t_ns: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(positions (S,3), values (S,), variances (S,)) of the stations with a reading of component c.

        A change point seen at one station is a possible jump at every station that has not reported since:
        such stale stations carry the intervention variance of each later change point elsewhere."""
        events = [(t, j, i) for i, s in enumerate(fb.stations) for t, j in s.changes[c] if t <= t_ns]
        pos, m, v = [], [], []
        for i, s in enumerate(fb.stations):
            ts = s.t_ns[c]
            if ts is None:
                continue
            age = max(0.0, (t_ns - ts) / NS_PER_S)
            shock = sum(j for t, j, k in events if k != i and t > ts)
            pos.append(s.pos)
            m.append(s.m[c])
            v.append(s.P[c] + (fb.q[c] + s.qa[c]) * age + shock)
        return np.asarray(pos, dtype=np.float64).reshape(-1, 3), np.asarray(m), np.asarray(v)

    def _update_tau2(self, fb: FieldBelief, t_ns: int) -> None:
        """Empirical Bayes: joint MAP of tau^2 and the kernel length-scale multipliers under the marginal
        likelihood of the station values (log-normal priors centred on the configured values)."""
        s = fb.spec
        if not self.cfg.switches.spatial_correlation:
            fb.tau2[:] = 0.0
            return
        tau0 = s.local_sd**2
        t2 = tau0 * 10.0**self._decades  # (G,)
        lp_t = -0.5 * (self._decades / self.fm.local_var_prior_log10_sd) ** 2
        mults = [(h, v) for h in self.fm.length_scale_multipliers for v in self.fm.length_scale_multipliers]
        lp_l = np.array(
            [
                -0.5 * (math.log2(h) ** 2 + math.log2(v) ** 2) / self.fm.length_scale_prior_log2_sd**2
                for h, v in mults
            ]
        )
        p0 = s.prior_sd**2
        for c in range(s.components):
            pos, m, v = self._station_arrays(fb, c, t_ns)
            if m.size == 0:
                fb.tau2[c] = tau0
                fb.ls_mult[c] = (1.0, 1.0)
                continue
            ks = np.stack([kernel_between(pos, pos, s, hv) for hv in mults])  # (L,S,S)
            r = m - s.prior_mean
            eye = np.eye(m.size)
            hs, pb = self._basis(s, pos)
            mean_cov = hs @ pb @ hs.T
            cmats = (
                mean_cov
                + t2[None, :, None, None] * ks[:, None]
                + np.diag(v)[None, None]
                + self.fm.jitter_fraction * (p0 + t2)[None, :, None, None] * eye
            )  # (L,G,S,S)
            chol = np.linalg.cholesky(cmats)  # batched over the hyper-parameter grid
            rhs = np.broadcast_to(r, (*cmats.shape[:2], m.size))[..., None]
            z = np.linalg.solve(chol, rhs)[..., 0]
            logdet = np.sum(np.log(np.diagonal(chol, axis1=2, axis2=3)), axis=2)
            lp = lp_l[:, None] + lp_t[None, :] - logdet - 0.5 * np.sum(z * z, axis=2)
            li, gi = np.unravel_index(int(np.argmax(lp)), lp.shape)
            fb.tau2[c] = float(t2[gi])
            fb.ls_mult[c] = mults[li]

    # ------------------------------------------------------------------ posterior
    def _fit(self, fb: FieldBelief, t_ns: int) -> Fit:
        if fb.hyper_dirty:  # empirical Bayes once per batch of readings, at the belief's own time
            self._update_temporal(fb)
            self._update_tau2(fb, fb.time_ns if fb.time_ns is not None else t_ns)
            fb.hyper_dirty = False
        return self._posterior(fb, t_ns)

    def _posterior(self, fb: FieldBelief, t_ns: int) -> Fit:
        s = fb.spec
        n_cells = self.grid.n_cells
        comps = s.components
        level = np.full(comps, s.prior_mean, dtype=np.float64)
        level_var = np.full(comps, s.prior_sd**2, dtype=np.float64)
        resid = np.zeros((comps, n_cells))
        resid_var = np.repeat(fb.tau2[:, None], n_cells, axis=1).astype(np.float64)
        p0 = s.prior_sd**2
        for c in range(comps):
            pos, m, v = self._station_arrays(fb, c, t_ns)
            t2 = float(fb.tau2[c])
            if m.size == 0:
                continue
            hv = (float(fb.ls_mult[c, 0]), float(fb.ls_mult[c, 1]))
            k_ss = kernel_between(pos, pos, s, hv)
            k_cs = kernel_between(self.grid.centers, pos, s, hv)
            hs, pb = self._basis(s, pos)
            hc, _ = self._basis(s, self.grid.centers)
            a = pb @ hs.T  # cov(mean coefficients, station values) (B,S)
            cmat = hs @ a + t2 * k_ss + np.diag(v) + self.fm.jitter_fraction * (p0 + t2) * np.eye(m.size)
            cov_cs = hc @ a + t2 * k_cs  # cov(cell value, station values)
            nb = a.shape[0]
            sol = np.linalg.solve(cmat, np.column_stack([m - s.prior_mean, a.T, cov_cs.T]))
            alpha, c_inv_at, c_inv_cov = sol[:, 0], sol[:, 1 : 1 + nb], sol[:, 1 + nb :]
            mean_f = s.prior_mean + cov_cs @ alpha
            prior_f = np.einsum("ij,jk,ik->i", hc, pb, hc) + t2
            var_f = prior_f - np.einsum("ij,ji->i", cov_cs, c_inv_cov)
            level[c] = s.prior_mean + float(a[0] @ alpha)  # the level coefficient (value at mean depth)
            level_var[c] = p0 - float(a[0] @ c_inv_at[:, 0])
            resid[c] = mean_f - level[c]
            resid_var[c] = var_f - level_var[c]
        floor_l = self.cfg.variance_floor_fraction * p0
        floor_r = self.cfg.variance_floor_fraction * s.local_sd**2
        level_var = np.maximum(level_var, floor_l)
        resid_var = np.maximum(resid_var, floor_r)
        if fb.sink is not None:
            resid = resid + fb.sink * self._sink_decay(fb, t_ns)
        if s.nonnegative:
            resid = np.maximum(resid, -level[:, None])
        return Fit(level, level_var, resid, resid_var)

    def _sink_decay(self, fb: FieldBelief, t_ns: int) -> float:
        if fb.sink_ns is None or not self.cfg.switches.field_dynamics:
            return 1.0
        return math.exp(-max(0.0, (t_ns - fb.sink_ns) / NS_PER_S) / fb.spec.correlation_time_s)

    # ------------------------------------------------------------------ temporal
    def _components_at(self, fb: FieldBelief, to_ns: int) -> Moments:
        t = to_ns if fb.time_ns is None else max(to_ns, fb.time_ns)
        f = fb.fit(t)
        return f.level.copy(), f.level_var.copy(), f.resid.copy(), f.resid_var.copy()

    def moments_at_time(self, name: str, to_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """Predicted per-cell (mean, var) at ``to_ns`` WITHOUT mutating the belief."""
        lv, lvar, r, rvar = self._components_at(self.fields[name], to_ns)
        return lv[:, None] + r, lvar[:, None] + rvar

    def predicted_uo(self, name: str, to_ns: int, cells: np.ndarray | None = None) -> float:
        """U_O at ``to_ns``: mean predicted LOCAL variance over the local-deviation scale (1 = never observed)."""
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
        t = fb.time_ns or 0
        mean = fb.mean
        old = np.zeros_like(mean) if fb.sink is None else fb.sink * self._sink_decay(fb, t)
        fb.sink = old + mean * (np.exp(-dt_s * total)[None, :] - 1.0)
        fb.sink_ns = t
        fb.invalidate()

    # ------------------------------------------------------------------ measurement
    def sample(self, name: str, p: np.ndarray, component: int = 0) -> tuple[float, float]:
        fb = self.fields[name]
        f = fb.fit()
        w = self.grid.interp_weights(p)
        r = sum(wi * f.resid[component, c] for c, wi in w)
        rv = sum(wi * f.resid_var[component, c] for c, wi in w)
        return float(f.level[component] + r), float(f.level_var[component] + rv)

    def _station_for(self, fb: FieldBelief, p: np.ndarray) -> Station:
        best, best_d = None, self.fm.station_merge_radius_m
        for s in fb.stations:
            d = float(np.linalg.norm(s.pos - p))
            if d <= best_d:
                best, best_d = s, d
        if best is not None:
            best.n_pos += 1
            best.pos = best.pos + (p - best.pos) / best.n_pos
            return best
        c = fb.spec.components
        st = Station(
            pos=np.asarray(p, dtype=np.float64).copy(),
            n_pos=1,
            m=np.zeros(c),
            P=np.full(c, np.inf),
            t_ns=[None] * c,
            buf=[[] for _ in range(c)],
            changes=[[] for _ in range(c)],
            qa=np.zeros(c),
        )
        fb.stations.append(st)
        if len(fb.stations) > self.fm.max_stations:
            fb.stations.sort(key=lambda x: max((t for t in x.t_ns if t is not None), default=-1))
            fb.stations.pop(0)
        return st

    def _refilter(self, st: Station, c: int, q: float, rho: float) -> None:
        """Re-run a station's local-level filter from its buffer (late reading or new hyper-parameters)."""
        st.P[c], st.t_ns[c] = np.inf, None
        st.changes[c] = []
        st.qa[c] = 0.0
        for t, y, r in st.buf[c]:
            self._filter_step(st, c, t, y, rho * r, q)

    @property
    def _detect_changes(self) -> bool:
        return self.fm.change_detection and self.cfg.switches.field_dynamics

    def _filter_step(self, st: Station, c: int, t_ns: int, y: float, r: float, q: float) -> bool:
        """One local-level step. Returns True when the reading is a CHANGE POINT.

        Change point (West & Harrison intervention): the standardized one-step innovation e^2 / S exceeds
        ``change_z``^2, i.e. the reading is not explained by drift q dt plus sensor noise. The state then
        receives the unexplained variance e^2 - S before the update (method of moments), so the station
        re-acquires the new level in one step instead of averaging the jump away. The intervention is
        recorded so that other stations' stale values get the same variance (``_station_arrays``)."""
        prev = st.t_ns[c]
        changed = False
        if prev is None or not np.isfinite(st.P[c]):
            st.m[c], st.P[c] = y, r
        else:
            dt = max(0.0, (t_ns - prev) / NS_PER_S)
            pp = st.P[c] + (q + st.qa[c]) * dt
            e2 = (y - st.m[c]) ** 2
            s = pp + r
            if self._detect_changes and t_ns >= prev and e2 > self.fm.change_z**2 * s:
                pp += e2 - s
                st.changes[c].append((t_ns, e2 - s))
                changed = True
            elif self._detect_changes and dt > 0:
                # innovation-based process-noise inflation (covariance matching with exponential forgetting):
                # E[e^2] = P + q dt + r under the base model, so the excess over it, per second, is an unbiased
                # estimate of extra drift. Only the running estimate is clipped at 0, so a correct model
                # keeps qa near 0 and a sustained, sub-threshold trend (e.g. settling after a spike) raises it.
                w = 1.0 - math.exp(-dt / self.fm.volatility_memory_s)
                excess = (e2 - (st.P[c] + q * dt + r)) / dt
                st.qa[c] = max(0.0, (1.0 - w) * st.qa[c] + w * excess)
            gain = pp / (pp + r)
            st.m[c] += gain * (y - st.m[c])
            st.P[c] = (1.0 - gain) * pp
        st.t_ns[c] = t_ns if prev is None else max(prev, t_ns)
        return changed

    def update_point(
        self, name: str, component: int, p: np.ndarray, value: float, meas_var: float, t_ns: int
    ) -> PointUpdate:
        fb = self.fields[name]
        c = component
        p = np.asarray(p, dtype=np.float64)
        late = fb.time_ns is not None and t_ns < fb.time_ns
        lag_s = 0.0
        if late:
            assert fb.time_ns is not None
            lag_s = (fb.time_ns - t_ns) / NS_PER_S
            fb.late_count += 1
        # innovation against the belief predicted to the reading's own time (hyper-parameters as they stand)
        f = self._posterior(fb, t_ns)
        w = self.grid.interp_weights(p)
        m_x = float(f.level[c] + sum(wi * f.resid[c, k] for k, wi in w))
        v_x = float(f.level_var[c] + sum(wi * f.resid_var[c, k] for k, wi in w))
        st = self._station_for(fb, p)
        buf = st.buf[c]
        prev = buf[-1] if buf else None
        in_order = prev is None or t_ns >= prev[0]
        rho = float(fb.noise_scale[c])
        if in_order:
            prev_change = bool(st.changes[c]) and bool(buf) and st.changes[c][-1][0] == buf[-1][0]
            if not self._filter_step(st, c, t_ns, value, rho * meas_var, float(fb.q[c])):
                # a jump is neither drift nor sensor noise: change points stay out of the variogram
                self._accumulate(fb, c, buf, t_ns, value, meas_var, slope=not prev_change)
            buf.append((t_ns, value, meas_var))
        else:
            bisect.insort(buf, (t_ns, value, meas_var))
            self._refilter(st, c, float(fb.q[c]), rho)
        if len(buf) > self.fm.max_station_readings:
            del buf[0]
        if not late:
            self.advance(name, t_ns)
        fb.hyper_dirty = True
        fb.invalidate()
        eff_var = rho * meas_var
        s = v_x + eff_var
        innov = value - m_x
        fb.n_obs += 1
        nis = innov * innov / s
        fb.nis_ema = (1 - _EMA) * fb.nis_ema + _EMA * nis
        ood = abs(value - fb.spec.prior_mean) > 4.0 * fb.spec.prior_sd
        fb.ood_ema = (1 - _EMA) * fb.ood_ema + _EMA * float(ood)
        fb.meas_var_ema = (1 - _EMA) * fb.meas_var_ema + _EMA * eff_var if fb.n_obs > 1 else eff_var
        return PointUpdate(innov, s, late, lag_s)

    # ------------------------------------------------------------------ summaries
    def region_cells(self, center: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
        d = np.abs(self.grid.centers - center)
        return np.asarray(np.all(d <= np.maximum(half_extent, 0.5 * self.grid.spacing), axis=1))

    def coverage(self, name: str, cells: np.ndarray | None = None) -> float:
        """Fraction of cells whose LOCAL variance a sensor has at least halved."""
        fb = self.fields[name]
        rv = fb.resid_var
        v = rv if cells is None else rv[:, cells]
        return float(np.mean(v < 0.5 * fb.local_var)) if v.size else 0.0

    def uncertainty_channels(
        self, name: str, cells: np.ndarray | None = None
    ) -> tuple[float, float, float, float]:
        """(UA, UE, UC, UO), driven by inputs: sensor noise, OOD readings, innovation excess, local coverage."""
        fb = self.fields[name]
        rv = fb.resid_var
        v = rv if cells is None else rv[:, cells]
        ua = min(1.0, fb.meas_var_ema / (fb.local_var + fb.meas_var_ema)) if fb.n_obs else 0.0
        ue = float(fb.ood_ema)
        excess = max(0.0, fb.nis_ema - 1.0)
        uc = excess / (1.0 + excess) if fb.n_obs else 0.0
        uo = float(np.clip(np.mean(v) / fb.local_var, 0.0, 1.0)) if v.size else 1.0
        if fb.n_obs == 0:
            uo = 1.0
        return ua, ue, uc, uo
