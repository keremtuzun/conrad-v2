"""Per-observation cell masses: measurement splats, ray free space, then one pose blur. BELIEF PLANE.

1. Hits are splatted with the MEASUREMENT sigma only (range noise, sonar bin/beam width).
2. Free space is sampled along each ray from the sensor up to ``range - k*sigma`` (never through the hit,
   never beyond max range, never on a ray without a return); samples at <= res/2 spacing, consecutive
   samples in one cell count once per ray (sampling traversal, not an exact DDA).
3. One observation gives at most unit hit and unit free mass per cell (hits win inside the scan).
4. Pose error is COMMON-MODE for the whole scan, so the scan's hit and free fields are blurred together
   with one normalised Gaussian kernel of the pose sigma. A blurred value is the probability that the
   surface (or free space) is in that cell given the pose uncertainty: a large sigma spreads the footprint
   and lowers every cell's evidence, while open water far from any surface stays free.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from conrad.domains.spatial.config import SpatialConfig
from conrad.domains.spatial.keys import FloatArr, IntArr, cube_offsets, decode, encode, point_to_index


@dataclass(frozen=True)
class Partial:
    codes: IntArr
    mass: FloatArr
    ua: FloatArr  # mass-weighted mean


@dataclass(frozen=True)
class CellMasses:
    codes: IntArr
    hit: FloatArr
    miss: FloatArr
    ua: FloatArr
    ue: FloatArr
    sigma: FloatArr

    def __len__(self) -> int:
        return len(self.codes)

    def select(self, keep: np.ndarray) -> CellMasses:
        arrays = (self.codes, self.hit, self.miss, self.ua, self.ue, self.sigma)
        return CellMasses(*(np.asarray(a)[keep] for a in arrays))


MassFn = Callable[[float, bool], "CellMasses | None"]
"""(resolution, tail_only) -> masses of one observation, or None = inherit the coarser level's masses."""


def empty_partial() -> Partial:
    z = np.zeros(0)
    return Partial(np.zeros(0, dtype=np.int64), z, z)


def _aggregate(codes: IntArr, mass: FloatArr, ua: FloatArr) -> Partial:
    keep = mass > 0
    if not keep.any():
        return empty_partial()
    uniq, inv = np.unique(codes[keep], return_inverse=True)
    m = np.asarray(np.bincount(inv, weights=mass[keep]), dtype=np.float64)
    ua_m = np.asarray(np.bincount(inv, weights=mass[keep] * ua[keep]) / np.maximum(m, 1e-12))
    return Partial(uniq, m, ua_m)


def _stencil(res: float, sigma: float, cfg: SpatialConfig) -> tuple[IntArr, FloatArr]:
    """Offsets and normalised Gaussian weights for a kernel of ``sigma`` metres at resolution ``res``."""
    pc = cfg.pose
    r = min(math.ceil(pc.splat_sigma_k * sigma / res), pc.max_splat_radius_cells)
    if sigma < 0.25 * res or r == 0:
        return np.zeros((1, 3), dtype=np.int64), np.ones(1)
    offs = cube_offsets(max(r, 1))
    w = np.exp(-np.sum((offs * res) ** 2, axis=1) / (2.0 * sigma**2))
    return offs, np.asarray(w / w.sum())


def hit_masses(
    res: float, points: FloatArr, sigma_meas: FloatArr, mass: FloatArr, ua: FloatArr, cfg: SpatialConfig
) -> Partial:
    """Splat returns over neighbouring cells with Gaussian weights (measurement noise), normalised per return."""
    if len(points) == 0:
        return empty_partial()
    pc = cfg.pose
    radius = np.minimum(np.ceil(pc.splat_sigma_k * sigma_meas / res), pc.max_splat_radius_cells).astype(
        np.int64
    )
    radius = np.where(sigma_meas < 0.25 * res, 0, np.maximum(radius, 1))
    codes, masses, uas = [], [], []
    base = point_to_index(points, res)
    for r in np.unique(radius).tolist():
        sel = np.nonzero(radius == r)[0]
        offs = cube_offsets(int(r))
        cells = base[sel, None, :] + offs[None, :, :]
        if r == 0:
            wts = np.ones((len(sel), 1))
        else:
            d2 = np.sum(((cells + 0.5) * res - points[sel, None, :]) ** 2, axis=2)
            wts = np.exp(-d2 / (2.0 * np.maximum(sigma_meas[sel, None], 1e-9) ** 2))
            wts /= np.maximum(wts.sum(axis=1, keepdims=True), 1e-12)
        codes.append(encode(cells.reshape(-1, 3)))
        masses.append((wts * mass[sel, None]).reshape(-1))
        uas.append(np.repeat(ua[sel], offs.shape[0]))
    return _aggregate(np.concatenate(codes), np.concatenate(masses), np.concatenate(uas))


def free_masses(
    res: float,
    origin: FloatArr,
    dirs: FloatArr,
    t_start: FloatArr,
    t_end: FloatArr,
    weight: FloatArr,
    ua: FloatArr,
    cfg: SpatialConfig,
) -> Partial:
    """Free-space mass along rays ``origin + t * dirs`` for t in [t_start, t_end), one count per ray-cell."""
    ok = t_end > t_start
    if not ok.any():
        return empty_partial()
    dirs, t0, t1, weight, ua = dirs[ok], t_start[ok], t_end[ok], weight[ok], ua[ok]
    step = res * cfg.sensor.ray_step_fraction
    n = math.ceil(float(np.max(t1 - t0)) / step) + 1
    t = t0[:, None] + step * (np.arange(n)[None, :] + 0.5)
    valid = t < t1[:, None]
    pts = origin[None, None, :] + t[..., None] * dirs[:, None, :]
    codes = encode(point_to_index(pts.reshape(-1, 3), res)).reshape(t.shape)
    first = np.ones_like(valid)
    first[:, 1:] = codes[:, 1:] != codes[:, :-1]
    keep = valid & first
    w = np.broadcast_to(weight[:, None], t.shape)
    uas = np.broadcast_to(ua[:, None], t.shape)
    return _aggregate(codes[keep], w[keep], uas[keep])


def combine(hits: Partial, frees: Partial, cfg: SpatialConfig) -> CellMasses:
    """Union of hit and free contributions for ONE observation (before the pose blur)."""
    codes = np.union1d(hits.codes, frees.codes)
    n = len(codes)
    h, m, acc_ua = np.zeros(n), np.zeros(n), np.zeros(n)
    for part, target in ((hits, h), (frees, m)):
        if len(part.codes):
            pos = np.searchsorted(codes, part.codes)
            target[pos] = part.mass
            acc_ua[pos] += part.mass * part.ua
    ua = acc_ua / np.maximum(h + m, 1e-12)
    if cfg.occupancy.per_evidence_mass_cap:
        h = np.minimum(h, 1.0)
        m = np.minimum(m, 1.0) * (1.0 - h)  # a cell returning an echo in this scan is not also free
    else:
        m = np.where(h > 0, 0.0, m)  # classical grid: per-ray counts, hits win within one scan
    z = np.zeros(n)
    return CellMasses(codes, h, m, ua, z, z.copy())


def pose_blur(cm: CellMasses, res: float, sigma: float, cfg: SpatialConfig) -> CellMasses:
    """Blur one observation's hit/free fields with the (common-mode) pose sigma; U_E = sigma/(sigma+res)."""
    ue = sigma / (sigma + res) if sigma > 0 else 0.0
    offs, w = _stencil(res, sigma, cfg)
    if len(cm) == 0 or len(w) == 1:
        n = len(cm)
        return CellMasses(cm.codes, cm.hit, cm.miss, cm.ua, np.full(n, ue), np.full(n, sigma))
    cells = (decode(cm.codes)[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    uniq, inv = np.unique(encode(cells), return_inverse=True)
    ww = np.broadcast_to(w[None, :], (len(cm), len(w)))
    h = np.asarray(np.bincount(inv, weights=(cm.hit[:, None] * ww).reshape(-1)), dtype=np.float64)
    m = np.asarray(np.bincount(inv, weights=(cm.miss[:, None] * ww).reshape(-1)), dtype=np.float64)
    tot = (cm.hit + cm.miss)[:, None] * ww
    ua = np.bincount(inv, weights=(cm.ua[:, None] * tot).reshape(-1)) / np.maximum(h + m, 1e-12)
    keep = (h + m) > 1e-6
    n = int(keep.sum())
    return CellMasses(uniq[keep], h[keep], m[keep], ua[keep], np.full(n, ue), np.full(n, sigma))
