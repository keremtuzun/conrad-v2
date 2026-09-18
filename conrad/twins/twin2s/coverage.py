"""Cumulative observability V_1:t(x) and three-mask ground truth (T2S-OBS-02). TRUTH PLANE.

Not a sum of visibilities: per query point we keep observation count, best quality, effective independent
views (viewpoint diversity), sensor modality set, last observation time and cumulative coverage.
Repeated identical views do not count as independent observations.

Three masks: OBSERVED comes from coverage under sensor-specific criteria. An unobserved point is
INFERABLE only when an explicit ``inferable`` mask is supplied (e.g. :func:`counterfactual_consensus`);
it is never labelled inferable just because it is hidden-but-nearby. Everything else is UNKNOWN.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from conrad.twins.twin2s.config import ObservedCriteria
from conrad.twins.twin2s.sdf import Arr
from conrad.twins.twin2s.visibility import VisibilityResult
from conrad.twins.twin2s.world import SpatialWorld

Mask = NDArray[np.bool_]


class CoverageAccumulator:
    def __init__(self, points_m: Arr, criteria: ObservedCriteria | None = None) -> None:
        self.points = np.atleast_2d(np.asarray(points_m, dtype=np.float64))
        self.criteria = criteria or ObservedCriteria()
        n = len(self.points)
        self.count = np.zeros(n, dtype=np.int64)
        self.best_quality = np.zeros(n)
        self.last_time_s = np.full(n, np.nan)
        self.modalities: list[set[str]] = [set() for _ in range(n)]
        self._views: list[list[Arr]] = [[] for _ in range(n)]
        self.effective_views = np.zeros(n, dtype=np.int64)

    def add(self, vis: VisibilityResult, time_s: float, modality: str) -> None:
        good = vis.visible & (vis.score >= self.criteria.min_quality)
        self.best_quality = np.maximum(self.best_quality, vis.score)
        cos_thr = np.cos(self.criteria.diversity_angle_rad)
        for i in np.nonzero(good)[0]:
            self.count[i] += 1
            self.last_time_s[i] = time_s
            self.modalities[i].add(modality)
            v = vis.view_dirs[i]
            if all(float(v @ u) < cos_thr for u in self._views[i]):
                self._views[i].append(v)
                self.effective_views[i] += 1

    @property
    def observed(self) -> Mask:
        return np.asarray(self.effective_views >= 1)

    def coverage(self, subset: Mask | None = None) -> float:
        obs = self.observed if subset is None else self.observed[subset]
        return float(obs.mean()) if obs.size else 0.0

    def viewpoint_diversity(self) -> Arr:
        """D_V(x) in [0, 1]: 1 - |mean unit view vector| over effective views (0 = one direction only)."""
        out = np.zeros(len(self.points))
        for i, views in enumerate(self._views):
            if len(views) > 1:
                out[i] = 1.0 - float(np.linalg.norm(np.mean(views, axis=0)))
        return out

    def summary(self) -> dict[str, float]:
        return {
            "cumulative_coverage": self.coverage(),
            "mean_observation_count": float(self.count.mean()) if self.count.size else 0.0,
            "mean_effective_views": float(self.effective_views.mean()) if self.count.size else 0.0,
            "redundancy_ratio": float(self.count.sum() / max(self.effective_views.sum(), 1)),
            "mean_viewpoint_diversity": float(self.viewpoint_diversity().mean()) if self.count.size else 0.0,
        }


def three_masks(observed: Mask, inferable: Mask | None = None) -> dict[str, Mask]:
    """M_observed / M_inferable / M_unknown, mutually exclusive and exhaustive."""
    obs = np.asarray(observed, dtype=bool)
    inf = np.zeros_like(obs) if inferable is None else np.asarray(inferable, dtype=bool) & ~obs
    return {"observed": obs, "inferable": inf, "unknown": ~obs & ~inf}


def counterfactual_consensus(worlds: Sequence[SpatialWorld], points_m: Arr) -> Mask:
    """True where every observation-compatible world agrees on occupancy (the plausible set collapses).

    ``worlds`` must all be compatible with the observations (see counterfactual verification). Points where
    they disagree must stay UNKNOWN / multi-hypothesis.
    """
    if len(worlds) < 2:
        raise ValueError("consensus needs at least two observation-compatible worlds")
    occ = np.stack([w.occupied(points_m) for w in worlds])
    return np.asarray(np.all(occ == occ[0], axis=0))
