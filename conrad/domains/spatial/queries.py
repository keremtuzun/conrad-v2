"""Belief-only map queries for other modules (navigation, MCBR). BELIEF PLANE.

UNKNOWN is never occupied inside the map. How a consumer treats UNKNOWN is an explicit policy argument
(:class:`UnknownPolicy`), kept outside the map state (ch14 "Unknown != occupied").

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from conrad.domains.spatial.config import SpatialConfig
from conrad.domains.spatial.grid import OBSERVED, PREDICTED, STATUS_NAMES, UNKNOWN
from conrad.domains.spatial.hierarchy import PointState
from conrad.domains.spatial.keys import FloatArr, index_to_center, points_to_codes, region_indices
from conrad.domains.spatial.sensing import sensor_fov, sensor_frame
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.frames import FrameError, Pose, SpatialSupport
from conrad.schemas.world import SensorSpec


class UnknownPolicy(str, Enum):
    CONSERVATIVE = "CONSERVATIVE"  # UNKNOWN is not free (default for navigation)
    PERMISSIVE = "PERMISSIVE"  # UNKNOWN counts as free; an explicit, caller-owned risk decision


@dataclass(frozen=True)
class VisibilityPrediction:
    points_m: FloatArr
    probability: FloatArr  # per target point: in-FOV * belief transmission
    in_fov: NDArray[np.bool_]
    unknown_cells_on_path: FloatArr
    p_unknown_block: float

    @property
    def mean(self) -> float:
        return float(self.probability.mean()) if len(self.probability) else 0.0


class SpatialQueries(ABC):
    cfg: SpatialConfig

    @abstractmethod
    def occupancy_state(self, points: FloatArr) -> PointState: ...

    def _check_frame(self, frame_id: str) -> None:
        if frame_id != self.cfg.grid.frame_id:
            raise FrameError(f"query in {frame_id!r}; the map frame is {self.cfg.grid.frame_id!r}")

    def occupancy_probability(self, points: FloatArr) -> FloatArr:
        return self.occupancy_state(points).probability

    def occupancy_status(self, points: FloatArr) -> list[KnowledgeStatus]:
        return [KnowledgeStatus(STATUS_NAMES[s]) for s in self.occupancy_state(points).status.tolist()]

    def is_free(
        self, points: FloatArr, unknown_policy: UnknownPolicy = UnknownPolicy.CONSERVATIVE
    ) -> NDArray[np.bool_]:
        """Free = directly observed (or predicted) with p <= p_free. INFERRED cells are never free space.
        UNKNOWN follows ``unknown_policy`` only; the map itself never calls UNKNOWN occupied or free."""
        s = self.occupancy_state(points)
        known_free = np.isin(s.status, (OBSERVED, PREDICTED)) & (s.probability <= self.cfg.occupancy.p_free)
        if unknown_policy is UnknownPolicy.PERMISSIVE:
            return np.asarray(known_free | (s.status == UNKNOWN))
        return np.asarray(known_free)

    def _region_points(self, region: SpatialSupport) -> FloatArr:
        self._check_frame(region.frame_id)
        res = self.cfg.grid.base_voxel_m
        idx = region_indices(np.asarray(region.center_m), np.asarray(region.half_extent_m), res)
        return index_to_center(idx, res)

    def coverage(self, region: SpatialSupport) -> float:
        """Mean cell coverage over the region at base resolution (never-touched cells count 0)."""
        return float(self.occupancy_state(self._region_points(region)).coverage.mean())

    def unknown_fraction(self, region: SpatialSupport) -> float:
        return float((self.occupancy_state(self._region_points(region)).status == UNKNOWN).mean())

    def predicted_visibility(
        self,
        pose: Pose,
        target_region: SpatialSupport,
        sensor: SensorSpec,
        p_unknown_block: float | None = None,
    ) -> VisibilityPrediction:
        """Ray-cast through the BELIEF map from ``pose`` (an estimated/candidate pose) to the region cells.

        Blocking probability per traversed cell: 0 for cells believed free (p <= p_free), p for other
        known cells, ``p_unknown_block`` for UNKNOWN cells (UNKNOWN may block, it is not assumed empty).
        """
        pu = self.cfg.query.unknown_block_probability if p_unknown_block is None else p_unknown_block
        targets = self._region_points(target_region)
        cap = self.cfg.query.max_visibility_targets
        if len(targets) > cap:
            targets = targets[:: int(np.ceil(len(targets) / cap))]
        frame = sensor_frame(pose, sensor, self.cfg)
        fov = sensor_fov(sensor)
        local = (targets - frame.origin) @ frame.rotation
        dist = np.linalg.norm(local, axis=1)
        az = np.arctan2(local[:, 1], local[:, 0])
        el = np.arctan2(local[:, 2], np.hypot(local[:, 0], local[:, 1]))
        in_fov = (local[:, 0] > 0) & (np.abs(az) <= fov.hfov_rad / 2) & (np.abs(el) <= fov.vfov_rad / 2)
        in_fov &= (dist <= fov.max_range_m) & (dist >= fov.min_range_m)
        prob = np.zeros(len(targets))
        unk = np.zeros(len(targets))
        sel = np.nonzero(in_fov)[0]
        if sel.size:
            res = self.cfg.grid.base_voxel_m
            step = 0.5 * res
            n = int(np.ceil(dist[sel].max() / step))
            t = step * (np.arange(n) + 0.5)
            dirs = (targets[sel] - frame.origin) / dist[sel, None]
            pts = frame.origin + t[None, :, None] * dirs[:, None, :]
            valid = t[None, :] < (dist[sel, None] - 0.5 * res)
            codes = points_to_codes(pts.reshape(-1, 3), res).reshape(valid.shape)
            first = np.ones_like(valid)
            first[:, 1:] = codes[:, 1:] != codes[:, :-1]
            keep = valid & first
            state = self.occupancy_state(pts[keep])
            block = np.where(state.probability <= self.cfg.occupancy.p_free, 0.0, state.probability)
            block = np.where(state.status == UNKNOWN, pu, block)
            log_t = np.zeros(keep.shape)
            log_t[keep] = np.log(np.clip(1.0 - block, 1e-12, 1.0))
            unknown = np.zeros(keep.shape)
            unknown[keep] = state.status == UNKNOWN
            prob[sel] = np.exp(log_t.sum(axis=1))
            unk[sel] = unknown.sum(axis=1)
        return VisibilityPrediction(targets, prob, in_fov, unk, pu)
