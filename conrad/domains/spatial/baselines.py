"""Model2S baselines with the SAME interface (Model2Child + map queries). BELIEF PLANE.

- :class:`PlainOccupancyGrid` (S-B0): classical log-odds grid, per-ray counting with OctoMap-style clamps,
  no knowledge status (any touched cell reports OBSERVED), no pose covariance, no inference, no refinement.
- :class:`MaxLikelihoodSurfaceFill`: the plain grid plus naive "complete the geometry": every UNKNOWN
  cell within ``fill_radius_cells`` of an occupied cell is reported occupied, every other UNKNOWN cell free,
  both with confident probabilities. It exists to expose the unsupported-confidence failure mode.

implementation_status: EXPERIMENTAL_CANDIDATE (baselines)
"""

from __future__ import annotations

from uuid import UUID

import numpy as np

from conrad.domains.spatial.config import SpatialConfig, plain_grid_config
from conrad.domains.spatial.grid import INFERRED, OBSERVED, UNKNOWN
from conrad.domains.spatial.hierarchy import PointState
from conrad.domains.spatial.keys import FloatArr, IntArr, cube_offsets, decode, encode, points_to_codes
from conrad.domains.spatial.model import Model2S
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.repository import Repository
from conrad.schemas.belief import BeliefMessage
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import TimeStamp


class PlainOccupancyGrid(Model2S):
    def __init__(
        self,
        ids: IdFactory,
        store: ObjectStore,
        cfg: SpatialConfig | None = None,
        repository: Repository | None = None,
        run_id: UUID | None = None,
    ) -> None:
        super().__init__(ids, store, plain_grid_config(cfg), repository, run_id)


class MaxLikelihoodSurfaceFill(PlainOccupancyGrid):
    def __init__(
        self,
        ids: IdFactory,
        store: ObjectStore,
        cfg: SpatialConfig | None = None,
        repository: Repository | None = None,
        run_id: UUID | None = None,
        fill_radius_cells: int = 2,
        fill_occupied_probability: float = 0.97,
        fill_free_probability: float = 0.03,
    ) -> None:
        super().__init__(ids, store, cfg, repository, run_id)
        self.model_version = "baseline-ml-surface-fill-v1"
        self.fill_radius_cells = fill_radius_cells
        self.p_fill_occ = fill_occupied_probability
        self.p_fill_free = fill_free_probability
        self._dilated: IntArr | None = None

    def update_beliefs(self, now: TimeStamp) -> list[BeliefMessage]:
        self._dilated = None
        return super().update_beliefs(now)

    def _completion(self) -> IntArr:
        if self._dilated is None:
            base = self.map.base
            rows = base.all_rows()
            occ = base.codes[rows][(base.status(rows) == OBSERVED) & (base.probability(rows) >= 0.5)]
            if occ.size == 0:
                self._dilated = np.zeros(0, dtype=np.int64)
            else:
                idx = decode(occ)[:, None, :] + cube_offsets(self.fill_radius_cells)[None, :, :]
                self._dilated = np.unique(encode(idx.reshape(-1, 3)))
        return self._dilated

    def occupancy_state(self, points: FloatArr) -> PointState:
        s = self.map.point_state(points)
        unknown = s.status == UNKNOWN
        if not unknown.any():
            return s
        codes = points_to_codes(np.atleast_2d(points)[unknown], self.cfg.grid.base_voxel_m)
        near = np.isin(codes, self._completion())
        p, st, unc = s.probability.copy(), s.status.copy(), s.uncertainty.copy()
        p[unknown] = np.where(near, self.p_fill_occ, self.p_fill_free)
        st[unknown] = INFERRED
        unc[unknown] = 0.0  # the naive completion asserts its geometry without an uncertainty model
        return PointState(p, st, unc, s.coverage, p * (1 - p), s.level)
