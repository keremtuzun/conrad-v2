"""Fixed-resolution field grids (ch33: regional 64x64x8, local 32x32x16). TRUTH PLANE.

Axis order is (x, y, z) with z pointing up in the grid's named frame; values sit at cell centres.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from conrad.schemas.frames import FrameError
from conrad.twins.twin2e.config import GridConfig


@dataclass(frozen=True)
class FieldGrid:
    name: str
    frame_id: str
    origin_m: tuple[float, float, float]
    spacing_m: tuple[float, float, float]
    shape: tuple[int, int, int]

    @classmethod
    def from_config(cls, name: str, cfg: GridConfig) -> FieldGrid:
        if any(s < 2 for s in cfg.shape):
            raise ValueError(f"grid {name} needs at least 2 cells per axis, got {cfg.shape}")
        if any(d <= 0 for d in cfg.spacing_m):
            raise ValueError(f"grid {name} spacing must be positive")
        return cls(name, cfg.frame_id, cfg.origin_m, cfg.spacing_m, cfg.shape)

    @property
    def cell_volume_m3(self) -> float:
        return float(np.prod(self.spacing_m))

    @property
    def extent_max_m(self) -> tuple[float, float, float]:
        o, d, n = self.origin_m, self.spacing_m, self.shape
        return (o[0] + d[0] * n[0], o[1] + d[1] * n[1], o[2] + d[2] * n[2])

    def axis_centres(self, axis: int) -> np.ndarray:
        return self.origin_m[axis] + (np.arange(self.shape[axis]) + 0.5) * self.spacing_m[axis]

    def centres(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = (self.axis_centres(a) for a in range(3))
        gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
        return gx, gy, gz

    def contains(self, xyz: np.ndarray | tuple[float, float, float]) -> bool:
        p = np.asarray(xyz, dtype=np.float64)
        hi = self.extent_max_m
        return bool(all(self.origin_m[a] <= p[a] <= hi[a] for a in range(3)))

    def contains_grid(self, other: FieldGrid) -> bool:
        return self.contains(other.origin_m) and self.contains(other.extent_max_m)

    def check_frame(self, frame_id: str) -> None:
        if frame_id != self.frame_id:
            raise FrameError(f"position is in {frame_id!r} but grid {self.name} is in {self.frame_id!r}")

    def cell_index(self, xyz: np.ndarray | tuple[float, float, float]) -> tuple[int, int, int]:
        p = np.asarray(xyz, dtype=np.float64)
        idx = [
            int(np.clip(np.floor((p[a] - self.origin_m[a]) / self.spacing_m[a]), 0, self.shape[a] - 1))
            for a in range(3)
        ]
        return (idx[0], idx[1], idx[2])

    def fractional_index(self, pts: np.ndarray) -> np.ndarray:
        """(3, N) fractional cell-centre coordinates of world points ``pts`` (N, 3)."""
        o = np.asarray(self.origin_m)[:, None]
        d = np.asarray(self.spacing_m)[:, None]
        return (np.asarray(pts, dtype=np.float64).T - o) / d - 0.5

    def sample(self, values: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Trilinear sample of a scalar (nx,ny,nz) field at points (N,3); edges clamp."""
        coords = self.fractional_index(np.atleast_2d(pts))
        return np.asarray(map_coordinates(values, coords, order=1, mode="nearest"), dtype=np.float64)

    def resample_from(self, source: FieldGrid, values: np.ndarray) -> np.ndarray:
        """Interpolate a scalar field defined on ``source`` onto this grid's cell centres."""
        gx, gy, gz = self.centres()
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        return source.sample(values, pts).reshape(self.shape)

    def owning_cells(self, finer: FieldGrid) -> np.ndarray:
        """Flat index in this grid of the cell containing each cell centre of ``finer``."""
        gx, gy, gz = finer.centres()
        idx = []
        for a, g in enumerate((gx, gy, gz)):
            i = np.floor((g - self.origin_m[a]) / self.spacing_m[a]).astype(np.int64)
            idx.append(np.clip(i, 0, self.shape[a] - 1))
        return np.ravel_multi_index((idx[0], idx[1], idx[2]), self.shape).ravel()

    def geometry(self) -> dict[str, object]:
        return {
            "name": self.name,
            "frame_id": self.frame_id,
            "origin_m": list(self.origin_m),
            "spacing_m": list(self.spacing_m),
            "shape": list(self.shape),
            "axis_order": "x,y,z (z up), cell-centred",
        }
