"""Sparse voxel-octree occupancy/SDF export derived from canonical geometry (ch33). TRUTH PLANE.

Base voxel 0.25 m, refinement 0.125 m / 0.0625 m by default (simulation configuration defaults, not
physical-map claims). A cell is split when the surface may pass through it: |sdf(centre)| is below the
cell half-diagonal. This is an evaluation/export product; canonical truth stays the analytic geometry.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.observation import PayloadRef
from conrad.twins.twin2s.config import OctreeConfig
from conrad.twins.twin2s.sdf import V3, Arr
from conrad.twins.twin2s.world import SpatialWorld

_CHILD = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=np.float64)


@dataclass(frozen=True)
class OctreeExport:
    centers_m: Arr  # (N, 3) WORLD frame
    size_m: Arr  # (N,) cell edge
    level: NDArray[np.int8]  # 0 = base
    sdf_m: Arr  # (N,) signed distance at the centre
    occupied: NDArray[np.bool_]  # sdf < 0
    surface: NDArray[np.bool_]  # surface may cross the cell
    entity_index: NDArray[np.int64]  # nearest entity, index into ``entity_ids``
    entity_ids: tuple[UUID, ...]

    def __len__(self) -> int:
        return len(self.size_m)

    def store(self, store: ObjectStore) -> dict[str, PayloadRef]:
        return {
            "centers_m": store.put_array(self.centers_m),
            "size_m": store.put_array(self.size_m),
            "level": store.put_array(self.level),
            "sdf_m": store.put_array(self.sdf_m),
            "occupied": store.put_array(self.occupied),
            "surface": store.put_array(self.surface),
            "entity_index": store.put_array(self.entity_index),
        }


def export_octree(
    world: SpatialWorld,
    cfg: OctreeConfig,
    bounds_min: V3 | None = None,
    bounds_max: V3 | None = None,
    include_free: bool = False,
) -> OctreeExport:
    """Leaves of the sparse octree. ``include_free=False`` keeps only occupied or surface cells."""
    lo = np.asarray(bounds_min if bounds_min is not None else world.bounds_min, dtype=np.float64)
    hi = np.asarray(bounds_max if bounds_max is not None else world.bounds_max, dtype=np.float64)
    sizes = (cfg.base_voxel_m, *cfg.refinement_voxels_m)
    for a, b in itertools.pairwise(sizes):
        if not math.isclose(a, 2.0 * b):
            raise ValueError("octree refinement levels must halve the parent voxel")
    n = np.maximum(np.ceil((hi - lo) / cfg.base_voxel_m - 1e-9).astype(int), 1)
    if int(n.prod()) > cfg.max_leaf_cells:
        raise ValueError(f"octree base grid {tuple(n)} exceeds max_leaf_cells={cfg.max_leaf_cells}")
    axes = [lo[k] + (np.arange(n[k]) + 0.5) * cfg.base_voxel_m for k in range(3)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    out: list[tuple[Arr, float, int, Arr, NDArray[np.bool_], NDArray[np.int64]]] = []
    for level, size in enumerate(sizes):
        d, idx = world.sdf_with_index(centers)
        mixed = np.abs(d) < 0.5 * math.sqrt(3.0) * size
        last = level == len(sizes) - 1
        leaf = np.ones(len(centers), dtype=bool) if last else ~mixed
        keep = leaf if include_free else leaf & ((d < 0.0) | mixed)
        out.append((centers[keep], size, level, d[keep], mixed[keep], idx[keep]))
        if last:
            break
        parents = centers[mixed]
        centers = (parents[:, None, :] + 0.25 * size * _CHILD[None, :, :]).reshape(-1, 3)
        if len(centers) > cfg.max_leaf_cells:
            raise ValueError("octree refinement exceeds max_leaf_cells")
    sdf = np.concatenate([o[3] for o in out])
    return OctreeExport(
        centers_m=np.concatenate([o[0] for o in out]),
        size_m=np.concatenate([np.full(len(o[0]), o[1]) for o in out]),
        level=np.concatenate([np.full(len(o[0]), o[2], dtype=np.int8) for o in out]),
        sdf_m=sdf,
        occupied=sdf < 0.0,
        surface=np.concatenate([o[4] for o in out]),
        entity_index=np.concatenate([o[5] for o in out]),
        entity_ids=tuple(e.entity_id for e in world.entities),
    )
