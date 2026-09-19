"""Unregistered obstacles added to a built MissionWorld (I5 blocked-route scenario). TRUTH PLANE. SYNTHETIC_ONLY.

The obstacle is a Twin2S entity only: it is not in the asset registry, not in the mission context and has no
Twin2T / Twin2E state, so the runtime can know about it only through its geometric sensors. It is appended to
the live ``SpatialWorld`` that the kernel (collision SDF) and the sensing suite already read.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from conrad.schemas.ids import IdFactory
from conrad.sim.mission.options import LaneObstacleOptions
from conrad.twins.twin2s.sdf import Box
from conrad.twins.twin2s.world import SpatialEntity, class_index

if TYPE_CHECKING:
    from conrad.sim.mission.world import MissionWorld


def lane_point(lane: tuple[tuple[float, float, float], ...], fraction: float) -> np.ndarray:
    """Point at ``fraction`` of the lane polyline's length."""
    pts = np.asarray(lane, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    target = float(fraction) * float(seg.sum())
    for i, length in enumerate(seg):
        if target <= length or i == len(seg) - 1:
            return np.asarray(pts[i] + (pts[i + 1] - pts[i]) * min(target / max(length, 1e-12), 1.0))
        target -= length
    return np.asarray(pts[-1])


def add_lane_obstacle(world: MissionWorld, opts: LaneObstacleOptions) -> dict[str, Any]:
    """Append one unregistered box on the transit lane; returns its truth record (evaluation only)."""
    centre = lane_point(world.context.transit_lane, opts.lane_fraction)
    entity = SpatialEntity(
        entity_id=IdFactory(world.seed).child("lane_obstacle").new(),
        semantic_class="rock",
        primitive=Box(
            center=(float(centre[0]), float(centre[1]), float(centre[2])), half_extents=opts.half_extent_m
        ),
        material_id=opts.material_id,
    )
    class_index(entity.semantic_class)
    world.t2s.world.entities.append(entity)
    record = {
        "entity_id": str(entity.entity_id),
        "center_m": [float(v) for v in centre],
        "half_extent_m": list(opts.half_extent_m),
        "lane_fraction": opts.lane_fraction,
    }
    world.recorder.meta["lane_obstacle"] = record
    return record
