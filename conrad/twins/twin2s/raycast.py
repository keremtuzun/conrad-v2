"""Vectorised sphere tracing against the canonical SDF world. TRUTH PLANE.

Sensor frame (simulation default): +X boresight, +Y left, +Z up.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from conrad.schemas.frames import Pose, quat_to_matrix
from conrad.twins.twin2s.config import RaycastConfig
from conrad.twins.twin2s.sdf import Arr
from conrad.twins.twin2s.world import SpatialWorld


@dataclass(frozen=True)
class RayHits:
    range_m: Arr  # (N,) distance along the ray; inf where nothing was hit inside max range
    hit: NDArray[np.bool_]
    entity_index: NDArray[np.int64]  # -1 where no hit
    points_m: Arr  # (N, 3) WORLD hit points (NaN where no hit)


def sensor_world_pose(robot_pose: Pose, mount_pose: Pose | None) -> tuple[Arr, Arr]:
    """(R_world_from_sensor, t_world) for a robot WORLD pose composed with a ROBOT-frame mount pose."""
    r_wr = quat_to_matrix(robot_pose.orientation_wxyz)
    t_wr = np.asarray(robot_pose.position_m, dtype=np.float64)
    if mount_pose is None:
        return r_wr, t_wr
    r_rs = quat_to_matrix(mount_pose.orientation_wxyz)
    return r_wr @ r_rs, r_wr @ np.asarray(mount_pose.position_m, dtype=np.float64) + t_wr


def pinhole_directions(width: int, height: int, hfov_deg: float) -> Arr:
    """Unit ray directions (H*W, 3) in the sensor frame, row-major; u grows right (-Y), v grows down (-Z)."""
    f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
    u = (np.arange(width) + 0.5 - 0.5 * width) / f
    v = (np.arange(height) + 0.5 - 0.5 * height) / f
    uu, vv = np.meshgrid(u, v)
    d = np.stack([np.ones_like(uu), -uu, -vv], axis=-1).reshape(-1, 3)
    return np.asarray(d / np.linalg.norm(d, axis=1, keepdims=True), dtype=np.float64)


def fan_directions(
    n_beams: int, hfov_deg: float, n_elev: int, vfov_deg: float
) -> tuple[Arr, NDArray[np.int64]]:
    """Sonar fan: directions (n_beams*n_elev, 3) and the beam index of every ray."""
    az = np.radians(np.linspace(0.5 * hfov_deg, -0.5 * hfov_deg, n_beams))
    el = np.radians(np.linspace(-0.5 * vfov_deg, 0.5 * vfov_deg, n_elev)) if n_elev > 1 else np.zeros(1)
    a, e = np.meshgrid(az, el, indexing="ij")
    d = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1).reshape(-1, 3)
    beam = np.repeat(np.arange(n_beams, dtype=np.int64), len(el))
    return np.asarray(d, dtype=np.float64), beam


def sphere_trace(
    world: SpatialWorld, origins: Arr, directions: Arr, max_range_m: float, cfg: RaycastConfig
) -> RayHits:
    """March every ray by the (conservative) SDF value until |sdf| < epsilon or range is exceeded."""
    n = len(directions)
    o = np.broadcast_to(np.asarray(origins, dtype=np.float64), (n, 3))
    t = np.full(n, cfg.start_offset_m)
    hit = np.zeros(n, dtype=bool)
    ent = np.full(n, -1, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    for _ in range(cfg.max_steps):
        ia = np.nonzero(active)[0]
        if ia.size == 0:
            break
        d, idx = world.sdf_with_index(o[ia] + t[ia, None] * directions[ia])
        h = d < cfg.hit_epsilon_m
        hit[ia[h]] = True
        ent[ia[h]] = idx[h]
        adv = ia[~h]
        t[adv] += np.maximum(d[~h], cfg.min_step_m)
        active[ia[h]] = False
        active[adv[t[adv] > max_range_m]] = False
    # Refine each hit on its own entity's SDF, so the range is independent of the marching path
    # (and therefore of any geometry the ray never touched).
    for i in np.unique(ent[hit]):
        sel = np.nonzero(hit & (ent == i))[0]
        for _ in range(cfg.refine_iterations):
            t[sel] += world.entity_sdf(int(i), o[sel] + t[sel, None] * directions[sel])
    hit &= t <= max_range_m
    ent[~hit] = -1
    rng_m = np.where(hit, t, np.inf)
    pts = np.where(hit[:, None], o + t[:, None] * directions, np.nan)
    return RayHits(rng_m, hit, ent, np.asarray(pts, dtype=np.float64))


def segment_clear(
    world: SpatialWorld, origin: Arr, targets: Arr, cfg: RaycastConfig, tolerance_m: float
) -> NDArray[np.bool_]:
    """Line-of-sight oracle: True where the first surface along origin->target is within ``tolerance_m`` of target."""
    delta = np.atleast_2d(targets) - np.asarray(origin, dtype=np.float64)[None, :]
    dist = np.linalg.norm(delta, axis=1)
    dirs = delta / np.maximum(dist, 1e-12)[:, None]
    hits = sphere_trace(world, origin, dirs, float(dist.max()) + tolerance_m, cfg)
    return np.asarray(hits.range_m >= dist - tolerance_m)
