"""OBSERVABILITY TRUTH: what a sensor at a pose could have observed (T2S-OBS-01). TRUTH PLANE.

V(x, s, p, t) in [0, 1] with a stored factor decomposition (LOS, FOV, range, incidence, environment), and a
categorical reason. The product formula is an EXPERIMENTAL_CANDIDATE, the factors are the contract.

implementation_status: EXPERIMENTAL_CANDIDATE (exact observability estimator is not frozen, ch14)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from conrad.schemas.frames import Pose
from conrad.schemas.world import SensorSpec
from conrad.twins.twin2s.config import Twin2SConfig
from conrad.twins.twin2s.raycast import segment_clear, sensor_world_pose
from conrad.twins.twin2s.sdf import Arr
from conrad.twins.twin2s.world import SpatialWorld


class VisibilityReason(IntEnum):
    VISIBLE = 0
    OCCLUDED = 1
    OUT_OF_FOV = 2
    OUT_OF_RANGE = 3
    POOR_INCIDENCE = 4
    ENVIRONMENT_LIMITED = 5
    SENSOR_LIMITED = 6


@dataclass(frozen=True)
class SensorGeometry:
    hfov_rad: float
    vfov_rad: float
    max_range_m: float
    min_range_m: float = 0.0


def sensor_geometry(spec: SensorSpec) -> SensorGeometry:
    """Viewing frustum of a ranging/imaging sensor. Non-imaging sensors (IMU, pressure) have none."""
    p = spec.parameters
    if "hfov_deg" not in p or "max_range_m" not in p:
        raise ValueError(f"sensor modality {spec.modality} has no viewing geometry")
    hfov = math.radians(float(p["hfov_deg"]))
    if "vfov_deg" in p:
        vfov = math.radians(float(p["vfov_deg"]))
    else:
        vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * float(p["height_px"]) / float(p["width_px"]))
    return SensorGeometry(hfov, vfov, float(p["max_range_m"]), float(p.get("min_range_m", 0.0)))


@dataclass(frozen=True)
class VisibilityResult:
    score: Arr  # (N,) in [0, 1]
    reason: NDArray[np.int8]  # VisibilityReason
    line_of_sight: NDArray[np.bool_]
    in_fov: NDArray[np.bool_]
    in_range: NDArray[np.bool_]
    range_m: Arr
    incidence_cos: Arr  # NaN when no normals were supplied
    range_factor: Arr
    environment_factor: Arr
    view_dirs: Arr  # (N, 3) unit vector surface -> sensor, WORLD

    @property
    def visible(self) -> NDArray[np.bool_]:
        return np.asarray(self.reason == VisibilityReason.VISIBLE)


class VisibilityOracle:
    def __init__(self, world: SpatialWorld, cfg: Twin2SConfig | None = None) -> None:
        self.world = world
        self.cfg = cfg or Twin2SConfig()

    def visibility(
        self,
        sensor: SensorSpec,
        robot_pose: Pose,
        points_m: Arr,
        normals: Arr | None = None,
        turbidity: float = 0.0,
        tolerance_m: float | None = None,
    ) -> VisibilityResult:
        """Visibility of WORLD points (surface samples or voxel centres) from a TRUE robot pose."""
        geo = sensor_geometry(sensor)
        rot, origin = sensor_world_pose(robot_pose, sensor.mount_pose)
        pts = np.atleast_2d(np.asarray(points_m, dtype=np.float64))
        local = (pts - origin) @ rot
        rng = np.linalg.norm(local, axis=1)
        az = np.arctan2(local[:, 1], local[:, 0])
        el = np.arctan2(local[:, 2], np.hypot(local[:, 0], local[:, 1]))
        in_fov = (np.abs(az) <= geo.hfov_rad / 2) & (np.abs(el) <= geo.vfov_rad / 2) & (local[:, 0] > 0)
        in_range = (rng <= geo.max_range_m) & (rng >= geo.min_range_m)
        los = np.zeros(len(pts), dtype=bool)
        cand = np.nonzero(in_fov & in_range)[0]
        if cand.size:
            tol = self.cfg.observed.visibility_tolerance_m if tolerance_m is None else tolerance_m
            los[cand] = segment_clear(self.world, origin, pts[cand], self.cfg.raycast, tol)
        view = (origin[None, :] - pts) / np.maximum(rng, 1e-12)[:, None]
        inc = np.full(len(pts), np.nan) if normals is None else np.einsum("ij,ij->i", view, normals)
        range_factor = np.clip(1.0 - rng / geo.max_range_m, 0.0, 1.0) ** 0.5
        env = np.exp(-self.cfg.water_attenuation_per_m * (1.0 + 4.0 * turbidity) * rng)
        inc_factor = np.ones(len(pts)) if normals is None else np.clip(inc, 0.0, 1.0)
        score = np.where(los, range_factor * env * inc_factor, 0.0)
        reason = np.full(len(pts), VisibilityReason.VISIBLE, dtype=np.int8)
        oc = self.cfg.observed
        reason[score < oc.min_quality] = VisibilityReason.ENVIRONMENT_LIMITED
        if normals is not None:
            reason[inc < oc.min_incidence_cos] = VisibilityReason.POOR_INCIDENCE
        reason[~los] = VisibilityReason.OCCLUDED
        reason[~in_range] = VisibilityReason.OUT_OF_RANGE
        reason[~in_fov] = VisibilityReason.OUT_OF_FOV
        return VisibilityResult(score, reason, los, in_fov, in_range, rng, inc, range_factor, env, view)


def sample_surface(
    world: SpatialWorld, entity_index: int, n: int, rng: np.random.Generator, exposed_only: bool = True
) -> tuple[Arr, Arr]:
    """Deterministic surface samples (points, outward normals) of one entity by SDF projection."""
    ent = world.entities[entity_index]
    lo, hi = ent.primitive.bounds()
    lo, hi = np.maximum(lo, world.bounds_min), np.minimum(hi, world.bounds_max)
    off = world.entity_offset(entity_index)
    p = rng.uniform(lo, hi, size=(max(4 * n, 64), 3)) + off
    for _ in range(12):
        d = world.entity_sdf(entity_index, p)
        g = np.zeros_like(p)
        for k in range(3):
            e = np.zeros(3)
            e[k] = 1e-3
            g[:, k] = world.entity_sdf(entity_index, p + e) - world.entity_sdf(entity_index, p - e)
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        p = p - d[:, None] * g
    ok = np.abs(world.entity_sdf(entity_index, p)) < 2e-3
    ok &= np.all((p >= np.asarray(world.bounds_min)) & (p <= np.asarray(world.bounds_max)), axis=1)
    if exposed_only:
        ok &= world.sdf(p) > -5e-3  # not buried inside another entity
    p = p[ok][:n]
    return p, world.normals(p) if len(p) else np.zeros((0, 3))
