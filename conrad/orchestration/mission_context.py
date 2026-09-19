"""Mission context handed to the deployment side: asset registry, design geometry, mission spec, sensors.

DEPLOYMENT PLANE. Registry IDs are asset-registry identities issued for the mission; they are never
simulator world-entity IDs. Design geometry is what an operator's integrity database would hold (survey
noise included); hidden condition is never part of it.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np
from pydantic import Field

from conrad.schemas.base import ConradModel
from conrad.schemas.frames import WORLD, Pose, SpatialSupport
from conrad.schemas.world import MissionSpec, SensorSpec

Vec3 = tuple[float, float, float]


class DesignComponent(ConradModel):
    """Surveyed design geometry of one registry component (WORLD frame)."""

    registry_id: UUID
    component_type: str
    shape: str = Field(pattern="^(CAPSULE|SPHERE|BOX)$")
    p0_m: Vec3
    p1_m: Vec3 = (0.0, 0.0, 0.0)
    radius_m: float = 0.0
    half_extent_m: Vec3 = (0.0, 0.0, 0.0)
    survey_sigma_m: float = Field(default=0.0, ge=0)

    def region(self, margin_m: float = 0.0) -> SpatialSupport:
        lo, hi = self.bounds()
        c = (lo + hi) / 2.0
        h = (hi - lo) / 2.0 + margin_m
        return SpatialSupport(
            frame_id=WORLD,
            center_m=(float(c[0]), float(c[1]), float(c[2])),
            half_extent_m=(float(h[0]), float(h[1]), float(h[2])),
            position_sigma_m=self.survey_sigma_m,
        )

    def inspection_station(self, along_half_m: float, margin_m: float) -> SpatialSupport:
        """Mid-span inspection station of a capsule (its surveyed mid-point +- ``along_half_m``)."""
        if self.shape != "CAPSULE":
            return self.region(margin_m)
        a, b = np.asarray(self.p0_m), np.asarray(self.p1_m)
        c = (a + b) / 2.0
        d = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        cross = self.radius_m + margin_m
        h = np.abs(d) * along_half_m + (1.0 - np.abs(d)) * cross
        h = np.maximum(h, cross * np.array([0.0, 0.0, 1.0]))
        return SpatialSupport(
            frame_id=WORLD,
            center_m=(float(c[0]), float(c[1]), float(c[2])),
            half_extent_m=(float(h[0]), float(h[1]), float(h[2])),
            position_sigma_m=self.survey_sigma_m,
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(self.p0_m)
        if self.shape == "CAPSULE":
            b = np.asarray(self.p1_m)
            return np.minimum(a, b) - self.radius_m, np.maximum(a, b) + self.radius_m
        if self.shape == "SPHERE":
            return a - self.radius_m, a + self.radius_m
        h = np.asarray(self.half_extent_m)
        return a - h, a + h

    def closest_surface_point(self, p: np.ndarray) -> np.ndarray:
        """Closest point on the design surface to ``p`` (the component's 'projected design geometry')."""
        a = np.asarray(self.p0_m)
        if self.shape == "BOX":
            h = np.asarray(self.half_extent_m)
            q = np.clip(p, a - h, a + h)
            if np.any(np.abs(p - a) > h):
                return q
            axis = int(np.argmin(h - np.abs(p - a)))
            q = p.copy()
            q[axis] = a[axis] + np.sign(p[axis] - a[axis] or 1.0) * h[axis]
            return q
        if self.shape == "CAPSULE":
            b = np.asarray(self.p1_m)
            ab = b - a
            t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
            c = a + t * ab
        else:
            c = a
        d = p - c
        n = float(np.linalg.norm(d))
        return c + (d / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])) * self.radius_m

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(pts)
        a = np.asarray(self.p0_m)
        if self.shape == "BOX":
            q = np.abs(p - a) - np.asarray(self.half_extent_m)
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            return np.asarray(outside + np.minimum(np.max(q, axis=1), 0.0))
        if self.shape == "CAPSULE":
            ab = np.asarray(self.p1_m) - a
            t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
            return np.asarray(np.linalg.norm(p - (a + t[:, None] * ab), axis=1) - self.radius_m)
        return np.asarray(np.linalg.norm(p - a, axis=1) - self.radius_m)


class MissionContext(ConradModel):
    """Everything the deployment side is given before the mission starts."""

    mission_id: UUID
    spec: MissionSpec
    asset_registry: dict[str, Any] = Field(description="Model2T registry format, registry IDs only")
    design: tuple[DesignComponent, ...]
    critical_component_ids: tuple[UUID, ...] = Field(
        description="components whose condition the mission needs"
    )
    sensors: tuple[SensorSpec, ...]
    structural_sensor_ids: tuple[UUID, ...]
    launch_pose: Pose
    transit_lane: tuple[Vec3, ...] = Field(min_length=2)
    seabed_z_m: float
    lane_speed_mps: float = Field(default=0.3, gt=0)

    def component(self, registry_id: UUID) -> DesignComponent:
        for c in self.design:
            if c.registry_id == registry_id:
                return c
        raise KeyError(f"registry component {registry_id} not in the design registry")

    def sensor(self, sensor_id: UUID) -> SensorSpec:
        for s in self.sensors:
            if s.sensor_id == sensor_id:
                return s
        raise KeyError(f"sensor {sensor_id} not declared in the mission context")

    def design_distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance to the surveyed design geometry plus the charted seabed plane."""
        p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        d = p[:, 2] - self.seabed_z_m
        for c in self.design:
            d = np.minimum(d, c.signed_distance(p))
        return np.asarray(d)

    def model2e_registry(self) -> list[dict[str, Any]]:
        out = []
        for c in self.design:
            lo, hi = c.bounds()
            centre = (lo + hi) / 2.0
            out.append(
                {
                    "registry_entity_id": c.registry_id,
                    "entity_type": "pipeline_" + c.component_type.lower(),
                    "position_m": [float(v) for v in centre],
                    "radius_m": float(np.max(hi - lo) / 2.0),
                }
            )
        return out

    def model2s_registry(self) -> list[dict[str, Any]]:
        out = []
        for c in self.design:
            r = c.region()
            out.append(
                {
                    "registry_entity_id": c.registry_id,
                    "center_m": r.center_m,
                    "half_extent_m": r.half_extent_m,
                    "semantic_class": c.component_type.lower(),
                }
            )
        return out
