"""Pose/scene-derived capsule support. Geometry only; no structural truth reads.

The legacy five-probe helper is retained for development comparisons. Mission
support requires a conservative rectangle certificate; sampled visibility
alone never establishes healthy coverage over a continuous surface.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from conrad.schemas.capsule_surface import CapsuleSurfaceGrid, capsule_basis
from conrad.schemas.capsule_visibility import capsule_geometry_certificate
from conrad.schemas.structural_sensor import StructuralSensorModelV2
from conrad.schemas.structural_support import CapsuleSurfaceSupport
from conrad.schemas.world import SensorSpec
from conrad.twins.twin2s.config import Twin2SConfig
from conrad.twins.twin2s.visibility import sensor_geometry
from conrad.twins.twin2s.world import SpatialWorld

Visibility = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.bool_]]
Certificate = Callable[[float, float, float, float], bool]
VISIBILITY_CERTIFICATE_VERSION = "capsule-sdf-recursive-v1"


def pose_visible_capsule_supports(
    grid: CapsuleSurfaceGrid,
    axis_start_m: NDArray[np.float64],
    axis_end_m: NDArray[np.float64],
    sensor_origin_m: NDArray[np.float64],
    sensor_forward_world: NDArray[np.float64],
    model: StructuralSensorModelV2,
    visible: Visibility,
    *,
    frame_id: str,
    certify: Certificate | None = None,
) -> tuple[CapsuleSurfaceSupport, ...]:
    """Derive the nominal footprint from the sensor ray, then clip by scene visibility."""
    a = np.asarray(axis_start_m, dtype=np.float64)
    origin = np.asarray(sensor_origin_m, dtype=np.float64)
    forward = np.asarray(sensor_forward_world, dtype=np.float64)
    if origin.shape != (3,) or forward.shape != (3,) or not np.isfinite(forward).all():
        raise ValueError("invalid sensor pose")
    norm = float(np.linalg.norm(forward))
    if norm <= 0:
        raise ValueError("zero sensor forward vector")
    forward /= norm
    d, u, v = capsule_basis(a, axis_end_m)
    q = origin - a
    q_perp = q - float(q @ d) * d
    f_perp = forward - float(forward @ d) * d
    qa = float(f_perp @ f_perp)
    qb = 2.0 * float(q_perp @ f_perp)
    qc = float(q_perp @ q_perp) - grid.radius_m**2
    disc = qb * qb - 4.0 * qa * qc
    hit = None
    if qa > 1e-12 and disc >= 0:
        roots = sorted(((-qb - math.sqrt(disc)) / (2 * qa), (-qb + math.sqrt(disc)) / (2 * qa)))
        hit = next(
            (
                (t, origin + t * forward)
                for t in roots
                if t > 0 and grid.length_m >= float((origin + t * forward - a) @ d) >= 0.0
            ),
            None,
        )
    if hit is not None:
        point = hit[1]
        x_center = float((point - a) @ d)
        normal = (point - a - x_center * d) / grid.radius_m
        distance = hit[0]
    else:
        # A broad-FOV inspection payload can see the pipe while its centre ray
        # passes above it. Centre the candidate footprint on the near surface
        # at the closest ray/axis approach; the visibility oracle and continuous
        # certificate below still decide whether any rectangle earns credit.
        if certify is None:
            return ()
        if float(np.linalg.norm(q_perp)) <= grid.radius_m:
            return ()
        t_closest = max(0.0, -float(q_perp @ f_perp) / qa) if qa > 1e-12 else 0.0
        x_center = float(np.clip((q + t_closest * forward) @ d, 0.0, grid.length_m))
        normal = q_perp / float(np.linalg.norm(q_perp))
        point = a + x_center * d + grid.radius_m * normal
        distance = float(np.linalg.norm(point - origin))
    if not model.range_min_m <= distance <= model.range_max_m:
        return ()
    angle_center = math.atan2(float(normal @ v), float(normal @ u)) % (2 * math.pi)
    half_angle = model.footprint_height_m / (2 * grid.radius_m)

    def within_footprint(x0: float, x1: float, t0: float, t1: float) -> bool:
        return (
            x0 >= x_center - model.footprint_width_m / 2 - 1e-12
            and x1 <= x_center + model.footprint_width_m / 2 + 1e-12
            and abs(((t0 + t1) / 2 - angle_center + math.pi) % (2 * math.pi) - math.pi) + (t1 - t0) / 2
            <= half_angle + 1e-12
        )

    return visible_capsule_supports(
        grid,
        a,
        axis_end_m,
        model,
        visible,
        frame_id=frame_id,
        certify=certify,
        select=within_footprint,
    )


def visible_capsule_supports(
    grid: CapsuleSurfaceGrid,
    axis_start_m: NDArray[np.float64],
    axis_end_m: NDArray[np.float64],
    model: StructuralSensorModelV2,
    visible: Visibility,
    *,
    frame_id: str,
    certify: Certificate | None = None,
    select: Certificate | None = None,
) -> tuple[CapsuleSurfaceSupport, ...]:
    """Return only resolution-sized rectangles admitted by the scene oracle.

    The callback is normally Twin2S visibility at a robot pose and sensor
    orientation. Its inputs are world points and outward surface normals.
    No condition value or Twin2T identifier is accepted by this function.
    """
    a = np.asarray(axis_start_m, dtype=np.float64)
    b = np.asarray(axis_end_m, dtype=np.float64)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if a.shape != (3,) or b.shape != (3,) or abs(length - grid.length_m) > 1e-6:
        raise ValueError("capsule axis must match the surveyed surface grid")
    d, u, v = capsule_basis(a, b)
    rects: list[tuple[float, float, float, float]] = []
    for i in range(grid.axial_cells):
        x_base = grid.length_m * i / grid.axial_cells
        x_limit = grid.length_m * (i + 1) / grid.axial_cells
        nx = math.ceil((x_limit - x_base) / model.axial_resolution_m)
        for j in range(grid.sectors):
            angle_base = 2 * math.pi * j / grid.sectors
            angle_limit = 2 * math.pi * (j + 1) / grid.sectors
            na = math.ceil(grid.radius_m * (angle_limit - angle_base) / model.lateral_resolution_m)
            for ix in range(nx):
                x0 = x_base + ix * model.axial_resolution_m
                x1 = min(x_limit, x0 + model.axial_resolution_m)
                for ia in range(na):
                    t0 = angle_base + ia * model.lateral_resolution_m / grid.radius_m
                    t1 = min(angle_limit, t0 + model.lateral_resolution_m / grid.radius_m)
                    if select is None or select(x0, x1, t0, t1):
                        rects.append((x0, x1, t0, t1))
    if not rects:
        return ()
    # A corner exactly on a seam or cap can be ambiguous for ray casting;
    # probe just inside it while retaining the full declared support bounds.
    fractions = ((0.5, 0.5), (0.05, 0.05), (0.05, 0.95), (0.95, 0.05), (0.95, 0.95))
    pts = []
    normals = []
    for x0, x1, t0, t1 in rects:
        for fx, ft in fractions:
            x = x0 + fx * (x1 - x0)
            angle = t0 + ft * (t1 - t0)
            normal = math.cos(angle) * u + math.sin(angle) * v
            pts.append(a + x * d + grid.radius_m * normal)
            normals.append(normal)
    mask = np.asarray(visible(np.asarray(pts), np.asarray(normals)), dtype=bool)
    if mask.shape != (len(pts),):
        raise ValueError("visibility oracle returned wrong shape")
    admitted = [
        bool(row.all()) and (certify is None or certify(*rect))
        for rect, row in zip(rects, mask.reshape(len(rects), len(fractions)), strict=True)
    ]
    clipped = not all(admitted)
    if model.position_uncertainty_m is None or model.footprint_uncertainty_m is None:
        axial_uncertainty = None
    else:
        axial_uncertainty = model.position_uncertainty_m + model.footprint_uncertainty_m
    angular_uncertainty = (
        None
        if axial_uncertainty is None or model.orientation_uncertainty_rad is None
        else model.orientation_uncertainty_rad + axial_uncertainty / grid.radius_m
    )
    return tuple(
        CapsuleSurfaceSupport(
            sensor_model_version=model.version,
            sensor_config_digest=model.digest,
            frame_id=frame_id,
            axial_start_m=x0,
            axial_end_m=x1,
            angle_start_rad=t0,
            angle_end_rad=t1,
            axial_uncertainty_m=axial_uncertainty,
            angular_uncertainty_rad=angular_uncertainty,
            occlusion_clipped=clipped,
            aggregation_kernel=model.aggregation_kernel,
        )
        for (x0, x1, t0, t1), keep in zip(rects, admitted, strict=True)
        if keep
    )


def capsule_visibility_certificate(
    world: SpatialWorld,
    target_index: int,
    axis_start_m: NDArray[np.float64],
    axis_end_m: NDArray[np.float64],
    radius_m: float,
    sensor_origin_m: NDArray[np.float64],
    sensor_rotation_world: NDArray[np.float64],
    sensor: SensorSpec,
    config: Twin2SConfig,
    *,
    ray_intervals: int = 64,
) -> Certificate:
    """Certify continuous visibility of each surface rectangle using SDF bounds.

    Each active primitive's SDF is 1-Lipschitz. A ray to any point in a
    rectangle differs from the centre ray by at most the surface half-diagonal.
    Sampling the centre ray in equal intervals and subtracting both this bound
    and half the ray step certifies clearance everywhere. FOV, range, incidence,
    and quality use the same conservative surface-displacement bound. The
    target cylinder is convex; positive incidence throughout the rectangle
    excludes self-occlusion. A failed certificate simply credits no support.
    """
    if ray_intervals < 2 or radius_m <= 0:
        raise ValueError("invalid certificate resolution or radius")
    origin = np.asarray(sensor_origin_m, dtype=np.float64)
    rot = np.asarray(sensor_rotation_world, dtype=np.float64)
    geo = sensor_geometry(sensor)
    active_other = [i for i, e in enumerate(world.entities) if i != target_index and e.active]
    fractions = np.linspace(0.0, 1.0, ray_intervals + 1)

    def clear(point: NDArray[np.float64], displacement: float, distance: float) -> bool:
        delta = point - origin
        centre_ray = origin + fractions[:, None] * delta
        half_step = distance / (2 * ray_intervals)
        for i in active_other:
            if float(np.min(world.entity_sdf(i, centre_ray))) <= displacement + half_step:
                return False
        return True

    return capsule_geometry_certificate(
        axis_start_m,
        axis_end_m,
        radius_m,
        origin,
        rot,
        min_range_m=geo.min_range_m,
        max_range_m=geo.max_range_m,
        hfov_rad=geo.hfov_rad,
        vfov_rad=geo.vfov_rad,
        min_incidence_cos=config.observed.min_incidence_cos,
        min_quality=config.observed.min_quality,
        water_attenuation_per_m=config.water_attenuation_per_m,
        clearance=clear,
    )
