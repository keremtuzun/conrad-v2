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
from conrad.schemas.structural_sensor import StructuralSensorModelV2
from conrad.schemas.structural_support import CapsuleSurfaceSupport
from conrad.schemas.world import SensorSpec
from conrad.twins.twin2s.config import Twin2SConfig
from conrad.twins.twin2s.visibility import sensor_geometry
from conrad.twins.twin2s.world import SpatialWorld

Visibility = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.bool_]]
Certificate = Callable[[float, float, float, float], bool]


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
    if qa <= 1e-12 or disc < 0:
        return ()
    roots = sorted(((-qb - math.sqrt(disc)) / (2 * qa), (-qb + math.sqrt(disc)) / (2 * qa)))
    hit = next(
        (
            (t, origin + t * forward)
            for t in roots
            if t > 0 and grid.length_m >= float((origin + t * forward - a) @ d) >= 0.0
        ),
        None,
    )
    if hit is None or not model.range_min_m <= hit[0] <= model.range_max_m:
        return ()
    point = hit[1]
    x_center = float((point - a) @ d)
    normal = (point - a - x_center * d) / grid.radius_m
    angle_center = math.atan2(float(normal @ v), float(normal @ u)) % (2 * math.pi)
    half_angle = model.footprint_height_m / (2 * grid.radius_m)
    all_visible = visible_capsule_supports(
        grid, a, axis_end_m, model, visible, frame_id=frame_id, certify=certify
    )
    return tuple(
        support
        for support in all_visible
        if support.axial_start_m >= x_center - model.footprint_width_m / 2 - 1e-12
        and support.axial_end_m <= x_center + model.footprint_width_m / 2 + 1e-12
        and abs(
            ((support.angle_start_rad + support.angle_end_rad) / 2 - angle_center + math.pi) % (2 * math.pi)
            - math.pi
        )
        + (support.angle_end_rad - support.angle_start_rad) / 2
        <= half_angle + 1e-12
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
                    rects.append((x0, x1, t0, t1))
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
    a = np.asarray(axis_start_m, dtype=np.float64)
    origin = np.asarray(sensor_origin_m, dtype=np.float64)
    rot = np.asarray(sensor_rotation_world, dtype=np.float64)
    d, u, v = capsule_basis(a, axis_end_m)
    geo = sensor_geometry(sensor)
    tan_h = math.tan(geo.hfov_rad / 2)
    tan_v = math.tan(geo.vfov_rad / 2)
    active_other = [i for i, e in enumerate(world.entities) if i != target_index and e.active]
    fractions = np.linspace(0.0, 1.0, ray_intervals + 1)

    def certify(x0: float, x1: float, t0: float, t1: float) -> bool:
        xc, tc = (x0 + x1) / 2, (t0 + t1) / 2
        half_angle = (t1 - t0) / 2
        displacement = math.hypot((x1 - x0) / 2, radius_m * half_angle)
        n = math.cos(tc) * u + math.sin(tc) * v
        point = a + xc * d + radius_m * n
        delta = point - origin
        distance = float(np.linalg.norm(delta))
        if distance <= displacement or distance - displacement < geo.min_range_m:
            return False
        if distance + displacement > geo.max_range_m:
            return False
        local = rot.T @ delta
        forward_min = local[0] - displacement
        if forward_min <= 0:
            return False
        if abs(local[1]) + displacement > forward_min * tan_h:
            return False
        if abs(local[2]) + displacement > forward_min * tan_v:
            return False
        # Numerator of incidence over the rectangle; normal rotation is
        # bounded by half_angle and every surface point by displacement.
        incidence_num = float((origin - point) @ n) - distance * half_angle - displacement
        incidence_min = incidence_num / (distance + displacement)
        if incidence_min < config.observed.min_incidence_cos:
            return False
        far = distance + displacement
        quality_min = (
            math.sqrt(max(0.0, 1.0 - far / geo.max_range_m))
            * math.exp(-config.water_attenuation_per_m * far)
            * incidence_min
        )
        if quality_min < config.observed.min_quality:
            return False
        centre_ray = origin + fractions[:, None] * delta
        half_step = distance / (2 * ray_intervals)
        for i in active_other:
            if float(np.min(world.entity_sdf(i, centre_ray))) <= displacement + half_step:
                return False
        return True

    return certify
