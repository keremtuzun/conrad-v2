"""Pose/scene-derived capsule support. Geometry only; no structural truth reads.

The visibility oracle must apply sensor FOV, range, incidence and scene
occlusion at the supplied robot pose. A cell is admitted only when its centre
and four interior probes are visible. This is a finite sampling approximation;
physical and Unity parity claims require separate validation.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from conrad.schemas.capsule_surface import CapsuleSurfaceGrid
from conrad.schemas.structural_sensor import StructuralSensorModelV2
from conrad.schemas.structural_support import CapsuleSurfaceSupport

Visibility = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.bool_]]


def visible_capsule_supports(
    grid: CapsuleSurfaceGrid,
    axis_start_m: NDArray[np.float64],
    axis_end_m: NDArray[np.float64],
    model: StructuralSensorModelV2,
    visible: Visibility,
    *,
    frame_id: str,
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
    d = axis / length
    u = np.cross(d, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(u) < 1e-9:
        u = np.cross(d, np.array([0.0, 1.0, 0.0]))
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
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
    admitted = [bool(row.all()) for row in mask.reshape(len(rects), len(fractions))]
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
