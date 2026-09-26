"""Truth-free continuous geometry certificate for capsule surface support.

The runtime sensor adds a scene-clearance callback; belief-side planning can
use the same field-of-view, incidence, and quality bound without scene truth.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from conrad.schemas.capsule_surface import capsule_basis

Certificate = Callable[[float, float, float, float], bool]
Clearance = Callable[[NDArray[np.float64], float, float], bool]
MAX_SUBDIVISION_DEPTH = 5


def capsule_geometry_certificate(
    axis_start_m: NDArray[np.float64],
    axis_end_m: NDArray[np.float64],
    radius_m: float,
    sensor_origin_m: NDArray[np.float64],
    sensor_rotation_world: NDArray[np.float64],
    *,
    min_range_m: float,
    max_range_m: float,
    hfov_rad: float,
    vfov_rad: float,
    min_incidence_cos: float,
    min_quality: float,
    water_attenuation_per_m: float,
    clearance: Clearance | None = None,
) -> Certificate:
    """Certify every point of a rectangle by bounded subdivision.

    A scene callback may further reject rectangles but can never relax the
    geometric certificate. An unresolved edge earns no support.
    """
    if radius_m <= 0 or max_range_m <= min_range_m:
        raise ValueError("invalid capsule visibility geometry")
    a = np.asarray(axis_start_m, dtype=np.float64)
    origin = np.asarray(sensor_origin_m, dtype=np.float64)
    rot = np.asarray(sensor_rotation_world, dtype=np.float64)
    d, u, v = capsule_basis(a, axis_end_m)
    tan_h, tan_v = math.tan(hfov_rad / 2), math.tan(vfov_rad / 2)

    def certify_box(x0: float, x1: float, t0: float, t1: float) -> bool:
        xc, tc = (x0 + x1) / 2, (t0 + t1) / 2
        half_angle = (t1 - t0) / 2
        displacement = math.hypot((x1 - x0) / 2, radius_m * half_angle)
        n = math.cos(tc) * u + math.sin(tc) * v
        point = a + xc * d + radius_m * n
        delta = point - origin
        distance = float(np.linalg.norm(delta))
        if distance <= displacement or distance - displacement < min_range_m:
            return False
        if distance + displacement > max_range_m:
            return False
        local = rot.T @ delta
        forward_min = local[0] - displacement
        if forward_min <= 0:
            return False
        if abs(local[1]) + displacement > forward_min * tan_h:
            return False
        if abs(local[2]) + displacement > forward_min * tan_v:
            return False
        incidence_num = float((origin - point) @ n) - distance * half_angle - displacement
        incidence_min = incidence_num / (distance + displacement)
        if incidence_min < min_incidence_cos:
            return False
        far = distance + displacement
        quality_min = (
            math.sqrt(max(0.0, 1.0 - far / max_range_m))
            * math.exp(-water_attenuation_per_m * far)
            * incidence_min
        )
        if quality_min < min_quality:
            return False
        return clearance(point, displacement, distance) if clearance is not None else True

    def certify(x0: float, x1: float, t0: float, t1: float) -> bool:
        def cover(ax0: float, ax1: float, at0: float, at1: float, depth: int) -> bool:
            if certify_box(ax0, ax1, at0, at1):
                return True
            if depth == 0:
                return False
            xm, tm = (ax0 + ax1) / 2, (at0 + at1) / 2
            return all(
                cover(*child, depth - 1)
                for child in (
                    (ax0, xm, at0, tm),
                    (ax0, xm, tm, at1),
                    (xm, ax1, at0, tm),
                    (xm, ax1, tm, at1),
                )
            )

        return cover(x0, x1, t0, t1, MAX_SUBDIVISION_DEPTH)

    return certify
