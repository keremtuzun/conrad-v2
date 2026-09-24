"""Conservative capsule support registration for a hard-bounded design survey.

The simulated sensor reports coordinates of its measured physical surface
footprint. This module bounds their difference from the surveyed design
coordinates using only the declared endpoint error bound and design geometry.
An unbounded Gaussian standard deviation is never treated as such a bound.
"""

from __future__ import annotations

import math

import numpy as np

REGISTRATION_VERSION = "exact-bounded-or-unregistered-v2"


def capsule_registration_uncertainty(
    design_start_m: tuple[float, float, float],
    design_end_m: tuple[float, float, float],
    radius_m: float,
    endpoint_bound_m: float,
) -> tuple[float, float] | None:
    """Return guaranteed axial and angular coordinate-error bounds.

    Each surveyed endpoint lies within ``endpoint_bound_m`` of its physical
    endpoint. The angular bound includes axis-direction, radial-projection,
    and capsule-basis errors. Near-vertical or ambiguous geometries earn no
    certificate because ``capsule_basis`` changes reference branch there.
    """
    if radius_m <= 0 or endpoint_bound_m <= 0:
        raise ValueError("registration radius and bound must be positive")
    a = np.asarray(design_start_m, dtype=np.float64)
    b = np.asarray(design_end_m, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid design axis")
    length = float(np.linalg.norm(b - a))
    error = endpoint_bound_m
    if length <= 2 * error:
        return None
    design_direction = (b - a) / length
    direction_error = 4 * error / (length - 2 * error)
    horizontal = float(np.linalg.norm(design_direction[:2]))
    if horizontal <= direction_error + 0.5:
        return None
    max_true_length = length + 2 * error
    axial_error = error + (max_true_length + radius_m) * direction_error
    radial_error = error + max_true_length * direction_error + axial_error
    if radial_error >= radius_m:
        return None
    unit_radial_error = 2 * radial_error / (radius_m - radial_error)
    angular_basis_error = 2 * direction_error / (horizontal - direction_error)
    other_basis_error = direction_error + angular_basis_error
    pair_error = math.sqrt(2) * (unit_radial_error + other_basis_error)
    if pair_error >= 2:
        return None
    angular_error = 2 * math.asin(pair_error / 2)
    # The analytic bounds are intentionally loose; round their result outward.
    return math.nextafter(axial_error, math.inf), math.nextafter(angular_error, math.inf)
