"""Common design-surface coordinates; this module contains no Twin or belief state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from conrad.schemas.structural_support import CapsuleSurfaceSupport


def capsule_basis(
    axis_start_m: NDArray[np.float64], axis_end_m: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Stable right-handed axial and angular basis for a surveyed capsule."""
    a, b = np.asarray(axis_start_m, dtype=np.float64), np.asarray(axis_end_m, dtype=np.float64)
    d = b - a
    length = float(np.linalg.norm(d))
    if a.shape != (3,) or b.shape != (3,) or length <= 0:
        raise ValueError("invalid capsule axis")
    d /= length
    u = np.cross(d, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(u) < 1e-9:
        u = np.cross(d, np.array([0.0, 1.0, 0.0]))
    u /= np.linalg.norm(u)
    return d, u, np.cross(d, u)


def surface_point(
    axis_start_m: NDArray[np.float64],
    axis_end_m: NDArray[np.float64],
    radius_m: float,
    axial_m: float,
    angle_rad: float,
) -> NDArray[np.float64]:
    if radius_m <= 0:
        raise ValueError("invalid capsule radius")
    d, u, v = capsule_basis(axis_start_m, axis_end_m)
    return (
        np.asarray(axis_start_m)
        + axial_m * d
        + radius_m * (math.cos(angle_rad) * u + math.sin(angle_rad) * v)
    )


def surface_coordinates(
    axis_start_m: NDArray[np.float64],
    axis_end_m: NDArray[np.float64],
    point_m: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Axial position, wrapped angle, and radial distance of a WORLD point."""
    a = np.asarray(axis_start_m, dtype=np.float64)
    p = np.asarray(point_m, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError("invalid surface point")
    d, u, v = capsule_basis(a, axis_end_m)
    q = p - a
    x = float(q @ d)
    radial = q - x * d
    return x, math.atan2(float(radial @ v), float(radial @ u)) % (2 * math.pi), float(np.linalg.norm(radial))


@dataclass(frozen=True)
class SurfaceRect:
    x0: float
    x1: float
    a0: float
    a1: float

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.a1 - self.a0)

    def intersection(self, other: SurfaceRect) -> SurfaceRect:
        return SurfaceRect(
            max(self.x0, other.x0), min(self.x1, other.x1), max(self.a0, other.a0), min(self.a1, other.a1)
        )

    def contains(self, other: SurfaceRect, eps: float = 1e-12) -> bool:
        return (
            self.x0 <= other.x0 + eps
            and self.x1 >= other.x1 - eps
            and self.a0 <= other.a0 + eps
            and self.a1 >= other.a1 - eps
        )


@dataclass(frozen=True)
class CapsuleSurfaceGrid:
    length_m: float
    radius_m: float
    axial_cells: int
    sectors: int

    def __post_init__(self) -> None:
        if self.length_m <= 0 or self.radius_m <= 0 or self.axial_cells <= 0 or self.sectors <= 0:
            raise ValueError("capsule grid dimensions must be positive")

    @property
    def n_cells(self) -> int:
        return self.axial_cells * self.sectors

    def cell(self, index: int) -> SurfaceRect:
        if not 0 <= index < self.n_cells:
            raise IndexError(index)
        i, j = divmod(index, self.sectors)
        dx, da = self.length_m / self.axial_cells, 2 * math.pi / self.sectors
        return SurfaceRect(i * dx, (i + 1) * dx, j * da, (j + 1) * da)

    def support_rect(self, support: CapsuleSurfaceSupport) -> SurfaceRect:
        if support.axial_end_m > self.length_m + 1e-12:
            raise ValueError("support extends beyond design surface")
        return SurfaceRect(
            support.axial_start_m, support.axial_end_m, support.angle_start_rad, support.angle_end_rad
        )


def union_area(rects: list[SurfaceRect]) -> float:
    """Exact union area of axis-aligned UV rectangles, used only for coverage."""
    xs = sorted({x for r in rects for x in (r.x0, r.x1)})
    area = 0.0
    for lo, hi in pairwise(xs):
        spans = sorted((r.a0, r.a1) for r in rects if r.x0 < hi and r.x1 > lo and r.area > 0)
        if not spans:
            continue
        start, end = spans[0]
        height = 0.0
        for a, b in spans[1:]:
            if a > end:
                height += end - start
                start, end = a, b
            else:
                end = max(end, b)
        height += end - start
        area += (hi - lo) * height
    return area
