"""Surface coverage of a component by structural readings (BELIEF PLANE; docs/audits/MODEL2T_REPAIR.md it. 3).

Model2T's corrosion / crack quantities are component WORST cases, but a structural reading shows only the part
of the surface the sensor looked at. Coverage is therefore tracked per component on a coarse partition of its
SURVEYED DESIGN surface (mission context, the same geometry association uses; never Twin truth):

* capsule (pipe segment): ``ceil(length / cell_m)`` bins along the axis x ``sectors`` around it;
* sphere / box: the six faces of the dominant outward axis.

A reading covers the cell that contains the closest design-surface point to its MEASURED surface point
(``Evidence.spatial_support``, attached by association). The belief side does not know how large the viewed
area was, so it never credits more than that cell. A reading without a measured point covers nothing on a
component with geometry. Components without design geometry keep the whole-component-view behaviour.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from conrad.domains.technical.config import CoverageConfig


class GeometryError(ValueError):
    pass


@dataclass(frozen=True)
class SurfaceGeometry:
    shape: str
    p0: tuple[float, float, float]
    p1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_m: float = 0.0
    half_extent_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    n_along: int = 1
    sectors: int = 6

    @property
    def n_cells(self) -> int:
        return self.n_along * self.sectors if self.shape == "CAPSULE" else 6

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        a, b = np.asarray(self.p0), np.asarray(self.p1)
        axis = b - a
        length = float(np.linalg.norm(axis))
        d = axis / max(length, 1e-12)
        ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = ref - (ref @ d) * d
        u /= np.linalg.norm(u)
        return d, u, np.cross(d, u), length

    def cell_of(self, point: Iterable[float]) -> int:
        p = np.asarray(tuple(point), dtype=np.float64)
        a = np.asarray(self.p0)
        if self.shape == "CAPSULE":
            d, u, v, length = self._basis()
            t = float(np.clip((p - a) @ d / max(length, 1e-12), 0.0, 1.0 - 1e-12))
            r = p - (a + t * length * d)
            ang = math.atan2(float(r @ v), float(r @ u)) % (2.0 * math.pi)
            sector = min(int(ang / (2.0 * math.pi) * self.sectors), self.sectors - 1)
            return int(t * self.n_along) * self.sectors + sector
        rel = p - a
        if self.shape == "BOX":
            rel = rel / np.maximum(np.asarray(self.half_extent_m), 1e-9)
        k = int(np.argmax(np.abs(rel)))
        return 2 * k + (0 if rel[k] >= 0.0 else 1)


def _vec(v: Any) -> tuple[float, float, float]:
    x = (0.0, 0.0, 0.0) if v is None else tuple(float(c) for c in v)
    if len(x) != 3:
        raise GeometryError("design vectors must have 3 components")
    return (x[0], x[1], x[2])


def geometry_from_context(context: Mapping[str, Any], cfg: CoverageConfig) -> dict[UUID, SurfaceGeometry]:
    """``context['design_geometry']``: surveyed design surfaces keyed by registry id (optional)."""
    raw = context.get("design_geometry") or []
    out: dict[UUID, SurfaceGeometry] = {}
    for item in raw:
        shape = str(item["shape"]).upper()
        if shape not in ("CAPSULE", "SPHERE", "BOX"):
            raise GeometryError(f"unknown design shape {shape!r}")
        p0, p1, he = (_vec(item.get(k)) for k in ("p0_m", "p1_m", "half_extent_m"))
        length = math.dist(p0, p1) if shape == "CAPSULE" else 0.0
        out[UUID(str(item["registry_id"]))] = SurfaceGeometry(
            shape=shape,
            p0=p0,
            p1=p1,
            radius_m=float(item.get("radius_m", 0.0)),
            half_extent_m=he,
            n_along=max(1, math.ceil(length / cfg.cell_m)) if shape == "CAPSULE" else 1,
            sectors=cfg.sectors if shape == "CAPSULE" else 6,
        )
    return out
