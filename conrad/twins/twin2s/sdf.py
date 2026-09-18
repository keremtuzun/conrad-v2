"""Canonical analytic geometry with signed-distance functions (T2S-WORLD-01). TRUTH PLANE.

Every primitive is independent of any Model2S representation. ``sdf`` is negative inside, positive
outside. Sphere, capsule, poly-capsule, capped cylinder, oriented box and torus are exact distances.
Ellipsoid and heightfield terrain are sign-exact and conservative (1-Lipschitz lower bounds), which is
what sphere tracing requires; this is declared through ``exact``.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from conrad.schemas.frames import quat_to_matrix

Arr = NDArray[np.float64]
V3 = tuple[float, float, float]


def _v(x: Any) -> Arr:
    return np.asarray(x, dtype=np.float64)


def _t3(x: Any) -> V3:
    a = _v(x)
    return (float(a[0]), float(a[1]), float(a[2]))


class Primitive(ABC):
    kind: ClassVar[str]
    exact: ClassVar[bool] = True

    @abstractmethod
    def sdf(self, p: Arr) -> Arr:
        """Signed distance in metres for points ``p`` of shape (N, 3) in the WORLD frame."""

    @abstractmethod
    def bounds(self) -> tuple[Arr, Arr]: ...

    @abstractmethod
    def params(self) -> dict[str, Any]: ...

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.params()}


@dataclass(frozen=True)
class Sphere(Primitive):
    center: V3
    radius: float
    kind: ClassVar[str] = "sphere"

    def sdf(self, p: Arr) -> Arr:
        return np.linalg.norm(p - _v(self.center), axis=1) - self.radius

    def bounds(self) -> tuple[Arr, Arr]:
        c = _v(self.center)
        return c - self.radius, c + self.radius

    def params(self) -> dict[str, Any]:
        return {"center": list(self.center), "radius": self.radius}


@dataclass(frozen=True)
class Ellipsoid(Primitive):
    """Axis-aligned-in-body ellipsoid, yaw-rotated. Conservative bound (|p/r|-1)*min(r)."""

    center: V3
    radii: V3
    rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    kind: ClassVar[str] = "ellipsoid"
    exact: ClassVar[bool] = False

    def sdf(self, p: Arr) -> Arr:
        q = (p - _v(self.center)) @ quat_to_matrix(self.rotation_wxyz)
        r = _v(self.radii)
        return (np.linalg.norm(q / r, axis=1) - 1.0) * float(r.min())

    def bounds(self) -> tuple[Arr, Arr]:
        c, m = _v(self.center), float(max(self.radii))
        return c - m, c + m

    def params(self) -> dict[str, Any]:
        return {
            "center": list(self.center),
            "radii": list(self.radii),
            "rotation_wxyz": list(self.rotation_wxyz),
        }


def _segment_distance(p: Arr, a: Arr, b: Arr) -> Arr:
    ab = b - a
    denom = float(ab @ ab)
    h = np.clip(((p - a) @ ab) / denom, 0.0, 1.0) if denom > 0 else np.zeros(len(p))
    return np.asarray(np.linalg.norm(p - a - h[:, None] * ab, axis=1), dtype=np.float64)


@dataclass(frozen=True)
class Capsule(Primitive):
    a: V3
    b: V3
    radius: float
    kind: ClassVar[str] = "capsule"

    def sdf(self, p: Arr) -> Arr:
        return _segment_distance(p, _v(self.a), _v(self.b)) - self.radius

    def bounds(self) -> tuple[Arr, Arr]:
        pts = np.stack([_v(self.a), _v(self.b)])
        return pts.min(0) - self.radius, pts.max(0) + self.radius

    def params(self) -> dict[str, Any]:
        return {"a": list(self.a), "b": list(self.b), "radius": self.radius}


@dataclass(frozen=True)
class PolyCapsule(Primitive):
    """Union of capsules along a polyline: pipe bends and cables."""

    points: tuple[V3, ...]
    radius: float
    kind: ClassVar[str] = "polycapsule"

    def sdf(self, p: Arr) -> Arr:
        pts = _v(self.points)
        d = np.full(len(p), np.inf)
        for i in range(len(pts) - 1):
            d = np.minimum(d, _segment_distance(p, pts[i], pts[i + 1]))
        return d - self.radius

    def bounds(self) -> tuple[Arr, Arr]:
        pts = _v(self.points)
        return pts.min(0) - self.radius, pts.max(0) + self.radius

    def params(self) -> dict[str, Any]:
        return {"points": [list(q) for q in self.points], "radius": self.radius}


@dataclass(frozen=True)
class Cylinder(Primitive):
    """Capped cylinder between ``a`` and ``b`` (exact)."""

    a: V3
    b: V3
    radius: float
    kind: ClassVar[str] = "cylinder"

    def sdf(self, p: Arr) -> Arr:
        a, b = _v(self.a), _v(self.b)
        axis = b - a
        length = float(np.linalg.norm(axis))
        u = axis / length
        rel = p - 0.5 * (a + b)
        y = rel @ u
        radial = np.linalg.norm(rel - y[:, None] * u, axis=1)
        dx, dy = radial - self.radius, np.abs(y) - 0.5 * length
        outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
        return np.asarray(np.minimum(np.maximum(dx, dy), 0.0) + outside, dtype=np.float64)

    def bounds(self) -> tuple[Arr, Arr]:
        pts = np.stack([_v(self.a), _v(self.b)])
        return pts.min(0) - self.radius, pts.max(0) + self.radius

    def params(self) -> dict[str, Any]:
        return {"a": list(self.a), "b": list(self.b), "radius": self.radius}


@dataclass(frozen=True)
class Box(Primitive):
    center: V3
    half_extents: V3
    rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    kind: ClassVar[str] = "box"

    def sdf(self, p: Arr) -> Arr:
        q = np.abs((p - _v(self.center)) @ quat_to_matrix(self.rotation_wxyz)) - _v(self.half_extents)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        return np.asarray(outside + np.minimum(q.max(axis=1), 0.0), dtype=np.float64)

    def bounds(self) -> tuple[Arr, Arr]:
        c, m = _v(self.center), float(np.linalg.norm(_v(self.half_extents)))
        return c - m, c + m

    def params(self) -> dict[str, Any]:
        return {
            "center": list(self.center),
            "half_extents": list(self.half_extents),
            "rotation_wxyz": list(self.rotation_wxyz),
        }


@dataclass(frozen=True)
class Torus(Primitive):
    """Ring around ``axis`` (weld bead)."""

    center: V3
    axis: V3
    major_radius: float
    minor_radius: float
    kind: ClassVar[str] = "torus"

    def sdf(self, p: Arr) -> Arr:
        u = _v(self.axis) / np.linalg.norm(_v(self.axis))
        rel = p - _v(self.center)
        h = rel @ u
        radial = np.linalg.norm(rel - h[:, None] * u, axis=1)
        return np.asarray(np.hypot(radial - self.major_radius, h) - self.minor_radius, dtype=np.float64)

    def bounds(self) -> tuple[Arr, Arr]:
        c, m = _v(self.center), self.major_radius + self.minor_radius
        return c - m, c + m

    def params(self) -> dict[str, Any]:
        return {
            "center": list(self.center),
            "axis": list(self.axis),
            "major_radius": self.major_radius,
            "minor_radius": self.minor_radius,
        }


@dataclass(frozen=True)
class SphereBlob(Primitive):
    """Union of spheres: marine growth / biological geometry / irregular debris."""

    centers: tuple[V3, ...]
    radii: tuple[float, ...]
    kind: ClassVar[str] = "blob"

    def sdf(self, p: Arr) -> Arr:
        c, r = _v(self.centers), _v(self.radii)
        d = np.linalg.norm(p[:, None, :] - c[None, :, :], axis=2) - r[None, :]
        return np.asarray(d.min(axis=1), dtype=np.float64)

    def bounds(self) -> tuple[Arr, Arr]:
        c, r = _v(self.centers), _v(self.radii)[:, None]
        return (c - r).min(0), (c + r).max(0)

    def params(self) -> dict[str, Any]:
        return {"centers": [list(q) for q in self.centers], "radii": list(self.radii)}
