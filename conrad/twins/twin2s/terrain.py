"""Analytic seafloor heightfield z = f(x, y) (T2S-GEN-01) and the primitive factory. TRUTH PLANE.

f = base + slope.(x,y) + sum_i A_i sin(k_i.(x,y) + phi_i) + sum_j ridge/trench Gaussians.
The SDF is (z - f) / sqrt(1 + L^2) with L an analytic gradient bound: sign-exact and conservative.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from conrad.twins.twin2s.sdf import (
    Arr,
    Box,
    Capsule,
    Cylinder,
    Ellipsoid,
    PolyCapsule,
    Primitive,
    Sphere,
    SphereBlob,
    Torus,
    _t3,
    _v,
)

Wave = tuple[float, float, float, float]  # amplitude_m, kx, ky, phase
Ridge = tuple[float, float, float, float, float]  # height_m (neg = trench), x0, y0, heading_rad, sigma_m


@dataclass(frozen=True)
class HeightfieldTerrain(Primitive):
    base_z: float
    slope: tuple[float, float]
    waves: tuple[Wave, ...]
    ridges: tuple[Ridge, ...]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    kind: ClassVar[str] = "heightfield"
    exact: ClassVar[bool] = False

    def height(self, xy: Arr) -> Arr:
        x, y = xy[:, 0], xy[:, 1]
        h = self.base_z + self.slope[0] * x + self.slope[1] * y
        for amp, kx, ky, ph in self.waves:
            h = h + amp * np.sin(kx * x + ky * y + ph)
        for height, x0, y0, heading, sigma in self.ridges:
            across = -(x - x0) * math.sin(heading) + (y - y0) * math.cos(heading)
            h = h + height * np.exp(-0.5 * (across / sigma) ** 2)
        return np.asarray(h, dtype=np.float64)

    def lipschitz(self) -> float:
        bound = math.hypot(*self.slope)
        bound += sum(abs(a) * math.hypot(kx, ky) for a, kx, ky, _ in self.waves)
        bound += sum(abs(hh) / s * math.exp(-0.5) for hh, _, _, _, s in self.ridges)
        return bound

    def sdf(self, p: Arr) -> Arr:
        return (p[:, 2] - self.height(p[:, :2])) / math.sqrt(1.0 + self.lipschitz() ** 2)

    def bounds(self) -> tuple[Arr, Arr]:
        return _v(self.bounds_min), _v(self.bounds_max)

    def params(self) -> dict[str, Any]:
        return {
            "base_z": self.base_z,
            "slope": list(self.slope),
            "waves": [list(w) for w in self.waves],
            "ridges": [list(r) for r in self.ridges],
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
        }


def _quat(x: Any) -> tuple[float, float, float, float]:
    a = _v(x)
    return (float(a[0]), float(a[1]), float(a[2]), float(a[3]))


def primitive_from_dict(d: dict[str, Any]) -> Primitive:
    """Inverse of :meth:`Primitive.to_dict`. Unknown kinds fail closed."""
    k = d["kind"]
    if k == "sphere":
        return Sphere(_t3(d["center"]), float(d["radius"]))
    if k == "ellipsoid":
        return Ellipsoid(_t3(d["center"]), _t3(d["radii"]), _quat(d["rotation_wxyz"]))
    if k == "capsule":
        return Capsule(_t3(d["a"]), _t3(d["b"]), float(d["radius"]))
    if k == "polycapsule":
        return PolyCapsule(tuple(_t3(q) for q in d["points"]), float(d["radius"]))
    if k == "cylinder":
        return Cylinder(_t3(d["a"]), _t3(d["b"]), float(d["radius"]))
    if k == "box":
        return Box(_t3(d["center"]), _t3(d["half_extents"]), _quat(d["rotation_wxyz"]))
    if k == "torus":
        return Torus(_t3(d["center"]), _t3(d["axis"]), float(d["major_radius"]), float(d["minor_radius"]))
    if k == "blob":
        return SphereBlob(tuple(_t3(q) for q in d["centers"]), tuple(float(r) for r in d["radii"]))
    if k == "heightfield":
        return HeightfieldTerrain(
            float(d["base_z"]),
            (float(d["slope"][0]), float(d["slope"][1])),
            tuple((float(w[0]), float(w[1]), float(w[2]), float(w[3])) for w in d["waves"]),
            tuple((float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in d["ridges"]),
            _t3(d["bounds_min"]),
            _t3(d["bounds_max"]),
        )
    raise ValueError(f"unknown primitive kind {k!r}")
