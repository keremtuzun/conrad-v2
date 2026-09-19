"""Twin2S primitive geometry -> Unity ``CONFIGURE_SCENE`` colliders, with a measured approximation report.

TRUTH PLANE (simulator setup). Unity V2 builds three collider kinds (box, capsule, heightfield). Twin2S uses
more analytic primitives, so some are approximated. Every approximation is measured, never assumed:

==============  ===========================================================================================
Twin2S kind     Unity primitives
==============  ===========================================================================================
capsule         1 capsule (exact)
polycapsule     1 capsule per polyline segment (exact union)
box             1 oriented box (exact)
sphere          1 capsule with a 1 mm segment (error <= 1 mm)
blob            1 such sphere-capsule per member sphere (error <= 1 mm)
torus           ``torus_segments`` capsules along the major circle (chord sagitta error)
cylinder        1 capsule with end points pulled in by the radius (rounded rims)
ellipsoid       1 capsule along the longest axis, radius sqrt(b*c) of the two shorter semi-axes
heightfield     a regular vertex grid (``heightfield_spacing_m``) sampled from the analytic terrain,
                extended ``heightfield_margin_m`` beyond the world bounds (triangulation error)
==============  ===========================================================================================

Primitive ids are opaque (``p<entity index>-<k>``); world-entity UUIDs never enter the scene description.
Dynamic entities (with a motion model) cannot be represented by static colliders and are refused.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import Field

from conrad.schemas.base import ConradModel
from conrad.schemas.frames import quat_to_matrix
from conrad.sim.unity.scene import BoxPrimitive, CapsulePrimitive, HeightfieldPrimitive, SceneGeometry
from conrad.twins.twin2s.sdf import (
    Box,
    Capsule,
    Cylinder,
    Ellipsoid,
    PolyCapsule,
    Primitive,
    Sphere,
    SphereBlob,
    Torus,
)
from conrad.twins.twin2s.terrain import HeightfieldTerrain
from conrad.twins.twin2s.visibility import sample_surface
from conrad.twins.twin2s.world import SpatialWorld

UnityPrim = BoxPrimitive | CapsulePrimitive | HeightfieldPrimitive
Vec3 = tuple[float, float, float]
_EPS_SEGMENT_M = 1e-3


class TwinSceneOptions(ConradModel):
    heightfield_spacing_m: float = Field(default=0.25, gt=0)
    heightfield_margin_m: float = Field(default=8.0, ge=0, description="beyond world bounds (sensor range)")
    torus_segments: int = Field(default=24, ge=6)
    error_samples_per_entity: int = Field(default=200, gt=10)
    error_seed: int = 0x5CE


class KindError(ConradModel):
    """Measured geometric disagreement between Twin2S and the Unity colliders for one primitive kind."""

    entities: int
    unity_primitives: int
    twin_to_unity_max_m: float = Field(description="max |Unity SDF| at Twin2S surface samples")
    unity_to_twin_max_m: float = Field(description="max |Twin2S SDF| at Unity-collider surface samples")
    rms_m: float


class SceneConversionReport(ConradModel):
    primitives: int
    kinds: dict[str, KindError]
    heightfield_vertices: int
    heightfield_vertical_max_error_m: float
    notes: tuple[str, ...] = ()

    @property
    def max_error_m(self) -> float:
        vals = [max(k.twin_to_unity_max_m, k.unity_to_twin_max_m) for k in self.kinds.values()]
        return max([self.heightfield_vertical_max_error_m, *vals])


def _v(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _t(x: np.ndarray) -> Vec3:
    return (float(x[0]), float(x[1]), float(x[2]))


def _sphere(pid: str, c: np.ndarray, r: float) -> CapsulePrimitive:
    d = np.array([_EPS_SEGMENT_M / 2, 0.0, 0.0])
    return CapsulePrimitive(id=pid, p0_m=_t(c - d), p1_m=_t(c + d), radius_m=r)


def _capsule(pid: str, a: np.ndarray, b: np.ndarray, r: float) -> CapsulePrimitive:
    if float(np.linalg.norm(b - a)) < _EPS_SEGMENT_M:
        return _sphere(pid, (a + b) / 2, r)
    return CapsulePrimitive(id=pid, p0_m=_t(a), p1_m=_t(b), radius_m=r)


def _terrain_grid(
    t: HeightfieldTerrain, world: SpatialWorld, off: np.ndarray, o: TwinSceneOptions
) -> HeightfieldPrimitive:
    lo = _v(world.bounds_min)[:2] - o.heightfield_margin_m
    hi = _v(world.bounds_max)[:2] + o.heightfield_margin_m
    s = o.heightfield_spacing_m
    nx, ny = math.ceil((hi[0] - lo[0]) / s) + 1, math.ceil((hi[1] - lo[1]) / s) + 1
    xs, ys = lo[0] + s * np.arange(nx), lo[1] + s * np.arange(ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    h = t.height(np.stack([xx.ravel() - off[0], yy.ravel() - off[1]], axis=1)).reshape(nx, ny) + off[2]
    return HeightfieldPrimitive(
        id="terrain",
        origin_m=(float(lo[0]), float(lo[1]), 0.0),
        spacing_m=(s, s),
        heights_m=tuple(tuple(float(v) for v in row) for row in h),
    )


def _convert(i: int, prim: Primitive, off: np.ndarray, o: TwinSceneOptions) -> list[UnityPrim]:
    pid = f"p{i:04d}"
    if isinstance(prim, Capsule):
        return [_capsule(pid, _v(prim.a) + off, _v(prim.b) + off, prim.radius)]
    if isinstance(prim, PolyCapsule):
        pts = _v(prim.points) + off
        return [_capsule(f"{pid}-{k}", pts[k], pts[k + 1], prim.radius) for k in range(len(pts) - 1)]
    if isinstance(prim, Box):
        return [
            BoxPrimitive(
                id=pid,
                center_m=_t(_v(prim.center) + off),
                size_m=_t(2.0 * _v(prim.half_extents)),
                orientation_wxyz=prim.rotation_wxyz,
            )
        ]
    if isinstance(prim, Sphere):
        return [_sphere(pid, _v(prim.center) + off, prim.radius)]
    if isinstance(prim, SphereBlob):
        return [
            _sphere(f"{pid}-{k}", _v(c) + off, float(r))
            for k, (c, r) in enumerate(zip(prim.centers, prim.radii, strict=False))
        ]
    if isinstance(prim, Torus):
        u = _v(prim.axis) / float(np.linalg.norm(_v(prim.axis)))
        e1 = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        c, n = _v(prim.center) + off, o.torus_segments
        ring = [
            c + prim.major_radius * (math.cos(a) * e1 + math.sin(a) * e2)
            for a in np.linspace(0, 2 * math.pi, n + 1)
        ]
        return [_capsule(f"{pid}-{k}", ring[k], ring[k + 1], prim.minor_radius) for k in range(n)]
    if isinstance(prim, Cylinder):
        a, b = _v(prim.a) + off, _v(prim.b) + off
        length = float(np.linalg.norm(b - a))
        d = (b - a) / length
        pull = min(prim.radius, 0.49 * length)
        return [_capsule(pid, a + pull * d, b - pull * d, prim.radius)]
    if isinstance(prim, Ellipsoid):
        r = _v(prim.radii)
        order = np.argsort(r)[::-1]
        rot = quat_to_matrix(prim.rotation_wxyz)
        axis = rot[:, order[0]]
        rad = math.sqrt(float(r[order[1]] * r[order[2]]))
        half = max(float(r[order[0]]) - rad, _EPS_SEGMENT_M / 2)
        c = _v(prim.center) + off
        return [_capsule(pid, c - half * axis, c + half * axis, rad)]
    raise ValueError(f"no Unity conversion for Twin2S primitive kind {prim.kind!r}")


# ------------------------------------------------------------------------------------ Unity-side SDFs
def _capsule_sdf(p: np.ndarray, c: CapsulePrimitive) -> np.ndarray:
    a, b = _v(c.p0_m), _v(c.p1_m)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.asarray(np.linalg.norm(p - (a + t[:, None] * ab), axis=1) - c.radius_m)


def _box_sdf(p: np.ndarray, bx: BoxPrimitive) -> np.ndarray:
    q = np.abs((p - _v(bx.center_m)) @ quat_to_matrix(bx.orientation_wxyz)) - _v(bx.size_m) / 2
    return np.asarray(np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0))


def _prim_sdf(p: np.ndarray, u: UnityPrim) -> np.ndarray:
    if isinstance(u, CapsulePrimitive):
        return _capsule_sdf(p, u)
    if isinstance(u, BoxPrimitive):
        return _box_sdf(p, u)
    raise ValueError("heightfield error is measured vertically")


def _sample_unity_surface(u: UnityPrim, n: int, rng: np.random.Generator) -> np.ndarray:
    """Points on a Unity collider surface: random points projected along the collider SDF gradient."""
    if isinstance(u, CapsulePrimitive):
        a, b, r = _v(u.p0_m), _v(u.p1_m), u.radius_m
        ax = (b - a) / float(np.linalg.norm(b - a))
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        along = d @ ax
        side = rng.random(n) < 0.5
        radial = d - along[:, None] * ax
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
        body = a + rng.uniform(0, 1, n)[:, None] * (b - a) + r * radial
        caps = np.where((along >= 0)[:, None], b, a) + r * d  # outward hemispheres at both ends
        return np.asarray(np.where(side[:, None], body, caps))
    assert isinstance(u, BoxPrimitive)
    h = _v(u.size_m) / 2
    local = rng.uniform(-h, h, size=(n, 3))
    face = rng.integers(0, 3, n)
    local[np.arange(n), face] = np.where(rng.random(n) < 0.5, -1.0, 1.0) * h[face]
    return np.asarray(local @ quat_to_matrix(u.orientation_wxyz).T + _v(u.center_m))


def _surface_of_union(prims: list[UnityPrim], pts: np.ndarray) -> np.ndarray:
    d = np.min(np.stack([_prim_sdf(pts, u) for u in prims]), axis=0)
    return np.asarray(np.abs(d) < 1e-6)


def _terrain_interp(hf: HeightfieldPrimitive, xy: np.ndarray) -> np.ndarray:
    """Height of the Unity triangulation (v00, v11, v10) / (v00, v01, v11) at WORLD xy."""
    h = _v(hf.heights_m)
    s = _v(hf.spacing_m)
    g = (xy - _v(hf.origin_m)[:2]) / s
    i = np.clip(np.floor(g[:, 0]).astype(int), 0, h.shape[0] - 2)
    j = np.clip(np.floor(g[:, 1]).astype(int), 0, h.shape[1] - 2)
    fx, fy = g[:, 0] - i, g[:, 1] - j
    h00, h10, h01, h11 = h[i, j], h[i + 1, j], h[i, j + 1], h[i + 1, j + 1]
    lower = fx >= fy  # triangle (v00, v10, v11)
    z_lo = h00 + fx * (h10 - h00) + fy * (h11 - h10)
    z_up = h00 + fy * (h01 - h00) + fx * (h11 - h01)
    return np.asarray(np.where(lower, z_lo, z_up) + hf.origin_m[2])


def twin_scene(
    world: SpatialWorld, options: TwinSceneOptions | None = None
) -> tuple[SceneGeometry, SceneConversionReport]:
    """Unity scene for the Twin2S world at its current physical time, plus the measured conversion error."""
    o = options or TwinSceneOptions()
    rng = np.random.default_rng(o.error_seed)
    prims: list[UnityPrim] = []
    per_kind: dict[str, dict[str, Any]] = {}
    hf_err, hf_vertices, notes = 0.0, 0, []
    for i, ent in enumerate(world.entities):
        if not ent.active:
            continue
        if ent.motion is not None:
            raise ValueError(
                f"entity #{i} ({ent.semantic_class}) is dynamic; Unity V2 scene colliders are static"
            )
        off = world.entity_offset(i)
        if isinstance(ent.primitive, HeightfieldTerrain):
            hf = _terrain_grid(ent.primitive, world, off, o)
            prims.append(hf)
            hf_vertices += len(hf.heights_m) * len(hf.heights_m[0])
            lo, hi = _v(world.bounds_min)[:2], _v(world.bounds_max)[:2]
            xy = rng.uniform(lo, hi, size=(20000, 2))
            true_h = ent.primitive.height(xy - off[:2]) + off[2]
            hf_err = max(hf_err, float(np.max(np.abs(_terrain_interp(hf, xy) - true_h))))
            continue
        conv = _convert(i, ent.primitive, off, o)
        prims += conv
        k = per_kind.setdefault(
            ent.primitive.kind, {"n": 0, "u": 0, "t2u": 0.0, "u2t": 0.0, "sq": [], "cnt": 0}
        )
        k["n"] += 1
        k["u"] += len(conv)
        tp, _ = sample_surface(world, i, o.error_samples_per_entity, rng, exposed_only=False)
        if len(tp):
            du = np.min(np.stack([_prim_sdf(tp, u) for u in conv]), axis=0)
            k["t2u"] = max(k["t2u"], float(np.max(np.abs(du))))
            k["sq"].append(du**2)
        up = np.concatenate([_sample_unity_surface(u, o.error_samples_per_entity, rng) for u in conv])
        up = up[_surface_of_union(conv, up)]
        if len(up):
            dt = world.entity_sdf(i, up)
            k["u2t"] = max(k["u2t"], float(np.max(np.abs(dt))))
            k["sq"].append(dt**2)
    kinds = {
        name: KindError(
            entities=v["n"],
            unity_primitives=v["u"],
            twin_to_unity_max_m=round(v["t2u"], 6),
            unity_to_twin_max_m=round(v["u2t"], 6),
            rms_m=round(float(np.sqrt(np.mean(np.concatenate(v["sq"])))) if v["sq"] else 0.0, 6),
        )
        for name, v in sorted(per_kind.items())
    }
    if "ellipsoid" in kinds:
        notes.append("ellipsoids are approximated by one capsule; see ellipsoid error")
    scene = SceneGeometry(primitives=tuple(prims))
    report = SceneConversionReport(
        primitives=len(prims),
        kinds=kinds,
        heightfield_vertices=hf_vertices,
        heightfield_vertical_max_error_m=round(hf_err, 6),
        notes=tuple(notes),
    )
    return scene, report
