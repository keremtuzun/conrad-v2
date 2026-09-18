"""Procedural building blocks: terrain, pipelines, welds, supports, clutter (T2S-GEN-01/02). TRUTH PLANE.

All randomness comes from an injected ``numpy.random.Generator``; all IDs from an injected ``IdFactory``.

implementation_status: EXPERIMENTAL_CANDIDATE (OCPWE)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.frames import WORLD
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import DomainOwnership, WorldEntity
from conrad.twins.twin2s.sdf import (
    V3,
    Arr,
    Box,
    Capsule,
    Ellipsoid,
    PolyCapsule,
    Primitive,
    SphereBlob,
    Torus,
    _t3,
)
from conrad.twins.twin2s.terrain import HeightfieldTerrain, Ridge, Wave
from conrad.twins.twin2s.world import Motion, SpatialEntity

TERRAIN_FAMILIES = ("smooth", "sloped", "ridged", "trench")


@dataclass
class WorldBuilder:
    ids: IdFactory
    created_at: TimeStamp
    world_entities: list[WorldEntity] = field(default_factory=list)
    spatial: list[SpatialEntity] = field(default_factory=list)
    adjacency: list[tuple[UUID, UUID, str]] = field(default_factory=list)

    def add(
        self,
        entity_type: str,
        primitive: Primitive | None,
        parent: UUID | None = None,
        technical: bool = False,
        ecological: bool = False,
        material_id: str = "unspecified",
        motion: Motion | None = None,
        active: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        eid = self.ids.new()
        self.world_entities.append(
            WorldEntity(
                id=eid,
                entity_type=entity_type,
                parent_id=parent,
                geometry_ref=None if primitive is None else f"twin2s:primitive:{primitive.kind}",
                reference_frame=WORLD,
                created_at=self.created_at,
                domain_ownership=DomainOwnership(
                    spatial=primitive is not None, technical=technical, ecological=ecological
                ),
                metadata=metadata or {},
            )
        )
        if primitive is not None:
            self.spatial.append(SpatialEntity(eid, entity_type, primitive, material_id, motion, active))
        return eid

    def link(self, a: UUID, b: UUID, relation: str) -> None:
        self.adjacency.append((a, b, relation))


def make_terrain(rng: np.random.Generator, family: str, lo: V3, hi: V3) -> HeightfieldTerrain:
    if family not in TERRAIN_FAMILIES:
        raise ValueError(f"unknown terrain family {family!r}")
    waves: list[Wave] = []
    for _ in range(3):
        k, ang = rng.uniform(0.2, 0.9), rng.uniform(0, math.pi)
        waves.append(
            (
                float(rng.uniform(0.02, 0.08)),
                k * math.cos(ang),
                k * math.sin(ang),
                float(rng.uniform(0, 6.28)),
            )
        )
    slope = (0.0, 0.0)
    ridges: list[Ridge] = []
    if family == "sloped":
        slope = (float(rng.uniform(-0.06, 0.06)), float(rng.uniform(-0.06, 0.06)))
    if family in ("ridged", "trench"):
        sign = 1.0 if family == "ridged" else -1.0
        ridges.append(
            (
                sign * float(rng.uniform(0.2, 0.5)),
                float(rng.uniform(lo[0] * 0.5, hi[0] * 0.5)),
                float(rng.uniform(lo[1] * 0.5, hi[1] * 0.5)),
                float(rng.uniform(0.9, 2.2)),
                float(rng.uniform(0.8, 1.6)),
            )
        )
    return HeightfieldTerrain(
        0.0, slope, tuple(waves), tuple(ridges), (lo[0], lo[1], lo[2]), (hi[0], hi[1], 1.5)
    )


def _h(terrain: HeightfieldTerrain, xy: Arr) -> float:
    return float(terrain.height(np.asarray(xy, dtype=np.float64)[None, :2])[0])


def pipeline_path(
    rng: np.random.Generator,
    start_xy: tuple[float, float],
    heading: float,
    n_segments: int,
    segment_length_m: float,
    bend_after: int | None,
    bend_angle_rad: float,
    bend_radius_m: float,
) -> list[tuple[str, list[Arr]]]:
    """Ordered pieces ('segment' | 'bend', xy polyline). A bend is a circular arc sampled every ~10 deg."""
    pieces: list[tuple[str, list[Arr]]] = []
    pos, hdg = np.array(start_xy, dtype=np.float64), heading
    for i in range(n_segments):
        nxt = pos + segment_length_m * np.array([math.cos(hdg), math.sin(hdg)])
        pieces.append(("segment", [pos, nxt]))
        pos = nxt
        if bend_after is not None and i == bend_after:
            sgn = 1.0 if bend_angle_rad >= 0 else -1.0
            centre = pos + sgn * bend_radius_m * np.array([-math.sin(hdg), math.cos(hdg)])
            steps = max(int(abs(bend_angle_rad) / math.radians(10.0)), 2)
            arc = []
            for s in range(steps + 1):
                a = hdg + bend_angle_rad * s / steps
                arc.append(centre - sgn * bend_radius_m * np.array([-math.sin(a), math.cos(a)]))
            pieces.append(("bend", arc))
            pos, hdg = arc[-1], hdg + bend_angle_rad
    return pieces


def add_pipeline(
    b: WorldBuilder,
    terrain: HeightfieldTerrain,
    pieces: list[tuple[str, list[Arr]]],
    radius_m: float,
    clearance_m: float,
    with_supports: bool,
    support_spacing_m: float,
    material_id: str = "steel_generic",
) -> dict[str, list[UUID]]:
    """Pipeline parent + segments/bends + welds at every joint + optional supports. Clearance < 0 buries."""
    parent = b.add("pipeline", None, technical=True, metadata={"radius_m": radius_m})
    out: dict[str, list[UUID]] = {
        "pipeline": [parent],
        "segments": [],
        "bends": [],
        "welds": [],
        "supports": [],
    }
    z_axis = max(_h(terrain, q) for _, poly in pieces for q in poly) + clearance_m + radius_m
    prev: UUID | None = None
    for kind, poly in pieces:
        pts = tuple((float(q[0]), float(q[1]), z_axis) for q in poly)
        prim: Primitive = (
            Capsule(pts[0], pts[1], radius_m) if kind == "segment" else PolyCapsule(pts, radius_m)
        )
        etype = "pipeline_segment" if kind == "segment" else "pipeline_bend"
        eid = b.add(etype, prim, parent, technical=True, material_id=material_id)
        out["segments" if kind == "segment" else "bends"].append(eid)
        if prev is not None:
            axis = np.array(pts[1]) - np.array(pts[0])
            weld = b.add(
                "weld",
                Torus(pts[0], _t3(axis), radius_m + 0.005, 0.02),
                parent,
                technical=True,
                material_id="weld_metal_generic",
            )
            out["welds"].append(weld)
            b.link(prev, weld, "JOINED_BY")
            b.link(weld, eid, "JOINED_BY")
        prev = eid
        if with_supports and kind == "segment" and clearance_m > 0.05:
            a, c = np.array(pts[0]), np.array(pts[1])
            n_sup = max(round(float(np.linalg.norm(c - a)) / support_spacing_m), 1)
            for s in range(n_sup):
                q = a + (c - a) * (s + 0.5) / n_sup
                ground = _h(terrain, q) - 0.1
                top = z_axis - radius_m * 0.6
                sup = b.add(
                    "support",
                    Box(
                        (float(q[0]), float(q[1]), 0.5 * (ground + top)),
                        (0.12, radius_m * 1.3, 0.5 * (top - ground)),
                    ),
                    parent,
                    technical=True,
                    material_id="concrete_generic",
                )
                out["supports"].append(sup)
                b.link(sup, eid, "SUPPORTS")
    return out


def add_clutter(
    b: WorldBuilder,
    rng: np.random.Generator,
    terrain: HeightfieldTerrain,
    lo: V3,
    hi: V3,
    n_rocks: int,
    n_debris: int,
    n_bio: int,
    n_dynamic: int,
    keep_clear: list[Arr] | None = None,
) -> dict[str, list[UUID]]:
    """Occluders: rocks, debris, biological blobs and moving entities (observability needs occlusion)."""
    out: dict[str, list[UUID]] = {"rocks": [], "debris": [], "biological": [], "dynamic": []}

    def spot() -> Arr:
        for _ in range(50):
            q = np.array([rng.uniform(lo[0] + 1, hi[0] - 1), rng.uniform(lo[1] + 1, hi[1] - 1)])
            if not keep_clear or min(float(np.linalg.norm(q - k[:2])) for k in keep_clear) > 0.9:
                return q
        return q

    for _ in range(n_rocks):
        q, r = spot(), rng.uniform(0.2, 0.7, size=3)
        prim: Primitive = Ellipsoid((float(q[0]), float(q[1]), _h(terrain, q) + 0.3 * float(r[2])), _t3(r))
        out["rocks"].append(b.add("rock", prim, material_id="rock_generic"))
    for _ in range(n_debris):
        q, he = spot(), rng.uniform(0.08, 0.4, size=3)
        prim = Box((float(q[0]), float(q[1]), _h(terrain, q) + float(he[2])), _t3(he))
        out["debris"].append(b.add("debris", prim, material_id="debris_generic"))
    for _ in range(n_bio):
        q = spot()
        base = np.array([q[0], q[1], _h(terrain, q) + 0.1])
        k = int(rng.integers(3, 7))
        centers = tuple(
            _t3(base + rng.normal(0, 0.15, size=3) * np.array([1, 1, 0.5]) + [0, 0, 0.1 * j])
            for j in range(k)
        )
        radii = tuple(float(x) for x in rng.uniform(0.08, 0.22, size=k))
        out["biological"].append(
            b.add("biological", SphereBlob(centers, radii), ecological=True, material_id="organic")
        )
    for _ in range(n_dynamic):
        q = spot()
        c = (float(q[0]), float(q[1]), _h(terrain, q) + float(rng.uniform(0.8, 2.0)))
        motion = Motion(
            _t3(rng.uniform(-0.15, 0.15, size=3) * np.array([1, 1, 0.1])),
            _t3(rng.uniform(0.0, 0.3, size=3)),
            float(rng.uniform(4.0, 12.0)),
        )
        prim = Ellipsoid(c, (0.35, 0.12, 0.15))
        out["dynamic"].append(
            b.add("dynamic_object", prim, ecological=True, material_id="organic", motion=motion)
        )
    return out
