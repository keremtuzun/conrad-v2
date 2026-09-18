"""WORLD TRUTH: canonical spatial world = entities with analytic geometry, semantics, dynamics.

S_true = (G, O, M, D): geometry (primitives), occupancy (sdf < 0), semantics (class + entity id),
dynamics (motion models + persistent changes). TRUTH PLANE.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from conrad.twins.twin2s.sdf import V3, Arr, Primitive, _t3, _v
from conrad.twins.twin2s.terrain import primitive_from_dict

SEMANTIC_CLASSES: tuple[str, ...] = (
    "water",
    "seafloor",
    "rock",
    "pipeline_segment",
    "pipeline_bend",
    "weld",
    "support",
    "frame",
    "cable",
    "debris",
    "biological",
    "dynamic_object",
    "anchor",
)
"""Index 0 ('water') means no surface. Indices are truth-plane labels only."""

EVENT_APPEAR = "SPATIAL_ENTITY_APPEAR"
EVENT_REMOVE = "SPATIAL_ENTITY_REMOVE"
EVENT_DISPLACE = "SPATIAL_ENTITY_DISPLACE"


def class_index(semantic_class: str) -> int:
    return SEMANTIC_CLASSES.index(semantic_class)


@dataclass(frozen=True)
class Motion:
    """x(t) = x0 + v t + A sin(2 pi t / T). Physical seconds, never a step index."""

    velocity_mps: V3 = (0.0, 0.0, 0.0)
    oscillation_amplitude_m: V3 = (0.0, 0.0, 0.0)
    oscillation_period_s: float = 1.0

    def offset(self, t_s: float) -> Arr:
        s = math.sin(2.0 * math.pi * t_s / self.oscillation_period_s)
        return _v(self.velocity_mps) * t_s + _v(self.oscillation_amplitude_m) * s

    def velocity(self, t_s: float) -> Arr:
        w = 2.0 * math.pi / self.oscillation_period_s
        return _v(self.velocity_mps) + _v(self.oscillation_amplitude_m) * w * math.cos(w * t_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "velocity_mps": list(self.velocity_mps),
            "oscillation_amplitude_m": list(self.oscillation_amplitude_m),
            "oscillation_period_s": self.oscillation_period_s,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Motion:
        return Motion(
            _t3(d["velocity_mps"]), _t3(d["oscillation_amplitude_m"]), float(d["oscillation_period_s"])
        )


@dataclass(frozen=True)
class SpatialEntity:
    entity_id: UUID
    semantic_class: str
    primitive: Primitive
    material_id: str = "unspecified"
    motion: Motion | None = None
    active: bool = True
    displacement_m: V3 = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_class": self.semantic_class,
            "primitive": self.primitive.to_dict(),
            "material_id": self.material_id,
            "motion": None if self.motion is None else self.motion.to_dict(),
            "active": self.active,
            "displacement_m": list(self.displacement_m),
        }

    @staticmethod
    def from_dict(entity_id: UUID, d: dict[str, Any]) -> SpatialEntity:
        return SpatialEntity(
            entity_id=entity_id,
            semantic_class=str(d["semantic_class"]),
            primitive=primitive_from_dict(d["primitive"]),
            material_id=str(d.get("material_id", "unspecified")),
            motion=None if d.get("motion") is None else Motion.from_dict(d["motion"]),
            active=bool(d.get("active", True)),
            displacement_m=_t3(d.get("displacement_m", (0.0, 0.0, 0.0))),
        )


class SpatialWorld:
    """Union of entity SDFs at physical time ``time_s``."""

    def __init__(self, entities: list[SpatialEntity], bounds_min: V3, bounds_max: V3, time_s: float = 0.0):
        class_index("water")
        for e in entities:
            class_index(e.semantic_class)
        self.entities = list(entities)
        self.bounds_min, self.bounds_max = bounds_min, bounds_max
        self.time_s = time_s

    # ---- construction -------------------------------------------------------------------------
    @staticmethod
    def from_spatial_state(spatial_state: dict[str, Any]) -> SpatialWorld:
        ents = [SpatialEntity.from_dict(UUID(str(k)), v) for k, v in spatial_state["entities"].items()]
        wb = spatial_state["world_bounds"]
        return SpatialWorld(ents, _t3(wb["min_m"]), _t3(wb["max_m"]))

    def entities_state(self) -> dict[str, Any]:
        return {str(e.entity_id): e.to_dict() for e in self.entities}

    def copy(self) -> SpatialWorld:
        return SpatialWorld(list(self.entities), self.bounds_min, self.bounds_max, self.time_s)

    def index_of(self, entity_id: UUID) -> int:
        for i, e in enumerate(self.entities):
            if e.entity_id == entity_id:
                return i
        raise KeyError(f"unknown world entity {entity_id}")

    def replace_entity(self, entity_id: UUID, **changes: Any) -> None:
        i = self.index_of(entity_id)
        self.entities[i] = replace(self.entities[i], **changes)

    # ---- geometry -----------------------------------------------------------------------------
    def entity_offset(self, i: int) -> Arr:
        e = self.entities[i]
        off = _v(e.displacement_m)
        return off if e.motion is None else off + e.motion.offset(self.time_s)

    def entity_sdf(self, i: int, p: Arr) -> Arr:
        return self.entities[i].primitive.sdf(p - self.entity_offset(i))

    def sdf_with_index(self, p: Arr) -> tuple[Arr, NDArray[np.int64]]:
        """Min signed distance and the index of the nearest active entity (-1 when none)."""
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        best = np.full(len(p), np.inf)
        idx = np.full(len(p), -1, dtype=np.int64)
        for i, e in enumerate(self.entities):
            if not e.active:
                continue
            off = self.entity_offset(i)
            lo, hi = e.primitive.bounds()
            q = p - off
            # distance to the bounding box is a lower bound on the entity SDF: cull far points.
            lb = np.linalg.norm(np.maximum(np.maximum(lo - q, q - hi), 0.0), axis=1)
            sel = np.nonzero(lb < best)[0]
            if sel.size == 0:
                continue
            d = e.primitive.sdf(q[sel])
            better = d < best[sel]
            hit = sel[better]
            best[hit] = d[better]
            idx[hit] = i
        return best, idx

    def sdf(self, p: Arr) -> Arr:
        return self.sdf_with_index(p)[0]

    def occupied(self, p: Arr) -> NDArray[np.bool_]:
        return np.asarray(self.sdf(p) < 0.0)

    def normals(self, p: Arr, eps: float = 1e-3) -> Arr:
        """Central-difference SDF gradient, normalised."""
        p = np.atleast_2d(p)
        g = np.zeros_like(p)
        for k in range(3):
            d = np.zeros(3)
            d[k] = eps
            g[:, k] = self.sdf(p + d) - self.sdf(p - d)
        n = np.linalg.norm(g, axis=1, keepdims=True)
        return np.asarray(g / np.maximum(n, 1e-12), dtype=np.float64)
