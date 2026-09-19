"""Static scene geometry for Unity (``CONFIGURE_SCENE``): boxes, capsules and heightfields in Conrad WORLD.

This is the loader the integrator uses to hand Twin 2S primitive geometry to Unity as colliders. It only
describes geometry; it never carries twin state, entity identity or truth labels (primitive ids are
opaque strings chosen by the caller and must not be simulator hidden entity ids used as features).

Coordinates are Conrad WORLD (right-handed, +X forward, +Y left, +Z up, metres). The C# side converts
them with the single documented axis map (``ConradFrames`` / ``conrad.adapters.unity.frames``).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from conrad.adapters.unity.protocol import SceneConfigureRequest
from conrad.schemas.base import VersionedModel

Vec3 = tuple[float, float, float]
_MAX_PRIMITIVES = 4096  # SceneGeometryBuilder.MaxPrimitives
_MAX_HEIGHTFIELD_VERTICES = 1 << 20


def _finite(values: tuple[float, ...], what: str) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{what} must be finite")


class BoxPrimitive(VersionedModel):
    kind: Literal["box"] = "box"
    id: str = Field(min_length=1)
    center_m: Vec3
    size_m: Vec3 = Field(description="full extents along the box's own x, y, z axes")
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def _check(self) -> BoxPrimitive:
        _finite(self.center_m + self.size_m + self.orientation_wxyz, f"box {self.id}")
        if min(self.size_m) <= 0:
            raise ValueError(f"box {self.id} size must be positive")
        if math.sqrt(sum(q * q for q in self.orientation_wxyz)) < 1e-9:
            raise ValueError(f"box {self.id} orientation must be a non-zero quaternion")
        return self


class CapsulePrimitive(VersionedModel):
    kind: Literal["capsule"] = "capsule"
    id: str = Field(min_length=1)
    p0_m: Vec3
    p1_m: Vec3
    radius_m: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> CapsulePrimitive:
        _finite(self.p0_m + self.p1_m + (self.radius_m,), f"capsule {self.id}")
        if math.dist(self.p0_m, self.p1_m) <= 0:
            raise ValueError(f"capsule {self.id} end points must differ")
        return self


class HeightfieldPrimitive(VersionedModel):
    """Vertex (i, j) is ``origin + (i*dx, j*dy, heights[i][j])`` in Conrad WORLD."""

    kind: Literal["heightfield"] = "heightfield"
    id: str = Field(min_length=1)
    origin_m: Vec3
    spacing_m: tuple[float, float]
    heights_m: tuple[tuple[float, ...], ...]

    @field_validator("spacing_m")
    @classmethod
    def _spacing(cls, v: tuple[float, float]) -> tuple[float, float]:
        if not (v[0] > 0 and v[1] > 0 and math.isfinite(v[0]) and math.isfinite(v[1])):
            raise ValueError("heightfield spacing must be positive and finite")
        return v

    @model_validator(mode="after")
    def _check(self) -> HeightfieldPrimitive:
        nx = len(self.heights_m)
        ny = len(self.heights_m[0]) if nx else 0
        if nx < 2 or ny < 2 or nx * ny > _MAX_HEIGHTFIELD_VERTICES:
            raise ValueError(f"heightfield {self.id} grid must be at least 2x2 and at most 2^20 vertices")
        if any(len(row) != ny for row in self.heights_m):
            raise ValueError(f"heightfield {self.id} rows must have equal length")
        _finite(self.origin_m + tuple(h for row in self.heights_m for h in row), f"heightfield {self.id}")
        return self


Primitive = Annotated[BoxPrimitive | CapsulePrimitive | HeightfieldPrimitive, Field(discriminator="kind")]
_PRIMITIVE: TypeAdapter[BoxPrimitive | CapsulePrimitive | HeightfieldPrimitive] = TypeAdapter(Primitive)


class SceneGeometry(VersionedModel):
    """A static collider world for Unity. ``replace`` clears the previous world (including the default one)."""

    frame: Literal["WORLD"] = "WORLD"
    replace: bool = True
    primitives: tuple[Primitive, ...] = Field(max_length=_MAX_PRIMITIVES)

    @model_validator(mode="after")
    def _unique_ids(self) -> SceneGeometry:
        ids = [p.id for p in self.primitives]
        if len(ids) != len(set(ids)):
            raise ValueError("scene primitive ids must be unique")
        return self

    def to_wire(self) -> SceneConfigureRequest:
        prims = tuple(_primitive_wire(p) for p in self.primitives)
        return SceneConfigureRequest(frame=self.frame, replace=self.replace, primitives=prims)

    def to_json(self) -> dict[str, Any]:
        """The ``scene_json`` accepted by ``UnityRobotHardware.configure_scene``."""
        return self.to_wire().model_dump(mode="json")


def _primitive_wire(p: BoxPrimitive | CapsulePrimitive | HeightfieldPrimitive) -> dict[str, Any]:
    raw = p.model_dump(mode="json", exclude={"schema_version"})
    return {k: _lists(v) for k, v in raw.items()}


def _lists(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_lists(x) for x in v]
    return v


def parse_primitive(raw: dict[str, Any]) -> BoxPrimitive | CapsulePrimitive | HeightfieldPrimitive:
    return _PRIMITIVE.validate_python(raw)


def load_scene_geometry(path: str | Path) -> SceneGeometry:
    """Read a JSON scene file ``{"frame": "WORLD", "replace": true, "primitives": [...]}``."""
    return SceneGeometry.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
