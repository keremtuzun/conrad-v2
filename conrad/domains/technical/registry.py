"""Asset registry supplied as MISSION CONTEXT (ch10 structural graph, brief rule 3).

The structural graph is built only from design/registry data (component type, material, coating,
nominal wall, relationships). Degradation state never enters here. Mechanism susceptibility is the
model's OWN engineering table, not the truth twin's registry.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import UUID

CORROSION_DEPTH = "corrosion_depth_m"
CRACK_LENGTH = "crack_length_m"
SURFACE_ANOMALY = "surface_anomaly"
QUANTITIES: tuple[str, ...] = (CORROSION_DEPTH, CRACK_LENGTH, SURFACE_ANOMALY)
QUANTITY_UNITS: dict[str, str] = {CORROSION_DEPTH: "m", CRACK_LENGTH: "m", SURFACE_ANOMALY: "1"}


class DegradationMechanism(str, Enum):
    CORROSION = "CORROSION"
    FATIGUE = "FATIGUE"


QUANTITY_MECHANISM: dict[str, DegradationMechanism] = {
    CORROSION_DEPTH: DegradationMechanism.CORROSION,
    CRACK_LENGTH: DegradationMechanism.FATIGUE,
    SURFACE_ANOMALY: DegradationMechanism.CORROSION,
}
PROPAGATED_QUANTITIES: tuple[str, ...] = (CORROSION_DEPTH, CRACK_LENGTH)
"""Surface appearance is never propagated: it is a sensing artefact, not a shared physical state."""

CONTAINER_TYPES = frozenset({"ASSET", "PIPELINE", "RISER", "STRUCTURAL_FRAME"})
LOAD_BEARING_TYPES = frozenset({"SEGMENT", "WELD", "JOINT", "SUPPORT"})
METALLIC_MATERIALS = frozenset({"carbon_steel", "stainless_steel_316", "duplex_steel", "steel"})
NON_METALLIC_MATERIALS = frozenset({"concrete", "hdpe"})


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ComponentSpec:
    registry_id: UUID
    component_type: str
    material: str | None = None
    coating: str | None = None
    wall_thickness_m: float | None = None
    parent_id: UUID | None = None

    @property
    def is_container(self) -> bool:
        return self.component_type in CONTAINER_TYPES

    @property
    def material_known(self) -> bool:
        return self.material is None or self.material in METALLIC_MATERIALS | NON_METALLIC_MATERIALS

    def susceptible(self, mechanism: DegradationMechanism) -> bool:
        """Own table: metal loss needs a metal; fatigue cracking needs a metal on a load path."""
        if self.is_container or (self.material is not None and self.material in NON_METALLIC_MATERIALS):
            return False
        if mechanism is DegradationMechanism.FATIGUE:
            return self.component_type in LOAD_BEARING_TYPES
        return True

    def valid_quantities(self) -> frozenset[str]:
        return frozenset(q for q, m in QUANTITY_MECHANISM.items() if self.susceptible(m))


@dataclass(frozen=True)
class RegistryRelation:
    source: UUID
    target: UUID
    relation_type: str


class AssetRegistry:
    def __init__(self, components: Iterable[ComponentSpec], relations: Iterable[RegistryRelation]) -> None:
        self.components: dict[UUID, ComponentSpec] = {}
        for c in components:
            if c.registry_id in self.components:
                raise RegistryError(f"duplicate registry id {c.registry_id}")
            self.components[c.registry_id] = c
        self.relations: tuple[RegistryRelation, ...] = tuple(relations)
        for r in self.relations:
            if r.source not in self.components or r.target not in self.components:
                raise RegistryError(f"relation {r.relation_type} references an unknown component")
            if r.source == r.target:
                raise RegistryError("self relation")

    @property
    def stateful_ids(self) -> tuple[UUID, ...]:
        return tuple(sorted((k for k, c in self.components.items() if not c.is_container), key=str))

    def neighbours(self, entity_id: UUID, types: frozenset[str] | None = None) -> list[tuple[UUID, str]]:
        """Undirected neighbours with the relation type; ``types=None`` means every type."""
        out: list[tuple[UUID, str]] = []
        for r in self.relations:
            if types is not None and r.relation_type not in types:
                continue
            if r.source == entity_id:
                out.append((r.target, r.relation_type))
            elif r.target == entity_id:
                out.append((r.source, r.relation_type))
        return out

    @staticmethod
    def from_context(context: Mapping[str, Any]) -> AssetRegistry:
        """``context['asset_registry'] = {'components': [...], 'relationships': [...]}``."""
        raw = context.get("asset_registry")
        if not isinstance(raw, Mapping):
            raise RegistryError("context must carry an 'asset_registry' mapping")
        allowed = {"registry_id", "component_type", "material", "coating", "wall_thickness_m", "parent_id"}
        comps = []
        for item in raw.get("components", []):
            extra = set(item) - allowed
            if extra:
                raise RegistryError(f"asset registry carries non-design fields {sorted(extra)}")
            wall = item.get("wall_thickness_m")
            comps.append(
                ComponentSpec(
                    registry_id=UUID(str(item["registry_id"])),
                    component_type=str(item["component_type"]).upper(),
                    material=item.get("material"),
                    coating=item.get("coating"),
                    wall_thickness_m=None if wall is None else float(wall),
                    parent_id=None if item.get("parent_id") is None else UUID(str(item["parent_id"])),
                )
            )
        rels = [
            RegistryRelation(UUID(str(r["source"])), UUID(str(r["target"])), str(r["type"]))
            for r in raw.get("relationships", [])
        ]
        return AssetRegistry(comps, rels)
