"""Structural graph G^T = (V^T, E^T) (ch10 Canonical structural hierarchy, Structural node schema).

Static component identity is kept separate from dynamic state (``state.py``). Relationships are typed
(:class:`RelationshipType`); missing edge features are masked (absent), never fabricated.

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from conrad.twins.twin2t.registry import ComponentTypeSpec, MaterialSpec, Mechanism, RelationshipType


class TopologyError(ValueError):
    """The structural graph references an unknown component or is otherwise inconsistent."""


@dataclass(frozen=True)
class StructuralComponent:
    """ch10 StructuralComponent. ``entity_id`` is the shared WorldEntity id (truth plane only)."""

    entity_id: UUID
    component_type: ComponentTypeSpec
    parent_id: UUID | None
    geometry_ref: str | None
    material: MaterialSpec | None
    design_metadata: dict[str, Any] = field(default_factory=dict)

    def mechanism_valid(self, mechanism: Mechanism) -> bool:
        """Validity mask M_i^state: both the component type and the material must admit it."""
        if self.material is None:
            return False
        return self.component_type.supports(mechanism) and self.material.supports(mechanism)

    @property
    def has_state(self) -> bool:
        return not self.component_type.is_container


@dataclass(frozen=True)
class StructuralRelationship:
    """R_ij^T = [type, features]. ``features`` only holds values that were supplied."""

    source: UUID
    target: UUID
    rel_type: RelationshipType
    features: dict[str, float] = field(default_factory=dict)


# Relationship types across which a mechanism-specific cross-component effect is physically justified.
SHARED_ENVIRONMENT_TYPES = frozenset(
    {RelationshipType.CONNECTED_TO, RelationshipType.ATTACHED_TO, RelationshipType.CONTACTS}
)
LOAD_PATH_TYPES = frozenset({RelationshipType.SUPPORTED_BY, RelationshipType.LOAD_TRANSFER})


class StructuralGraph:
    def __init__(
        self, components: Iterable[StructuralComponent], relationships: Iterable[StructuralRelationship]
    ) -> None:
        self._components: dict[UUID, StructuralComponent] = {}
        for comp in components:
            if comp.entity_id in self._components:
                raise TopologyError(f"duplicate component {comp.entity_id}")
            self._components[comp.entity_id] = comp
        self._relationships: list[StructuralRelationship] = []
        for rel in relationships:
            for end in (rel.source, rel.target):
                if end not in self._components:
                    raise TopologyError(f"relationship references unknown component {end}")
            if rel.source == rel.target:
                raise TopologyError("self-relationship is not allowed")
            self._relationships.append(rel)

    @property
    def component_ids(self) -> tuple[UUID, ...]:
        """Deterministic order: sorted by UUID string."""
        return tuple(sorted(self._components, key=str))

    def component(self, entity_id: UUID) -> StructuralComponent:
        if entity_id not in self._components:
            raise TopologyError(f"unknown component {entity_id}")
        return self._components[entity_id]

    def __contains__(self, entity_id: object) -> bool:
        return entity_id in self._components

    @property
    def relationships(self) -> tuple[StructuralRelationship, ...]:
        return tuple(self._relationships)

    def relationships_of(self, entity_id: UUID) -> tuple[StructuralRelationship, ...]:
        return tuple(r for r in self._relationships if entity_id in (r.source, r.target))

    def neighbours(self, entity_id: UUID, types: frozenset[RelationshipType]) -> tuple[UUID, ...]:
        """Undirected neighbours over the given relationship types (sorted, deduplicated)."""
        out: set[UUID] = set()
        for r in self._relationships:
            if r.rel_type not in types:
                continue
            if r.source == entity_id:
                out.add(r.target)
            elif r.target == entity_id:
                out.add(r.source)
        return tuple(sorted(out, key=str))

    def load_supporters(self, entity_id: UUID) -> tuple[UUID, ...]:
        """Components carrying load for ``entity_id``: X SUPPORTED_BY Y, or Y LOAD_TRANSFER X."""
        out: set[UUID] = set()
        for r in self._relationships:
            if r.rel_type is RelationshipType.SUPPORTED_BY and r.source == entity_id:
                out.add(r.target)
            elif r.rel_type is RelationshipType.LOAD_TRANSFER and r.target == entity_id:
                out.add(r.source)
        return tuple(sorted(out, key=str))

    def children(self, entity_id: UUID) -> tuple[UUID, ...]:
        return tuple(i for i in self.component_ids if self._components[i].parent_id == entity_id)

    def export(self) -> dict[str, Any]:
        return {
            "components": [
                {
                    "entity_id": str(c.entity_id),
                    "component_type": c.component_type.name,
                    "parent_id": None if c.parent_id is None else str(c.parent_id),
                    "geometry_ref": c.geometry_ref,
                    "material_id": None if c.material is None else c.material.name,
                    "design_metadata": dict(c.design_metadata),
                }
                for c in (self._components[i] for i in self.component_ids)
            ],
            "relationships": [
                {
                    "source": str(r.source),
                    "target": str(r.target),
                    "type": r.rel_type.value,
                    "features": dict(r.features),
                }
                for r in self._relationships
            ],
        }
