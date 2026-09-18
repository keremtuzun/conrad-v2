"""Registries for Twin2T: component types, relationship types, mechanisms, materials (ch10).

Component types live in a registry and are NOT pipeline-specific. Validity of a degradation mechanism
for a component is ``material.supports(mechanism) and component_type.supports(mechanism)``.

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Mechanism(str, Enum):
    CORROSION = "CORROSION"
    FATIGUE = "FATIGUE"
    COATING_FAILURE = "COATING_FAILURE"
    # Declared by ch10 but NOT implemented in V1 (kept so the coupling graph can name it).
    EROSION = "EROSION"


IMPLEMENTED_MECHANISMS: tuple[Mechanism, ...] = (
    Mechanism.CORROSION,
    Mechanism.FATIGUE,
    Mechanism.COATING_FAILURE,
)


class RelationshipType(str, Enum):
    """ch10 Relationship types, initial registry."""

    PART_OF = "PART_OF"
    CONNECTED_TO = "CONNECTED_TO"
    ADJACENT_TO = "ADJACENT_TO"
    SUPPORTED_BY = "SUPPORTED_BY"
    ATTACHED_TO = "ATTACHED_TO"
    LOAD_TRANSFER = "LOAD_TRANSFER"
    UPSTREAM_OF = "UPSTREAM_OF"
    DOWNSTREAM_OF = "DOWNSTREAM_OF"
    CONTACTS = "CONTACTS"
    EXPOSED_TO = "EXPOSED_TO"


class CouplingEdgeType(str, Enum):
    """ch10 Coupling graph edge types."""

    INCREASES_SUSCEPTIBILITY = "INCREASES_SUSCEPTIBILITY"
    MODIFIES_GEOMETRY = "MODIFIES_GEOMETRY"
    MODIFIES_LOADING = "MODIFIES_LOADING"
    ACCELERATES = "ACCELERATES"


@dataclass(frozen=True)
class ComponentTypeSpec:
    name: str
    is_container: bool
    """Containers (asset, pipeline) carry hierarchy only and no degradation state."""
    mechanisms: frozenset[Mechanism]
    load_bearing: bool = True

    def supports(self, mechanism: Mechanism) -> bool:
        return (not self.is_container) and mechanism in self.mechanisms


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    mechanisms: frozenset[Mechanism]
    corrosion_rate_factor: float
    """Dimensionless multiplier on the carbon-steel corrosion prior (ENGINEERING_ESTIMATE)."""
    citation: str

    def supports(self, mechanism: Mechanism) -> bool:
        return mechanism in self.mechanisms


_ALL = frozenset(IMPLEMENTED_MECHANISMS)
_CORR = frozenset({Mechanism.CORROSION, Mechanism.COATING_FAILURE})


class ComponentTypeRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ComponentTypeSpec] = {}

    def register(self, spec: ComponentTypeSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"component type {spec.name!r} already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ComponentTypeSpec:
        key = name.upper()
        if key not in self._specs:
            raise KeyError(f"unknown component type {name!r}; register it first")
        return self._specs[key]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.upper() in self._specs

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))


class MaterialRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, MaterialSpec] = {}

    def register(self, spec: MaterialSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"material {spec.name!r} already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> MaterialSpec:
        key = name.lower()
        if key not in self._specs:
            raise KeyError(f"unknown material {name!r}; register it first")
        return self._specs[key]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))


def default_component_types() -> ComponentTypeRegistry:
    reg = ComponentTypeRegistry()
    for name, container, mechs, load in (
        ("ASSET", True, frozenset(), False),
        ("PIPELINE", True, frozenset(), False),
        ("RISER", True, frozenset(), False),
        ("STRUCTURAL_FRAME", True, frozenset(), False),
        ("SEGMENT", False, _ALL, True),
        ("SURFACE_REGION", False, _CORR, False),
        ("WELD", False, _ALL, True),
        ("JOINT", False, _ALL, True),
        ("SUPPORT", False, _ALL, True),
        ("AUXILIARY_COMPONENT", False, _CORR, False),
    ):
        reg.register(ComponentTypeSpec(name, container, mechs, load))
    return reg


def default_materials() -> MaterialRegistry:
    reg = MaterialRegistry()
    est = "ENGINEERING_ESTIMATE: relative susceptibility class, not a measured rate"
    reg.register(MaterialSpec("carbon_steel", _ALL, 1.0, est))
    reg.register(MaterialSpec("stainless_steel_316", _ALL, 0.05, est))
    # V1 mechanisms (metal loss, Paris-law cracking) are not valid for these materials: masked out.
    reg.register(MaterialSpec("concrete", frozenset(), 0.0, est))
    reg.register(MaterialSpec("hdpe", frozenset(), 0.0, est))
    return reg
