"""Build the structural graph and per-component truth runtimes from a (shared) Scenario.

Refuses to attach structural state to an entity absent from ``world_entities`` or not technically owned.

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.world import Scenario
from conrad.twins.twin2t.priors import (
    HIDDEN_PARAMETER_NAMES,
    merged_priors,
    sample_parameter,
    supplied_parameter,
)
from conrad.twins.twin2t.registry import (
    ComponentTypeRegistry,
    MaterialRegistry,
    RelationshipType,
    default_component_types,
    default_materials,
)
from conrad.twins.twin2t.scenario import (
    StructuralScenarioError,
    populate_structural_state,
    technical_entities,
)
from conrad.twins.twin2t.state import (
    ComponentParameters,
    ComponentRuntime,
    DegradationState,
    Environment,
    Loading,
    validity_for,
)
from conrad.twins.twin2t.topology import StructuralComponent, StructuralGraph, StructuralRelationship

_UNITS = {
    "wall_thickness_m": "m",
    "initial_corrosion_depth_m": "m",
    "initial_crack_length_m": "m",
    "load_cycles_per_s": "1/s",
    "stress_range_pa": "Pa",
}


@dataclass
class Assembly:
    scenario: Scenario
    graph: StructuralGraph
    runtimes: dict[UUID, ComponentRuntime]
    prior_log: dict[str, list[dict[str, Any]]]


def _environment(entry: dict[str, Any], shared: dict[str, Any]) -> Environment:
    env = {
        **{
            k: v
            for k, v in shared.items()
            if k in {"temperature_c", "dissolved_oxygen_mg_l", "cp_active", "buried"}
        },
        **dict(entry.get("environment", {})),
    }
    return Environment(
        temperature_c=float(env["temperature_c"]),
        dissolved_oxygen_mg_l=float(env["dissolved_oxygen_mg_l"]),
        cp_active=bool(env.get("cp_active", False)),
        buried=bool(env.get("buried", False)),
    )


def assemble(
    scenario: Scenario,
    rng: np.random.Generator,
    *,
    prior_overrides: dict[str, dict[str, Any]] | None = None,
    component_types: ComponentTypeRegistry | None = None,
    materials: MaterialRegistry | None = None,
) -> Assembly:
    ctypes = component_types or default_component_types()
    mats = materials or default_materials()
    known = {e.id: e for e in scenario.world_entities}
    raw_entities = dict(scenario.structural_state.get("entities", {}))
    for key in raw_entities:
        eid = UUID(str(key))
        if eid not in known:
            raise StructuralScenarioError(
                f"refusing structural state for entity {key}: not in world_entities"
            )
        if not known[eid].domain_ownership.technical:
            raise StructuralScenarioError(
                f"refusing structural state for entity {key}: not technically owned"
            )
    original = {k: set(dict(v)) for k, v in raw_entities.items()}
    scenario = populate_structural_state(
        scenario, rng, prior_overrides=prior_overrides, component_types=ctypes
    )
    entities = scenario.structural_state["entities"]
    prior_log = {k: list(v) for k, v in scenario.structural_state.get("prior_log", {}).items()}
    priors = merged_priors(prior_overrides)
    comps: list[StructuralComponent] = []
    runtimes_spec: list[tuple[StructuralComponent, dict[str, Any]]] = []
    for ent in sorted(technical_entities(scenario), key=lambda e: str(e.id)):
        entry = entities[str(ent.id)]
        ctype = ctypes.get(entry["component_type"])
        material = None if ctype.is_container else mats.get(str(entry["material"]))
        parent = (
            ent.parent_id
            if ent.parent_id is not None and known[ent.parent_id].domain_ownership.technical
            else None
        )
        comp = StructuralComponent(
            ent.id, ctype, parent, ent.geometry_ref, material, {"coating": entry.get("coating")}
        )
        comps.append(comp)
        if not ctype.is_container:
            runtimes_spec.append((comp, entry))
    rels: list[StructuralRelationship] = []
    for raw in scenario.structural_state.get("relationships", []):
        try:
            rtype = RelationshipType(str(raw["type"]).upper())
        except ValueError as exc:
            raise StructuralScenarioError(f"unknown relationship type {raw.get('type')!r}") from exc
        src, dst = UUID(str(raw["source"])), UUID(str(raw["target"]))
        for end in (src, dst):
            if end not in known:
                raise StructuralScenarioError(f"relationship endpoint {end} not in world_entities")
        feats = {k: float(v) for k, v in dict(raw.get("features", {})).items()}
        rels.append(StructuralRelationship(src, dst, rtype, feats))
    graph = StructuralGraph(comps, rels)
    runtimes: dict[UUID, ComponentRuntime] = {}
    for comp, entry in runtimes_spec:
        key = str(comp.entity_id)
        supplied = original.get(key, set())
        params = {}
        for name in _UNITS:
            if name in supplied:
                params[name] = supplied_parameter(name, float(entry[name]), _UNITS[name])
        for name in HIDDEN_PARAMETER_NAMES:
            sp = sample_parameter(priors[name], rng)
            params[name] = sp
            prior_log.setdefault(key, []).append(sp.as_dict())
        cp = ComponentParameters(
            float(entry["wall_thickness_m"]), str(entry.get("coating", "none")) != "none", params
        )
        validity = validity_for(comp)
        a0 = float(entry["initial_crack_length_m"]) if validity["crack_length_m"] else 0.0
        d0 = (
            min(cp.wall_thickness_m, float(entry["initial_corrosion_depth_m"]))
            if validity["corrosion_depth_m"]
            else 0.0
        )
        cb0 = cp.value("coating_a") if (cp.coated and validity["coating_breakdown_fraction"]) else 0.0
        exposed = cb0 if cp.coated else 1.0
        state = DegradationState(
            corrosion_depth_m=d0,
            corrosion_area_fraction=exposed if d0 > 0 else 0.0,
            coating_breakdown_fraction=cb0,
            crack_length_m=a0,
            crack_depth_m=min(cp.wall_thickness_m, a0 / cp.value("crack_aspect_ratio")),
            validity=validity,
        )
        loading = Loading(float(entry["load_cycles_per_s"]), float(entry["stress_range_pa"]))
        runtimes[comp.entity_id] = ComponentRuntime(
            comp, cp, state, _environment(entry, scenario.environment), loading, initial_state=state
        )
    return Assembly(scenario, graph, runtimes, prior_log)
