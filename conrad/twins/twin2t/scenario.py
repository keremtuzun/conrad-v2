"""Shared-scenario structural section: fill-in from the logged prior, and a small standalone builder.

``scenario.structural_state`` convention:
  entities[str(world_entity_id)] = {component_type?, material?, wall_thickness_m?, coating?,
      initial_corrosion_depth_m?, initial_crack_length_m?, load_cycles_per_s?, stress_range_pa?,
      environment?: {temperature_c?, dissolved_oxygen_mg_l?, cp_active?, buried?}}
  relationships = [{source, target, type}]
  prior_log[str(world_entity_id)] = [{name, value, units, source, citation}]   (written here)

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp
from conrad.schemas.world import DomainOwnership, Scenario, WorldEntity
from conrad.twins.twin2t.priors import (
    ENGINEERING_ESTIMATE,
    PriorSpec,
    merged_priors,
    sample_parameter,
)
from conrad.twins.twin2t.registry import ComponentTypeRegistry, RelationshipType, default_component_types

ENTITY_NUMERIC_KEYS: tuple[str, ...] = (
    "wall_thickness_m",
    "initial_corrosion_depth_m",
    "initial_crack_length_m",
    "load_cycles_per_s",
    "stress_range_pa",
)
_CAT_CITATION = "engineering estimate for synthetic scenario generation; not calibrated against data"
MATERIAL_PRIOR: dict[str, dict[str, float]] = {
    "SUPPORT": {"carbon_steel": 0.6, "concrete": 0.4},
    "*": {"carbon_steel": 0.85, "stainless_steel_316": 0.15},
}
COATING_PRIOR: dict[str, float] = {"fbe": 0.7, "none": 0.3}
ENV_PRIORS: dict[str, tuple[float, float]] = {
    "temperature_c": (4.0, 25.0),
    "dissolved_oxygen_mg_l": (5.0, 9.0),
}


class StructuralScenarioError(ValueError):
    pass


def _log(name: str, value: Any, units: str) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "units": units,
        "source": ENGINEERING_ESTIMATE,
        "citation": _CAT_CITATION,
    }


def _choice(rng: np.random.Generator, table: dict[str, float]) -> str:
    keys = sorted(table)
    probs = np.asarray([table[k] for k in keys], dtype=np.float64)
    return keys[int(rng.choice(len(keys), p=probs / probs.sum()))]


def technical_entities(scenario: Scenario) -> tuple[WorldEntity, ...]:
    return tuple(e for e in scenario.world_entities if e.domain_ownership.technical)


def populate_structural_state(
    scenario: Scenario,
    rng: np.random.Generator,
    *,
    prior_overrides: dict[str, dict[str, Any]] | None = None,
    component_types: ComponentTypeRegistry | None = None,
    overwrite: bool = False,
) -> Scenario:
    """Return a new Scenario whose structural section is complete; every filled value is prior-logged."""
    reg = component_types or default_component_types()
    priors: dict[str, PriorSpec] = merged_priors(prior_overrides)
    section = dict(scenario.structural_state)
    known = {e.id for e in scenario.world_entities}
    entities: dict[str, dict[str, Any]] = {k: dict(v) for k, v in dict(section.get("entities", {})).items()}
    for key in entities:
        if UUID(key) not in known:
            raise StructuralScenarioError(f"structural state references unknown world entity {key}")
    prior_log: dict[str, list[dict[str, Any]]] = {
        k: list(v) for k, v in dict(section.get("prior_log", {})).items()
    }
    for ent in sorted(technical_entities(scenario), key=lambda e: str(e.id)):
        key = str(ent.id)
        entry = entities.setdefault(key, {})
        log = prior_log.setdefault(key, [])
        ctype = str(entry.get("component_type", ent.entity_type)).upper()
        if ctype not in reg:
            raise StructuralScenarioError(f"entity {key}: unknown component type {ctype!r}")
        entry["component_type"] = ctype
        if reg.get(ctype).is_container:
            continue
        if overwrite or "material" not in entry:
            entry["material"] = _choice(rng, MATERIAL_PRIOR.get(ctype, MATERIAL_PRIOR["*"]))
            log.append(_log("material", entry["material"], "category"))
        if overwrite or "coating" not in entry:
            entry["coating"] = "none" if entry["material"] == "concrete" else _choice(rng, COATING_PRIOR)
            log.append(_log("coating", entry["coating"], "category"))
        for name in ENTITY_NUMERIC_KEYS:
            if overwrite or name not in entry:
                sp = sample_parameter(priors[name], rng)
                entry[name] = sp.value
                log.append(sp.as_dict())
        env = dict(entry.get("environment", {}))
        for name, (lo, hi) in ENV_PRIORS.items():
            if name not in env and name not in scenario.environment:
                env[name] = float(rng.uniform(lo, hi))
                log.append(_log(f"environment.{name}", env[name], "degC" if "temp" in name else "mg/L"))
        entry["environment"] = env
    rels = list(section.get("relationships", []))
    if not rels:
        for ent in scenario.world_entities:
            if ent.parent_id is not None and ent.domain_ownership.technical:
                rels.append(
                    {
                        "source": str(ent.id),
                        "target": str(ent.parent_id),
                        "type": RelationshipType.PART_OF.value,
                    }
                )
    section.update(entities=entities, relationships=rels, prior_log=prior_log)
    data = dict(scenario)
    data["structural_state"] = section
    return Scenario(**data)


def build_small_scenario(
    seed: int,
    *,
    n_segments: int = 3,
    with_concrete_support: bool = True,
    ids: IdFactory | None = None,
    populate: bool = True,
) -> Scenario:
    """Asset -> pipeline -> segments (each with weld + surface region), supports, auxiliary component."""
    ids = ids or IdFactory(seed=seed)
    t0 = stamp(0.0, "sim")
    ents: list[WorldEntity] = []
    rels: list[dict[str, str]] = []
    structural: dict[str, dict[str, Any]] = {}
    tech = DomainOwnership(spatial=True, technical=True)

    def add(etype: str, parent: UUID | None, **attrs: Any) -> UUID:
        eid = ids.new()
        ents.append(
            WorldEntity(
                id=eid,
                entity_type=etype,
                parent_id=parent,
                reference_frame="WORLD",
                created_at=t0,
                domain_ownership=tech,
                geometry_ref=f"geom:{etype.lower()}",
            )
        )
        structural[str(eid)] = {"component_type": etype, **attrs}
        if parent is not None:
            rels.append({"source": str(eid), "target": str(parent), "type": "PART_OF"})
        return eid

    asset = add("ASSET", None)
    pipe = add("PIPELINE", asset)
    segs = [add("SEGMENT", pipe) for _ in range(n_segments)]
    for i, seg in enumerate(segs):
        add("SURFACE_REGION", seg)
        weld = add("WELD", seg)
        rels.append({"source": str(weld), "target": str(seg), "type": "ATTACHED_TO"})
        if i > 0:
            joint = add("JOINT", segs[i - 1])
            rels.append({"source": str(segs[i - 1]), "target": str(seg), "type": "CONNECTED_TO"})
            rels.append({"source": str(joint), "target": str(seg), "type": "CONNECTED_TO"})
    sup = add("SUPPORT", pipe, material="carbon_steel")
    rels.append({"source": str(segs[0]), "target": str(sup), "type": "SUPPORTED_BY"})
    if with_concrete_support:
        csup = add("SUPPORT", pipe, material="concrete", coating="none")
        rels.append({"source": str(segs[-1]), "target": str(csup), "type": "SUPPORTED_BY"})
    add("AUXILIARY_COMPONENT", asset)
    scenario = Scenario(
        scenario_id=ids.new(),
        scenario_version="twin2t-small-0.1",
        seed=seed,
        world_entities=tuple(ents),
        environment={"temperature_c": 12.0, "dissolved_oxygen_mg_l": 7.0},
        structural_state={"entities": structural, "relationships": rels},
        metadata={"generator": "twin2t.build_small_scenario"},
    )
    return populate_structural_state(scenario, np.random.default_rng(seed)) if populate else scenario
