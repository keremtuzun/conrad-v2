"""Per-twin sections of the ONE shared pipeline scenario. TRUTH PLANE.

The shared scenario (Twin2S geometry, world-entity IDs) is built once. Twin2T and Twin2E each get their own
copy of the Scenario whose section follows their documented contract; the world-entity IDs never change:

* Twin2T: generic Twin2S labels are mapped onto the Twin2T component/material registries, design topology
  (JOINED_BY adjacency) becomes CONNECTED_TO relations, and the hidden defect is set as the target segment's
  initial corrosion depth and crack length. Twin2T state is per component, but the defect is local (the
  far-side patch), so the rest of the target's surface gets its own truth-side Twin2T component (a
  "surface region" entity that exists only in the Twin2T copy, sampled from the ordinary priors). Near-side
  views of the target read that region: they legitimately report "no crack here" without revealing the
  defect. The region entity never reaches the registry, the mapping or any Observation.
* Twin2E: the environment carries values with units, every ecological entity gets ``position_m`` from the
  Twin2S geometry, and the sea surface is the scenario's declared water surface.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.capsule_surface import CapsuleSurfaceGrid, SurfaceRect
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import DomainOwnership, Scenario, WorldEntity
from conrad.sim.mission.options import MissionWorldOptions
from conrad.twins.twin2e import Twin2EConfig, populate_ecological_state
from conrad.twins.twin2e.config import GridConfig
from conrad.twins.twin2s.world import SpatialWorld
from conrad.twins.twin2t import populate_structural_state
from conrad.twins.twin2t.spatial_field import (
    LocalEvolutionRate,
    LocalStructuralState,
    SpatialEvolutionV1,
    SpatialStructuralTruth,
    TruthPatch,
    spatial_truth_record,
)

TYPE_MAP = {
    "pipeline": "PIPELINE",
    "pipeline_segment": "SEGMENT",
    "pipeline_bend": "SEGMENT",
    "weld": "WELD",
    "support": "SUPPORT",
}
MATERIAL_MAP = {
    "steel_generic": "carbon_steel",
    "weld_metal_generic": "carbon_steel",
    "concrete_generic": "concrete",
}
ECO_KIND = {"biological": "BIOFOULING_ON_STRUCTURE", "dynamic_object": "MOBILE_GROUP"}


def structural_scenario(scenario: Scenario, target: UUID, opts: MissionWorldOptions, seed: int) -> Scenario:
    """Twin2T copy of the scenario with the hidden defect injected on the target segment."""
    given = scenario.structural_state.get("entities", {})
    entities: dict[str, dict[str, Any]] = {}
    rels: list[dict[str, str]] = []
    for we in scenario.world_entities:
        if not we.domain_ownership.technical:
            continue
        base = given.get(str(we.id), {})
        ctype = TYPE_MAP.get(str(base.get("component_type", we.entity_type)), "AUXILIARY_COMPONENT")
        entry: dict[str, Any] = {"component_type": ctype}
        if ctype != "PIPELINE":
            entry["material"] = MATERIAL_MAP.get(str(base.get("material")), "carbon_steel")
        entities[str(we.id)] = entry
        if we.parent_id is not None:
            rels.append({"source": str(we.id), "target": str(we.parent_id), "type": "PART_OF"})
    for a, b, _kind in scenario.spatial_state.get("topology", {}).get("adjacency", []):
        if a in entities and b in entities:
            rels.append({"source": a, "target": b, "type": "CONNECTED_TO"})
    env = {k: v for k, v in scenario.environment.items() if k != "turbidity"}
    sc = scenario.model_copy(
        update={"structural_state": {"entities": entities, "relationships": rels}, "environment": env}
    )
    sc = populate_structural_state(sc, np.random.default_rng([seed, 0x2E7]))
    struct = {k: dict(v) for k, v in sc.structural_state["entities"].items()}
    struct[str(target)]["initial_corrosion_depth_m"] = (
        0.0 if opts.twin2t_truth_model == "spatial_v1" else opts.defect.corrosion_depth_m
    )
    struct[str(target)]["initial_crack_length_m"] = (
        0.0 if opts.twin2t_truth_model == "spatial_v1" else opts.defect.crack_length_m
    )
    sc = sc.model_copy(update={"structural_state": {**sc.structural_state, "entities": struct}})
    if opts.twin2t_truth_model == "spatial_v1":
        return sc
    sc = _with_rest_region(sc, target, seed)
    if opts.defect.pristine_rest:  # I5-NOMINAL-READABLE only: a truly intact target surface
        region = rest_region_of(sc, target)
        if region is not None:
            ents = {k: dict(v) for k, v in sc.structural_state["entities"].items()}
            ents[str(region)].update(initial_corrosion_depth_m=0.0, initial_crack_length_m=0.0)
            sc = sc.model_copy(update={"structural_state": {**sc.structural_state, "entities": ents}})
    return sc


def with_spatial_truth(
    scenario: Scenario, target: UUID, opts: MissionWorldOptions, length_m: float, radius_m: float
) -> Scenario:
    """Install explicit local truth and rates in the Twin2T scenario, not the mission context."""
    spec = opts.spatial_truth
    if opts.twin2t_truth_model != "spatial_v1" or spec is None:
        raise ValueError("spatial_v1 truth configuration required")
    grid = CapsuleSurfaceGrid(length_m, radius_m, spec.axial_cells, spec.sectors)
    base = tuple(
        LocalStructuralState(**value.model_dump())
        for value in (spec.cell_states or (spec.base,) * grid.n_cells)
    )
    patches = tuple(
        TruthPatch(
            SurfaceRect(
                patch.axial_start_fraction * length_m,
                patch.axial_end_fraction * length_m,
                patch.angle_start_rad,
                patch.angle_end_rad,
            ),
            LocalStructuralState(**patch.state.model_dump()),
        )
        for patch in spec.patches
    )
    truth = SpatialStructuralTruth(grid, base, patches)
    rates = SpatialEvolutionV1(
        tuple(
            LocalEvolutionRate(**value.model_dump())
            for value in (spec.cell_rates or (spec.base_rate,) * grid.n_cells)
        ),
        tuple(LocalEvolutionRate(**patch.rate.model_dump()) for patch in spec.patches),
    )
    structural = dict(scenario.structural_state)
    structural["spatial_fields"] = {str(target): spatial_truth_record(truth, rates)}
    return scenario.model_copy(update={"structural_state": structural})


REGION_OF = "surface_region_of"
"""WorldEntity.metadata key naming the component whose non-defect surface a region entity represents."""


def _with_rest_region(sc: Scenario, target: UUID, seed: int) -> Scenario:
    """Add the target's non-defect surface as its own Twin2T component (populated from the normal priors).

    Populated with a dedicated RNG after every real component, so no existing component's state changes."""
    base = next(e for e in sc.world_entities if e.id == target)
    region = WorldEntity(
        id=IdFactory(seed).child("twin2t-surface-regions").new(),
        entity_type=base.entity_type,
        reference_frame=base.reference_frame,
        created_at=base.created_at,
        domain_ownership=DomainOwnership(spatial=False, technical=True, ecological=False),
        metadata={REGION_OF: str(target)},
    )
    design = {
        k: v for k, v in sc.structural_state["entities"][str(target)].items() if not k.startswith("initial_")
    }  # same segment: same design, exposure and loading; its own initial degradation is sampled
    struct = {**sc.structural_state["entities"], str(region.id): design}
    sc = sc.model_copy(
        update={
            "world_entities": (*sc.world_entities, region),
            "structural_state": {**sc.structural_state, "entities": struct},
        }
    )
    return populate_structural_state(sc, np.random.default_rng([seed, 0x2E7, 1]))


def rest_region_of(t2t_scenario: Scenario, target: UUID) -> UUID | None:
    return next((e.id for e in t2t_scenario.world_entities if e.metadata.get(REGION_OF) == str(target)), None)


def entity_centres(world: SpatialWorld) -> dict[UUID, np.ndarray]:
    out: dict[UUID, np.ndarray] = {}
    for i, ent in enumerate(world.entities):
        lo, hi = ent.primitive.bounds()
        out[ent.entity_id] = (np.asarray(lo) + np.asarray(hi)) / 2.0 + world.entity_offset(i)
    return out


def ecological_scenario(
    scenario: Scenario, world: SpatialWorld, opts: MissionWorldOptions, seed: int
) -> Scenario:
    """Twin2E copy: unit-carrying environment, positions from the shared geometry, declared sea surface."""
    centres = entity_centres(world)
    spatial = {k: dict(v) for k, v in scenario.spatial_state.get("entities", {}).items()}
    eco: dict[str, dict[str, Any]] = {}
    for we in scenario.world_entities:
        if not we.domain_ownership.ecological:
            continue
        c = centres[we.id]
        spatial.setdefault(str(we.id), {})["position_m"] = [float(c[0]), float(c[1]), float(c[2])]
        eco[str(we.id)] = {"entity_kind": ECO_KIND.get(we.entity_type, "BIOFOULING_ON_STRUCTURE")}
    surface = float(scenario.environment.get("water_surface_z_m", 30.0))
    sc = scenario.model_copy(
        update={
            "environment": dict(opts.environment),
            "spatial_state": {**scenario.spatial_state, "entities": spatial},
            "ecological_state": {"entities": eco, "parameters": {"surface_z_m": surface}},
        }
    )
    return populate_ecological_state(sc, np.random.default_rng([seed, 0x2E3]))


def twin2e_config(world: SpatialWorld, surface_z_m: float) -> Twin2EConfig:
    """Small grids that contain the Twin2S world and reach the declared sea surface."""
    lo = np.asarray(world.bounds_min, dtype=np.float64) - 5.0
    hi = np.asarray(world.bounds_max, dtype=np.float64) + 5.0
    top = max(float(hi[2]), surface_z_m)
    span = np.array([hi[0] - lo[0], hi[1] - lo[1], top - lo[2]])
    local_shape = (6, 6, 6)
    local = GridConfig(
        origin_m=(float(lo[0]), float(lo[1]), float(lo[2])),
        spacing_m=tuple(float(s / n) for s, n in zip(span, local_shape, strict=True)),
        shape=local_shape,
    )
    regional = GridConfig(
        origin_m=(float(lo[0]) - 40.0, float(lo[1]) - 40.0, float(lo[2])),
        spacing_m=(float(span[0] + 80.0) / 4, float(span[1] + 80.0) / 4, float(span[2]) / 3),
        shape=(4, 4, 3),
    )
    return Twin2EConfig.model_validate(
        {
            "regional_grid": regional.model_dump(),
            "local_grid": local.model_dump(),
            "ecology": {"time_scale": 1.0},
            "clock_domain": "SIM",
        }
    )
