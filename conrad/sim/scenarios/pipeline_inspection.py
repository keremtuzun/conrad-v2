"""Shared pipeline-inspection scenario: ONE world, ONE set of WorldEntity IDs, three twin sections.

Twin2S geometry is generated here. Twin2T / Twin2E fill their own sections through these documented keys,
always keyed by the SAME world-entity UUID string:

    structural_state["entities"][<uuid>] = {"material": str, "component_type": str, ...Twin2T-owned keys}
    ecological_state["entities"][<uuid>] = {"ecological_role": str, ...Twin2E-owned keys}

Material names are generic simulation labels (SYNTHETIC_ONLY), not measured material data.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.ids import IdFactory
from conrad.schemas.world import MissionSpec, Scenario
from conrad.twins.twin2s.config import Twin2SConfig
from conrad.twins.twin2s.families import (
    GENERATOR_VERSION,
    WorldGenParams,
    family_split,
    generate_world,
    spatial_state_section,
)
from conrad.twins.twin2s.sensor_specs import default_robot_spec

_STRUCTURAL_TYPES = {"pipeline_segment", "pipeline_bend", "weld", "support", "frame", "cable", "anchor"}
_ECOLOGICAL_TYPES = {"biological": "sessile_growth", "dynamic_object": "mobile_fauna"}


def build_pipeline_inspection_scenario(
    seed: int,
    ids: IdFactory,
    family: str = "pipeline_with_supports",
    params: WorldGenParams | None = None,
    cfg: Twin2SConfig | None = None,
    sensor_overrides: dict[str, dict[str, Any]] | None = None,
    time_budget_s: float = 1800.0,
) -> Scenario:
    """Deterministic in (seed, ids seed). Returns a Scenario with robot, 5 sensors and a MissionSpec."""
    params, cfg = params or WorldGenParams(), cfg or Twin2SConfig()
    g = generate_world(family, np.random.default_rng(seed), ids, params)
    material = {e.entity_id: e.material_id for e in g.builder.spatial}
    structural: dict[str, dict[str, Any]] = {}
    ecological: dict[str, dict[str, Any]] = {}
    for we in g.builder.world_entities:
        if we.entity_type in _STRUCTURAL_TYPES and we.id in material:
            structural[str(we.id)] = {"material": material[we.id], "component_type": we.entity_type}
        if we.entity_type in _ECOLOGICAL_TYPES:
            ecological[str(we.id)] = {"ecological_role": _ECOLOGICAL_TYPES[we.entity_type]}
    targets: tuple[UUID, ...] = tuple(
        g.groups.get("segments", [])
        + g.groups.get("bends", [])
        + g.groups.get("welds", [])
        + g.groups.get("supports", [])
    )
    return Scenario(
        scenario_id=ids.new(),
        scenario_version=GENERATOR_VERSION,
        seed=seed,
        metadata={
            "name": "pipeline_inspection",
            "world_family": family,
            "split": family_split(family),
            "lineage": f"pipeline_inspection/{family}/{seed}",
            "sampled_parameters": g.sampled,
            "source_kind": "SYNTHETIC_ONLY",
        },
        world_entities=tuple(g.builder.world_entities),
        environment={
            "water_surface_z_m": params.water_surface_z_m,
            "turbidity": 0.0,
            "source_kind": "SYNTHETIC_ONLY",
        },
        structural_state={"entities": structural},
        ecological_state={"entities": ecological},
        spatial_state=spatial_state_section(g, family, params, cfg),
        robots=(default_robot_spec(ids, g.robot_spawn, sensor_overrides),),
        mission=MissionSpec(
            mission_id=ids.new(),
            mission_type="PIPELINE_INSPECTION",
            target_entity_ids=targets,
            boundary_min_m=params.bounds_min,
            boundary_max_m=params.bounds_max,
            time_budget_s=time_budget_s,
            parameters={"inspect": ["segments", "welds", "supports"]},
        ),
    )
