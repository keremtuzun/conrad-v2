"""World families and Scenario assembly. OOD splits hold out STRUCTURAL families, never textures.

implementation_status: EXPERIMENTAL_CANDIDATE (OCPWE)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.frames import WORLD, Pose, quat_from_euler
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp
from conrad.schemas.world import MissionSpec, Scenario
from conrad.twins.twin2s.config import Twin2SConfig
from conrad.twins.twin2s.generator import (
    TERRAIN_FAMILIES,
    WorldBuilder,
    add_clutter,
    add_pipeline,
    make_terrain,
    pipeline_path,
)
from conrad.twins.twin2s.sdf import V3, Box, Cylinder, PolyCapsule
from conrad.twins.twin2s.sensor_specs import default_robot_spec

GENERATOR_VERSION = "twin2s-ocpwe-0.1.0"
TRAIN_FAMILIES = ("straight_pipeline", "bent_pipeline", "pipeline_with_supports", "cluttered_field")
OOD_FAMILIES = ("ood_unseen_geometry",)
WORLD_FAMILIES = TRAIN_FAMILIES + OOD_FAMILIES


def family_split(family: str) -> str:
    """'ood' for held-out structural families, else 'in_distribution'."""
    if family not in WORLD_FAMILIES:
        raise ValueError(f"unknown world family {family!r}")
    return "ood" if family in OOD_FAMILIES else "in_distribution"


@dataclass(frozen=True)
class WorldGenParams:
    """Simulation defaults (SYNTHETIC_ONLY); ranges are sampled per scenario and logged in metadata."""

    bounds_min: V3 = (-10.0, -6.0, -1.5)
    bounds_max: V3 = (10.0, 6.0, 5.0)
    pipe_radius_range_m: tuple[float, float] = (0.15, 0.35)
    segment_length_m: float = 4.0
    n_segments: int = 3
    support_spacing_m: float = 2.0
    water_surface_z_m: float = 30.0
    terrain_family: str | None = None
    n_rocks: int | None = None


@dataclass(frozen=True)
class GeneratedWorld:
    builder: WorldBuilder
    groups: dict[str, list[UUID]]
    sampled: dict[str, Any]
    robot_spawn: Pose


def generate_world(
    family: str, rng: np.random.Generator, ids: IdFactory, params: WorldGenParams
) -> GeneratedWorld:
    family_split(family)
    lo, hi = params.bounds_min, params.bounds_max
    b = WorldBuilder(ids, stamp(0.0, Twin2SConfig().clock_domain))
    tfam = params.terrain_family or TERRAIN_FAMILIES[int(rng.integers(len(TERRAIN_FAMILIES)))]
    terrain = make_terrain(rng, tfam, lo, hi)
    groups: dict[str, list[UUID]] = {"seafloor": [b.add("seafloor", terrain, material_id="sediment")]}
    radius = float(rng.uniform(*params.pipe_radius_range_m))
    heading = float(rng.uniform(-0.25, 0.25))
    length = params.segment_length_m * params.n_segments
    start = (-0.5 * length * math.cos(heading), -0.5 * length * math.sin(heading) + float(rng.uniform(-1, 1)))
    bend: tuple[int | None, float] = (None, 0.0)
    clearance, supports = float(rng.uniform(0.05, 0.25)), False
    clutter = (3, 1, 1, 0)
    if family == "bent_pipeline":
        bend = (0, float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.2)))
    elif family == "pipeline_with_supports":
        clearance, supports = float(rng.uniform(0.35, 0.7)), True
    elif family == "cluttered_field":
        clearance, clutter = float(rng.uniform(-0.15, 0.1)), (9, 5, 4, 2)
    if family != "ood_unseen_geometry":
        pieces = pipeline_path(
            rng, start, heading, params.n_segments, params.segment_length_m, bend[0], bend[1], 1.5
        )
        groups.update(add_pipeline(b, terrain, pieces, radius, clearance, supports, params.support_spacing_m))
        keep = [q for _, poly in pieces for q in poly]
        focus = np.mean(np.array(keep), axis=0)
        focus_z = clearance + radius
    else:
        groups.update(_ood_structures(b, rng, radius))
        keep, focus, focus_z = [np.zeros(2)], np.zeros(2), 1.5
        clutter = (4, 2, 2, 1)
    n_rocks = clutter[0] if params.n_rocks is None else params.n_rocks
    groups.update(add_clutter(b, rng, terrain, lo, hi, n_rocks, clutter[1], clutter[2], clutter[3], keep))
    spawn_xy = focus + 3.0 * np.array([-math.sin(heading), math.cos(heading)])
    yaw = math.atan2(focus[1] - spawn_xy[1], focus[0] - spawn_xy[0])
    spawn = Pose(
        frame_id=WORLD,
        position_m=(float(spawn_xy[0]), float(spawn_xy[1]), float(focus_z + 1.0)),
        orientation_wxyz=quat_from_euler(0.0, 0.25, yaw),
    )
    sampled = {
        "terrain_family": tfam,
        "pipe_radius_m": radius,
        "heading_rad": heading,
        "clearance_m": clearance,
        "bend_angle_rad": bend[1],
        "source_kind": "SYNTHETIC_ONLY",
    }
    return GeneratedWorld(b, groups, sampled, spawn)


def _ood_structures(b: WorldBuilder, rng: np.random.Generator, radius: float) -> dict[str, list[UUID]]:
    """Held-out structural family: vertical riser, box-frame lattice, catenary cable, anchor block."""
    frame_parent = b.add("frame", None, technical=True)
    frames, x0 = [], float(rng.uniform(-2.0, -1.0))
    for dx in (0.0, 2.0):
        for dy in (-1.0, 1.0):
            frames.append(
                b.add(
                    "frame",
                    Box((x0 + dx, dy, 1.0), (0.06, 0.06, 1.1)),
                    frame_parent,
                    technical=True,
                    material_id="steel_generic",
                )
            )
    for z in (0.6, 2.0):
        frames.append(
            b.add("frame", Box((x0 + 1.0, -1.0, z), (1.0, 0.05, 0.05)), frame_parent, technical=True)
        )
        frames.append(
            b.add("frame", Box((x0 + 1.0, 1.0, z), (1.0, 0.05, 0.05)), frame_parent, technical=True)
        )
    riser = b.add(
        "pipeline_segment",
        Cylinder((x0 + 1.0, 0.0, -0.1), (x0 + 1.0, 0.0, 4.0), radius),
        technical=True,
        material_id="steel_generic",
        metadata={"role": "riser"},
    )
    anchor = b.add(
        "anchor", Box((4.0, 2.0, 0.2), (0.4, 0.4, 0.3)), technical=True, material_id="concrete_generic"
    )
    xs = np.linspace(x0 + 1.0, 4.0, 9)
    sag = 1.2 * (1 - ((xs - xs.mean()) / (0.5 * (xs[-1] - xs[0]))) ** 2)
    pts = tuple(
        (
            float(x),
            float(2.0 * (x - xs[0]) / (xs[-1] - xs[0])),
            float(3.0 - 2.6 * (x - xs[0]) / (xs[-1] - xs[0]) - 0.5 * s),
        )
        for x, s in zip(xs, sag, strict=True)
    )
    cable = b.add("cable", PolyCapsule(pts, 0.03), technical=True, material_id="cable_generic")
    b.link(cable, anchor, "ATTACHED_TO")
    b.link(cable, riser, "ATTACHED_TO")
    return {
        "frames": frames,
        "segments": [riser],
        "anchors": [anchor],
        "cables": [cable],
        "welds": [],
        "supports": [],
    }


def spatial_state_section(
    g: GeneratedWorld, family: str, params: WorldGenParams, cfg: Twin2SConfig
) -> dict[str, Any]:
    return {
        "generator_version": GENERATOR_VERSION,
        "family": family,
        "world_bounds": {
            "frame_id": WORLD,
            "min_m": list(params.bounds_min),
            "max_m": list(params.bounds_max),
        },
        "entities": {str(e.entity_id): e.to_dict() for e in g.builder.spatial},
        "topology": {"adjacency": [[str(a), str(c), r] for a, c, r in g.builder.adjacency]},
        "groups": {k: [str(u) for u in v] for k, v in g.groups.items()},
        "robot_spawn": g.robot_spawn.model_dump(mode="json"),
        "mission_regions": [
            {"name": "inspection_corridor", "entity_ids": [str(u) for u in g.groups.get("segments", [])]}
        ],
        "octree": {
            "base_voxel_m": cfg.octree.base_voxel_m,
            "refinement_voxels_m": list(cfg.octree.refinement_voxels_m),
            "source_kind": "SYNTHETIC_ONLY",
        },
    }


def generate_scenario(
    seed: int,
    ids: IdFactory,
    family: str = "straight_pipeline",
    params: WorldGenParams | None = None,
    cfg: Twin2SConfig | None = None,
) -> Scenario:
    """One OCPWE world as a shared :class:`Scenario` (spatial section filled; 2T/2E sections reserved empty)."""
    params, cfg = params or WorldGenParams(), cfg or Twin2SConfig()
    g = generate_world(family, np.random.default_rng(seed), ids, params)
    targets = tuple(g.groups.get("segments", []) + g.groups.get("bends", []) + g.groups.get("welds", []))
    return Scenario(
        scenario_id=ids.new(),
        scenario_version=GENERATOR_VERSION,
        seed=seed,
        metadata={
            "world_family": family,
            "split": family_split(family),
            "lineage": f"{family}/{seed}",
            "sampled_parameters": g.sampled,
            "source_kind": "SYNTHETIC_ONLY",
        },
        world_entities=tuple(g.builder.world_entities),
        environment={
            "water_surface_z_m": params.water_surface_z_m,
            "turbidity": 0.0,
            "source_kind": "SYNTHETIC_ONLY",
        },
        structural_state={"entities": {}},
        ecological_state={"entities": {}},
        spatial_state=spatial_state_section(g, family, params, cfg),
        robots=(default_robot_spec(ids, g.robot_spawn),),
        mission=MissionSpec(
            mission_id=ids.new(),
            mission_type="PIPELINE_INSPECTION",
            target_entity_ids=targets,
            boundary_min_m=params.bounds_min,
            boundary_max_m=params.bounds_max,
            time_budget_s=1800.0,
        ),
    )
