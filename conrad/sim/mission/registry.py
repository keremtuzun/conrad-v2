"""Mission context from the truth world: FRESH registry IDs, surveyed design geometry. TRUTH PLANE.

The registry-ID -> world-entity-ID mapping stays here (and in the truth record). Design geometry carries
survey noise; hidden condition (initial corrosion, cracks, the defect) is never copied.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from conrad.orchestration.mission_context import DesignComponent, MissionContext
from conrad.schemas.frames import WORLD, Pose, quat_from_euler
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import MissionSpec, Scenario, SensorSpec

DESIGN_KEYS = ("component_type", "material", "coating", "wall_thickness_m")


@dataclass(frozen=True)
class RegistryMapping:
    """TRUTH-SIDE ONLY: registry ID -> world-entity ID."""

    to_world: dict[UUID, UUID]

    @property
    def to_registry(self) -> dict[UUID, UUID]:
        return {w: r for r, w in self.to_world.items()}


def _noisy(v: Any, sigma: float, rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.asarray(v, dtype=np.float64) + sigma * rng.standard_normal(3)
    return (float(x[0]), float(x[1]), float(x[2]))


def _design(
    rid: UUID, ctype: str, prim: dict[str, Any], sigma: float, rng: np.random.Generator
) -> DesignComponent:
    kind = prim["kind"]
    if kind == "polycapsule":  # surveyed as its chord; the bend sagitta is folded into the radius
        pts = np.asarray(prim["points"], dtype=np.float64)
        a, b = pts[0], pts[-1]
        ab = b - a
        t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        sag = float(np.max(np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)))
        prim = {"kind": "capsule", "a": a.tolist(), "b": b.tolist(), "radius": float(prim["radius"]) + sag}
        kind = "capsule"
    if kind == "capsule":
        return DesignComponent(
            registry_id=rid,
            component_type=ctype,
            shape="CAPSULE",
            p0_m=_noisy(prim["a"], sigma, rng),
            p1_m=_noisy(prim["b"], sigma, rng),
            radius_m=float(prim["radius"]),
            survey_sigma_m=sigma,
        )
    if kind == "torus":
        r = float(prim["major_radius"]) + float(prim["minor_radius"])
        return DesignComponent(
            registry_id=rid,
            component_type=ctype,
            shape="SPHERE",
            p0_m=_noisy(prim["center"], sigma, rng),
            radius_m=r,
            survey_sigma_m=sigma,
        )
    half = np.asarray(prim.get("half_extents", (0.2, 0.2, 0.2)), dtype=np.float64)
    return DesignComponent(
        registry_id=rid,
        component_type=ctype,
        shape="BOX",
        p0_m=_noisy(prim["center"], sigma, rng),
        half_extent_m=(float(half.max()), float(half.max()), float(half[2])),
        survey_sigma_m=sigma,
    )


def transit_lane(axis: list[np.ndarray], offset: float, height: float, margin: float) -> list[np.ndarray]:
    """Polyline on the -Y side of the pipe axis (right-hand normal of the pipe heading)."""
    a, b = axis[0], axis[-1]
    d = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
    if d[1] > 0 or (d[1] == 0 and d[0] < 0):  # travel so that the pipe (+Y side) is on the LEFT
        axis = axis[::-1]
        a, b, d = b, a, -d
    right = np.array([d[1], -d[0], 0.0])
    pts = [a - margin * d, *axis, b + margin * d]
    return [p + offset * right + np.array([0.0, 0.0, height]) for p in pts]


def build_mission_context(
    scenario: Scenario,
    t2t_scenario: Scenario,
    target: UUID,
    sensors: tuple[SensorSpec, ...],
    structural_sensor_ids: tuple[UUID, ...],
    ids: IdFactory,
    rng: np.random.Generator,
    lane: list[np.ndarray],
    seabed_z_m: float,
    survey_sigma_m: float,
    launch_sigma_m: float,
) -> tuple[MissionContext, RegistryMapping]:
    struct = t2t_scenario.structural_state["entities"]
    prims = scenario.spatial_state["entities"]
    technical = [e for e in scenario.world_entities if e.domain_ownership.technical]
    to_world = {ids.new(): e.id for e in technical}
    to_reg = {w: r for r, w in to_world.items()}
    comps: list[dict[str, Any]] = []
    design: list[DesignComponent] = []
    for rid, wid in to_world.items():
        entry = struct[str(wid)]
        we = next(e for e in technical if e.id == wid)
        item: dict[str, Any] = {
            "registry_id": rid,
            "parent_id": to_reg.get(we.parent_id) if we.parent_id else None,
        }
        item.update({k: entry[k] for k in DESIGN_KEYS if k in entry})
        comps.append(item)
        prim = prims.get(str(wid), {}).get("primitive")
        if prim:
            design.append(_design(rid, str(entry["component_type"]), prim, survey_sigma_m, rng))
    rels = [
        {"source": to_reg[UUID(r["source"])], "target": to_reg[UUID(r["target"])], "type": r["type"]}
        for r in t2t_scenario.structural_state.get("relationships", [])
        if UUID(r["source"]) in to_reg and UUID(r["target"]) in to_reg
    ]
    assert scenario.mission is not None
    spec = MissionSpec(
        mission_id=scenario.mission.mission_id,
        mission_type="PIPELINE_INSPECTION",
        target_entity_ids=tuple(to_reg[w] for w in scenario.mission.target_entity_ids if w in to_reg),
        boundary_min_m=scenario.mission.boundary_min_m,
        boundary_max_m=scenario.mission.boundary_max_m,
        time_budget_s=scenario.mission.time_budget_s,
        parameters={"inspect": ["segments", "welds"], "source_kind": "SYNTHETIC_ONLY"},
    )
    start, nxt = lane[0], lane[1]
    yaw = math.atan2(float(nxt[1] - start[1]), float(nxt[0] - start[0]))
    launch = Pose(
        frame_id=WORLD,
        position_m=_noisy(start, launch_sigma_m, rng),
        orientation_wxyz=quat_from_euler(0, 0, yaw),
    )
    ctx = MissionContext(
        mission_id=spec.mission_id,
        spec=spec,
        asset_registry={"components": comps, "relationships": rels},
        design=tuple(design),
        critical_component_ids=(to_reg[target],),
        sensors=sensors,
        structural_sensor_ids=structural_sensor_ids,
        launch_pose=launch,
        transit_lane=tuple((float(p[0]), float(p[1]), float(p[2])) for p in lane),
        seabed_z_m=float(seabed_z_m + survey_sigma_m * rng.standard_normal()),
    )
    return ctx, RegistryMapping(to_world)
