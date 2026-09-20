"""MissionWorld: one shared scenario, three twins, the 6-DOF kernel and the payload suite. TRUTH PLANE.

Every component gets its own ``IdFactory(seed).child("<name>")`` stream so no two components mint the same
IDs. The robot is spawned at the start of the -Y transit lane; the hidden defect sits on the +Y side of the
target segment. The target is observable through two surface targets: the defect patch (Twin2T target
component) and the rest of its surface (a truth-side Twin2T region component without the defect), so a
near-side view reports what is really there instead of nothing.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from conrad.orchestration.mission_context import MissionContext
from conrad.persistence.object_store import ObjectStore
from conrad.robotics.hardware.config import load_robot_config
from conrad.schemas.frames import ROBOT, Pose, quat_from_euler
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Scenario, SensorSpec
from conrad.sim.kernel import FaultType, SimKernelConfig
from conrad.sim.mission.hardware import MissionHardware, build_mission_hardware
from conrad.sim.mission.options import MissionWorldOptions
from conrad.sim.mission.registry import RegistryMapping, build_mission_context, transit_lane
from conrad.sim.mission.sensing import MissionSensorSuite, SuiteSensors, SurfaceTarget
from conrad.sim.mission.structure import (
    ecological_scenario,
    rest_region_of,
    structural_scenario,
    twin2e_config,
)
from conrad.sim.mission.truth import MissionTruthRecorder
from conrad.sim.scenarios.pipeline_inspection import build_pipeline_inspection_scenario
from conrad.twins.twin2e import Twin2E
from conrad.twins.twin2s.twin import Twin2S
from conrad.twins.twin2s.visibility import sample_surface
from conrad.twins.twin2s.world import SpatialWorld
from conrad.twins.twin2t import Twin2T

ROBOT_CONFIG = "configs/robot/sim_reference.yaml"
REGION_READINGS = True
"""Emit readings of the target's non-defect surface (docs/audits/MODEL2T_REPAIR.md). ON since iteration 3:
Model2T tracks surface coverage on the surveyed design geometry, so a near-side lane view makes only the read
cells OBSERVED; the target's component-level condition stays UNKNOWN with high U_O and the information need is
still raised."""


def _spec(
    ids: IdFactory,
    modality: str,
    frame: str,
    rate: float,
    params: dict[str, Any],
    mount: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_deg: float = 0.0,
) -> SensorSpec:
    return SensorSpec(
        sensor_id=ids.new(),
        modality=modality,
        frame_id=frame,
        mount_pose=Pose(
            frame_id=ROBOT,
            position_m=mount,
            orientation_wxyz=quat_from_euler(0.0, 0.0, math.radians(yaw_deg)),
        ),
        rate_hz=rate,
        parameters=params,
    )


def _pipe_axis(world: SpatialWorld, segments: list[UUID]) -> list[np.ndarray]:
    ends: list[np.ndarray] = []
    for sid in segments:
        prim = world.entities[world.index_of(sid)].primitive
        ends += [np.asarray(prim.a, dtype=np.float64), np.asarray(prim.b, dtype=np.float64)]  # type: ignore[attr-defined]
    d = ends[-1] - ends[0]
    uniq: list[np.ndarray] = []
    for p in sorted(ends, key=lambda q: float(q @ d)):
        if not uniq or float(np.linalg.norm(p - uniq[-1])) > 1e-6:
            uniq.append(p)
    return uniq


def _patch_mask(
    pts: np.ndarray,
    nrm: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    axis: list[np.ndarray],
    opts: MissionWorldOptions,
) -> np.ndarray:
    """True for surface samples inside the hidden defect patch of the target segment."""
    d = (axis[-1] - axis[0]) / np.linalg.norm(axis[-1] - axis[0])
    left = np.array([-d[1], d[0], 0.0])
    if left[1] < 0:
        left = -left
    if opts.defect.side == "near":
        left = -left
    tilt = math.radians(opts.defect.patch_tilt_deg)
    left = math.cos(tilt) * left + math.sin(tilt) * np.array([0.0, 0.0, 1.0])
    t = ((pts - a) @ (b - a)) / float((b - a) @ (b - a))
    half = opts.defect.patch_axial_fraction / 2.0
    return np.asarray(
        (nrm @ left > math.cos(math.radians(opts.defect.patch_half_angle_deg))) & (np.abs(t - 0.5) <= half)
    )


def _ends(world: SpatialWorld, target: UUID) -> tuple[int, np.ndarray, np.ndarray]:
    i = world.index_of(target)
    prim = world.entities[i].primitive
    a, b = np.asarray(prim.a, dtype=np.float64), np.asarray(prim.b, dtype=np.float64)  # type: ignore[attr-defined]
    return i, a, b


def _patch(
    world: SpatialWorld,
    target: UUID,
    axis: list[np.ndarray],
    opts: MissionWorldOptions,
    rng: np.random.Generator,
) -> SurfaceTarget:
    i, a, b = _ends(world, target)
    pts, nrm = sample_surface(world, i, opts.defect.patch_samples * 12, rng)
    keep = _patch_mask(pts, nrm, a, b, axis, opts)
    idx = np.nonzero(keep)[0][: opts.defect.patch_samples]
    if len(idx) < 4:
        raise RuntimeError("defect patch has too few surface samples; increase patch size")
    return SurfaceTarget(target, pts[idx], nrm[idx], is_patch=True)


def _rest_of_target(
    world: SpatialWorld,
    target: UUID,
    region: UUID,
    axis: list[np.ndarray],
    opts: MissionWorldOptions,
    seed: int,
) -> SurfaceTarget | None:
    """The target's surface OUTSIDE the defect patch, read through its own Twin2T region component."""
    i, a, b = _ends(world, target)
    pts, nrm = sample_surface(
        world, i, opts.structural.surface_samples * 2, np.random.default_rng([seed, 0x5F, 2])
    )
    rest = ~_patch_mask(pts, nrm, a, b, axis, opts)
    if int(rest.sum()) < 4:
        return None
    return SurfaceTarget(target, pts[rest], nrm[rest], twin_id=region)


def _rest_tiles(
    world: SpatialWorld,
    target: UUID,
    region: UUID,
    axis: list[np.ndarray],
    opts: MissionWorldOptions,
    seed: int,
    tiles: tuple[int, int],
) -> list[SurfaceTarget]:
    """``_rest_of_target`` split into (axial x circumferential) tiles, each its own SurfaceTarget.

    Truth side only (``StructuralSensorOptions.region_tiles``): all tiles show the same Twin2T region
    component, but a view yields one reading per visible tile at that tile's visible surface, instead of one
    reading at the mean of everything visible. Used only by scenarios that set ``region_tiles``.
    """
    n_ax, n_sec = tiles
    if n_ax < 1 or n_sec < 1:
        raise ValueError(f"region_tiles must be positive, got {tiles}")
    i, a, b = _ends(world, target)
    pts, nrm = sample_surface(
        world,
        i,
        max(opts.structural.surface_samples * 2, 12 * n_ax * n_sec),
        np.random.default_rng([seed, 0x5F, 3]),
    )
    rest = ~_patch_mask(pts, nrm, a, b, axis, opts)
    pts, nrm = pts[rest], nrm[rest]
    d = (b - a) / float(np.linalg.norm(b - a))
    u = np.cross(d, [0.0, 0.0, 1.0])
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    v = np.cross(d, u)
    t = np.clip(((pts - a) @ d) / float(np.linalg.norm(b - a)), 0.0, 1.0 - 1e-12)
    ang = np.arctan2(nrm @ v, nrm @ u) % (2.0 * math.pi)
    key = (t * n_ax).astype(int) * n_sec + np.minimum((ang / (2.0 * math.pi) * n_sec).astype(int), n_sec - 1)
    out = []
    for k in np.unique(key):
        sel = key == k
        if int(sel.sum()) >= 2:
            out.append(SurfaceTarget(target, pts[sel], nrm[sel], twin_id=region))
    return out


def _seabed_height(world: SpatialWorld, floor: UUID, lane: list[np.ndarray]) -> float:
    i = world.index_of(floor)
    xy = np.concatenate([np.linspace(p, q, 8) for p, q in itertools.pairwise(lane)])
    probe = xy.copy()
    probe[:, 2] = 0.0
    return float(np.max(probe[:, 2] - world.entity_sdf(i, probe)))


@dataclass
class MissionWorld:
    seed: int
    scenario_id: str
    opts: MissionWorldOptions
    scenario: Scenario
    t2s: Twin2S
    t2t: Twin2T
    t2e: Twin2E | None
    hardware: MissionHardware
    context: MissionContext
    mapping: RegistryMapping
    target: UUID
    recorder: MissionTruthRecorder
    store: ObjectStore
    _pending_faults: list[Any]
    _pending_eco: list[Any]

    @classmethod
    def build(
        cls, seed: int, scenario_id: str, options: MissionWorldOptions, run_id: UUID, store_root: Path
    ) -> MissionWorld:
        root = IdFactory(seed)
        opts = options
        store = ObjectStore(store_root)
        scenario = build_pipeline_inspection_scenario(
            seed, root.child("scenario"), family=opts.family, sensor_overrides=opts.sensor_overrides
        )
        assert scenario.mission is not None
        t2s = Twin2S(root.child("twin2s"), store)
        t2s.initialize(scenario)
        t2s.reset(seed)
        groups = scenario.spatial_state["groups"]
        segments = [UUID(u) for u in groups["segments"]]
        target = segments[min(opts.target_segment_index, len(segments) - 1)]
        t2t_scenario = structural_scenario(scenario, target, opts, seed)
        t2t = Twin2T(root.child("twin2t"), store)
        t2t.initialize(t2t_scenario)
        t2e: Twin2E | None = None
        if opts.ecological_enabled:
            surface = float(scenario.environment.get("water_surface_z_m", 30.0))
            t2e = Twin2E(root.child("twin2e"), store, twin2e_config(t2s.world, surface))
            t2e.initialize(ecological_scenario(scenario, t2s.world, opts, seed), seed=seed)
        axis = _pipe_axis(t2s.world, segments)
        lane = transit_lane(axis, opts.lane_offset_m, opts.lane_height_m, opts.lane_margin_m)
        rng = np.random.default_rng([seed, 0x51])
        sid = root.child("sensors")
        geometric = tuple(
            s.model_copy(
                update={
                    "mount_pose": s.mount_pose.model_copy(
                        update={
                            "orientation_wxyz": quat_from_euler(
                                0.0, 0.0, math.radians(opts.structural.mount_yaw_deg)
                            )
                        }
                    )
                }
            )
            if s.modality in opts.side_looking_geometric
            else s
            for s in scenario.robots[0].sensors
            if s.modality in ("DEPTH_RANGE", "SONAR")
        )
        so = opts.structural
        sparams = {
            "hfov_deg": so.hfov_deg,
            "vfov_deg": so.vfov_deg,
            "max_range_m": so.max_range_m,
            "min_range_m": so.min_range_m,
            "t2t_fidelity": "T0",
        }
        sensors = SuiteSensors(
            geometric=geometric,
            structural=_spec(
                sid,
                "STRUCTURED",
                "SENSOR_INSPECTION",
                1 / opts.structural_period_s,
                sparams,
                so.mount_position_m,
                so.mount_yaw_deg,
            ),
            structural_aux=_spec(
                sid,
                "STRUCTURED",
                "SENSOR_INSPECTION_AUX",
                1 / opts.structural_period_s,
                sparams,
                so.mount_position_m,
                so.mount_yaw_deg,
            ),
            environmental=_spec(sid, "ENVIRONMENTAL", "SENSOR_ENV", 1 / opts.environmental_period_s, {}),
            survey=_spec(
                sid, "STRUCTURED", "SENSOR_SURVEY", 1 / opts.survey_period_s, {"footprint_radius_m": 4.0}
            ),
            fix_sensor_id=sid.new(),
        )
        all_specs = (
            *geometric,
            sensors.structural,
            sensors.structural_aux,
            sensors.environmental,
            sensors.survey,
        )
        floor = UUID(groups["seafloor"][0])
        ctx, mapping = build_mission_context(
            scenario,
            t2t_scenario,
            target,
            all_specs,
            (sensors.structural.sensor_id, sensors.structural_aux.sensor_id),
            root.child("registry"),
            rng,
            lane,
            _seabed_height(t2s.world, floor, lane),
            opts.survey_sigma_m,
            opts.launch_sigma_m,
        )
        surf_rng = np.random.default_rng([seed, 0x5F])
        targets = [_patch(t2s.world, target, axis, opts, surf_rng)]
        region = rest_region_of(t2t_scenario, target) if REGION_READINGS else None
        tiles = opts.structural.region_tiles
        if region is not None and tiles is not None:
            targets += _rest_tiles(t2s.world, target, region, axis, opts, seed, tiles)
        else:
            rest = None if region is None else _rest_of_target(t2s.world, target, region, axis, opts, seed)
            if rest is not None:
                targets.append(rest)
        for we in scenario.world_entities:
            if (
                we.domain_ownership.technical
                and we.id != target
                and str(we.id) in scenario.spatial_state["entities"]
            ):
                pts, nrm = sample_surface(
                    t2s.world, t2s.world.index_of(we.id), opts.structural.surface_samples, surf_rng
                )
                if len(pts):
                    targets.append(SurfaceTarget(we.id, pts, nrm))
        hw = build_mission_hardware(
            load_robot_config(ROBOT_CONFIG),
            seed,
            SimKernelConfig(
                physics_dt_s=opts.physics_dt_s,
                water_surface_z_m=float(scenario.environment["water_surface_z_m"]),
            ),
            t2s.world.sdf,
            (lambda p, t: np.asarray(opts.current_mps)) if any(opts.current_mps) else None,
            t2e is not None,
        )
        start, nxt = lane[0], lane[1]
        yaw = math.atan2(float(nxt[1] - start[1]), float(nxt[0] - start[0]))
        hw.kernel.reset(start, np.asarray(quat_from_euler(0.0, 0.0, yaw)))
        recorder = MissionTruthRecorder(
            {
                "seed": seed,
                "scenario_id": scenario_id,
                "world_family": opts.family,
                "scenario_uuid": str(scenario.scenario_id),
                "target_world_id": str(target),
                "target_registry_id": str(mapping.to_registry[target]),
                "registry_to_world": {str(r): str(w) for r, w in mapping.to_world.items()},
                "world_entity_ids": sorted(str(e.id) for e in scenario.world_entities),
                "defect": opts.defect.model_dump(mode="json"),
                "patch_centre_m": [float(v) for v in targets[0].points.mean(axis=0)],
            },
        )
        suite = MissionSensorSuite(
            opts,
            t2s,
            t2t,
            t2e,
            sensors,
            targets,
            root.child("sensing"),
            np.random.default_rng([seed, 0x5E]),
            ctx.mission_id,
            run_id,
            recorder.record,
        )
        hw.attach_suite(suite)
        world = cls(
            seed,
            scenario_id,
            opts,
            scenario,
            t2s,
            t2t,
            t2e,
            hw,
            ctx,
            mapping,
            target,
            recorder,
            store,
            sorted(opts.faults, key=lambda f: f.t_s),
            sorted(opts.eco_events, key=lambda e: e.t_s),
        )
        recorder.snapshot_target("start", t2t, target, 0.0)
        return world

    # ------------------------------------------------------------------ time
    @property
    def t_s(self) -> float:
        return float(self.hardware.kernel.t_s)

    def due_faults(self, inspection_started_s: float | None = None) -> list[dict[str, Any]]:
        """Apply scheduled faults and ecological events whose time has come; returns what fired."""
        fired: list[dict[str, Any]] = []
        keep = []
        for f in self._pending_faults:
            base = 0.0 if f.trigger == "TIME" else inspection_started_s
            if base is None or base + f.t_s > self.t_s + 1e-9:
                keep.append(f)
                continue
            if f.type == "FIX_OUTAGE":
                end = self.t_s + (f.duration_s or 30.0)
                assert self.hardware.suite is not None
                self.hardware.suite.dynamic_fix_outages.append((self.t_s, end))
            else:
                self.hardware.inject_fault(FaultType(f.type), f.target, f.magnitude, f.duration_s)
            fired.append({"kind": "FAULT", "fired_at_s": round(self.t_s, 3), **f.model_dump(mode="json")})
        self._pending_faults = keep
        while self._pending_eco and self._pending_eco[0].t_s <= self.t_s + 1e-9:
            ev = self._pending_eco.pop(0)
            if self.t2e is not None:
                self.t2e.step(1e-3, [_eco_event(ev.event_type, ev.parameters, self.t_s)])
            fired.append({"kind": "ECOLOGICAL_EVENT", "event_type": ev.event_type, "t_s": ev.t_s})
        return fired

    def advance(self, dt_s: float) -> None:
        self.hardware.advance(dt_s)
        self.t2t.step(dt_s)
        if self.t2e is not None:
            lag = self.t_s - self.t2e.t_s
            if lag > 1e-6:
                self.t2e.step(lag)
        self.recorder.track(self.hardware.truth_access())


def _eco_event(event_type: str, params: dict[str, Any], t_s: float) -> Any:
    from conrad.schemas.world import ScenarioEvent

    return ScenarioEvent(
        event_id=IdFactory(int(t_s * 1000) + 17).new(), time_s=t_s, event_type=event_type, parameters=params
    )
