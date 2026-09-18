"""Twin2S: canonical spatial truth + visibility oracle + sensor generation (ch14/15). TRUTH PLANE.

implementation_status: FROZEN_CONTRACT (interface) / EXPERIMENTAL_CANDIDATE (sensor models, OCPWE)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import numpy as np

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.truth import TruthState
from conrad.schemas.world import Domain, Scenario, ScenarioEvent, SensorSpec
from conrad.twins.base import SensingContext, Twin, TwinSample
from conrad.twins.twin2s.config import Twin2SConfig
from conrad.twins.twin2s.families import GENERATOR_VERSION
from conrad.twins.twin2s.observe import PoseHistory, make_observation, make_supervision
from conrad.twins.twin2s.octree import OctreeExport, export_octree
from conrad.twins.twin2s.raycast import sensor_world_pose
from conrad.twins.twin2s.sdf import Arr, _t3, _v
from conrad.twins.twin2s.sensors import check_degradation, render_depth, render_rgb, render_sonar
from conrad.twins.twin2s.visibility import VisibilityOracle, VisibilityResult
from conrad.twins.twin2s.world import EVENT_APPEAR, EVENT_DISPLACE, EVENT_REMOVE, SpatialWorld

_RENDERERS = {Modality.DEPTH_RANGE: render_depth, Modality.RGB: render_rgb, Modality.SONAR: render_sonar}


class Twin2S(Twin):
    domain = Domain.SPATIAL
    twin_version = GENERATOR_VERSION

    def __init__(self, ids: IdFactory, store: ObjectStore, cfg: Twin2SConfig | None = None) -> None:
        super().__init__(ids, store)
        self.cfg = cfg or Twin2SConfig()
        self._scenario: Scenario | None = None
        self._world: SpatialWorld | None = None
        self._noise_seed = 0
        self._imu: dict[UUID, PoseHistory] = {}
        self.change_log: list[dict[str, Any]] = []

    # ---- lifecycle ------------------------------------------------------------------------------
    def initialize(self, scenario: Scenario) -> None:
        """Reads ``spatial_state`` plus shared context (environment, entity registry)."""
        if "entities" not in scenario.spatial_state or "world_bounds" not in scenario.spatial_state:
            raise ValueError("scenario has no Twin2S spatial_state section")
        self._scenario = scenario
        self._world = SpatialWorld.from_spatial_state(scenario.spatial_state)
        self._noise_seed = scenario.seed
        self._imu = {}
        self.change_log = []

    def reset(self, seed: int) -> None:
        """Restore the initial world; ``seed`` selects the sensor-noise stream."""
        self.initialize(self.scenario)
        self._noise_seed = seed

    @property
    def scenario(self) -> Scenario:
        if self._scenario is None:
            raise RuntimeError("Twin2S used before initialize()")
        return self._scenario

    @property
    def world(self) -> SpatialWorld:
        if self._world is None:
            raise RuntimeError("Twin2S used before initialize()")
        return self._world

    @property
    def lineage(self) -> str:
        m = self.scenario.metadata
        return str(m.get("lineage", f"{m.get('world_family', 'unknown')}/{self.scenario.seed}"))

    # ---- dynamics -------------------------------------------------------------------------------
    def step(self, dt_s: float, events: Sequence[ScenarioEvent] = ()) -> None:
        """Advance physical time; moving entities follow their motion; events make persistent changes."""
        if dt_s < 0:
            raise ValueError("dt_s must be >= 0")
        w = self.world
        w.time_s += dt_s
        for ev in events:
            if ev.event_type not in (EVENT_APPEAR, EVENT_REMOVE, EVENT_DISPLACE):
                continue  # other twins' events
            if ev.target_entity_id is None:
                raise ValueError(f"spatial event {ev.event_id} has no target entity")
            ent = w.entities[w.index_of(ev.target_entity_id)]
            if ev.event_type == EVENT_APPEAR:
                w.replace_entity(ev.target_entity_id, active=True)
            elif ev.event_type == EVENT_REMOVE:
                w.replace_entity(ev.target_entity_id, active=False)
            else:
                new = _v(ent.displacement_m) + _v(ev.parameters["offset_m"])
                w.replace_entity(ev.target_entity_id, displacement_m=_t3(new))
            self.change_log.append(
                {
                    "time_s": w.time_s,
                    "event_id": str(ev.event_id),
                    "event_type": ev.event_type,
                    "entity_id": str(ev.target_entity_id),
                    "parameters": dict(ev.parameters),
                }
            )

    # ---- truth ----------------------------------------------------------------------------------
    def get_truth(self, timestamp: TimeStamp) -> list[TruthState]:
        w = self.world
        rel: dict[UUID, list[UUID]] = {}
        for a, b, _ in self.scenario.spatial_state.get("topology", {}).get("adjacency", []):
            rel.setdefault(UUID(a), []).append(UUID(b))
            rel.setdefault(UUID(b), []).append(UUID(a))
        out = []
        for i, e in enumerate(w.entities):
            lo, hi = e.primitive.bounds()
            off = w.entity_offset(i)
            vel = e.motion.velocity(w.time_s) if e.motion is not None else np.zeros(3)
            state = {
                **e.to_dict(),
                "offset_m": off.tolist(),
                "velocity_mps": vel.tolist(),
                "bounds_min_m": (lo + off).tolist(),
                "bounds_max_m": (hi + off).tolist(),
                "dynamic": e.motion is not None,
                "frame_id": "WORLD",
                "time_s": w.time_s,
            }
            out.append(
                TruthState(
                    scenario_id=self.scenario.scenario_id,
                    world_entity_id=e.entity_id,
                    domain=Domain.SPATIAL,
                    timestamp=timestamp,
                    state=state,
                    relationships=tuple(rel.get(e.entity_id, [])),
                    generator_version=GENERATOR_VERSION,
                )
            )
        return out

    def export_domain_state(self) -> dict[str, Any]:
        return {
            "generator_version": GENERATOR_VERSION,
            "time_s": self.world.time_s,
            "world_bounds": self.scenario.spatial_state["world_bounds"],
            "entities": self.world.entities_state(),
            "change_log": list(self.change_log),
        }

    def export_octree(self, bounds_min: Any = None, bounds_max: Any = None) -> OctreeExport:
        return export_octree(self.world, self.cfg.octree, bounds_min, bounds_max)

    def occupancy_truth_at(self, points_m: Arr) -> np.ndarray:
        """True occupancy (sdf < 0) at WORLD points at the current physical time."""
        return self.world.occupied(np.atleast_2d(points_m))

    def visibility(
        self, sensor: SensorSpec, true_robot_pose: Pose, points_m: Arr, normals: Arr | None = None
    ) -> VisibilityResult:
        """Observability truth V(x, s, p, t) for WORLD points from the TRUE pose."""
        turb = float(self.scenario.environment.get("turbidity", 0.0))
        return VisibilityOracle(self.world, self.cfg).visibility(
            sensor, true_robot_pose, points_m, normals, turb
        )

    # ---- sensing --------------------------------------------------------------------------------
    def _rng(self, ctx: SensingContext) -> np.random.Generator:
        key = [
            self._noise_seed,
            ctx.sensor.sensor_id.int,
            ctx.timestamp.time_ns,
            ctx.timestamp.sequence_index,
        ]
        return np.random.default_rng(np.random.SeedSequence(key))

    def generate_observation(self, ctx: SensingContext) -> list[TwinSample]:
        check_degradation(dict(ctx.degradation))
        if ctx.true_pose.frame_id != "WORLD":
            raise ValueError("Twin2S expects the true robot pose in WORLD")
        if ctx.degradation.get("fault", 0.0) >= 1.0:
            return []  # a faulted sensor senses nothing; nothing is fabricated
        modality = Modality(ctx.sensor.modality)
        deg = dict(ctx.degradation)
        if "turbidity" not in deg:
            deg["turbidity"] = float(self.scenario.environment.get("turbidity", 0.0))
        rng = self._rng(ctx)
        table = [e.entity_id for e in self.world.entities]
        if modality in _RENDERERS:
            r = _RENDERERS[modality](self.world, ctx.sensor, ctx.true_pose, deg, rng, self.cfg.raycast)
            samples = []
            main_key = {"DEPTH_RANGE": "range_image", "RGB": "image", "SONAR": "polar_image"}[modality.value]
            enc = {
                "layout": "HxW" if modality != Modality.SONAR else "range_bins x beams",
                "units": "m" if modality == Modality.DEPTH_RANGE else "relative",
                "missing": "NaN",
            }
            obs = make_observation(
                self.ids, ctx, modality, self.store.put_array(r.measured[main_key]), encoding=enc
            )
            tr = {k: v for k, v in r.truth.items() if not k.startswith("point_")}
            samples.append(TwinSample(obs, make_supervision(self.store, obs, tr, table, ctx, self.lineage)))
            if modality == Modality.DEPTH_RANGE and ctx.sensor.parameters.get("emit_point_cloud", False):
                pc = make_observation(
                    self.ids,
                    ctx,
                    Modality.POINT_CLOUD,
                    self.store.put_array(r.measured["points_sensor"]),
                    encoding={"layout": "Nx3", "frame": ctx.sensor.frame_id, "units": "m"},
                )
                ptr = {
                    "entity_index": r.truth["point_entity_index"],
                    "true_range": r.truth["point_true_range"],
                }
                samples.append(
                    TwinSample(pc, make_supervision(self.store, pc, ptr, table, ctx, self.lineage))
                )
            return samples
        if modality == Modality.PRESSURE_DEPTH:
            _, origin = sensor_world_pose(ctx.true_pose, ctx.sensor.mount_pose)
            depth = float(self.scenario.environment["water_surface_z_m"]) - float(origin[2])
            sigma = float(ctx.sensor.parameters["depth_noise_sigma_m"])
            meas = depth + float(rng.normal(0.0, sigma)) + ctx.degradation.get("calibration_bias_m", 0.0)
            obs = make_observation(self.ids, ctx, modality, inline=(meas,), units="m")
            return [
                TwinSample(
                    obs, make_supervision(self.store, obs, {}, [], ctx, self.lineage, {"true_depth_m": depth})
                )
            ]
        if modality == Modality.IMU:
            hist = self._imu.setdefault(ctx.sensor.sensor_id, PoseHistory([]))
            hist.push(ctx.timestamp.seconds, ctx.true_pose)
            got = hist.imu()
            if got is None:
                return []  # not enough motion history to synthesise inertial data
            f, w = got
            p = ctx.sensor.parameters
            fm = f + rng.normal(0.0, float(p["accel_noise_sigma_mps2"]), 3)
            wm = w + rng.normal(0.0, float(p["gyro_noise_sigma_rps"]), 3)
            obs = make_observation(
                self.ids,
                ctx,
                modality,
                inline=tuple(float(x) for x in (*fm, *wm)),
                units="m/s^2 x3, rad/s x3 (body frame)",
            )
            extra = {"true_specific_force_mps2": f.tolist(), "true_angular_rate_rps": w.tolist()}
            return [TwinSample(obs, make_supervision(self.store, obs, {}, [], ctx, self.lineage, extra))]
        return []  # modality not produced by Twin2S (e.g. POWER, ENVIRONMENTAL)
