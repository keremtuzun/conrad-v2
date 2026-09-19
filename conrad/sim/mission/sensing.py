"""MissionSensorSuite: payload observations rendered from the three twins at the TRUE pose. TRUTH PLANE.

Every emitted Observation carries only the ESTIMATED pose (from the deployment's pose provider). Structural
inspection readings come from Twin2T with visibility from the Twin2S oracle on surface samples (for the
target segment: the far-side defect patch), turbidity/biofouling from Twin2E, and a noisy sensor-frame
range/bearing/elevation to the observed surface. Samples are emitted in observation-ID order, never in
world-entity order, and no world-entity ID reaches an Observation.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np

from conrad.robotics.estimation.interface import POSITION_FIX_KIND
from conrad.schemas.frames import WORLD, FramedPoint, Pose, quat_to_matrix
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import SensorSpec
from conrad.sim.mission.options import MissionWorldOptions
from conrad.twins.base import SensingContext
from conrad.twins.twin2e import Twin2E
from conrad.twins.twin2s.twin import Twin2S
from conrad.twins.twin2t import Twin2T

Recorder = Callable[[str, dict[str, Any]], None]


@dataclass
class SurfaceTarget:
    world_id: UUID
    points: np.ndarray
    normals: np.ndarray
    is_patch: bool = False


@dataclass
class SuiteSensors:
    geometric: tuple[SensorSpec, ...]
    structural: SensorSpec
    structural_aux: SensorSpec
    environmental: SensorSpec
    survey: SensorSpec
    fix_sensor_id: UUID


@dataclass
class _Due:
    period_s: float
    next_s: float = 0.0

    def due(self, t: float) -> bool:
        if t + 1e-9 < self.next_s:
            return False
        self.next_s = t + self.period_s
        return True


@dataclass
class MissionSensorSuite:
    opts: MissionWorldOptions
    t2s: Twin2S
    t2t: Twin2T
    t2e: Twin2E | None
    sensors: SuiteSensors
    targets: list[SurfaceTarget]
    ids: IdFactory
    rng: np.random.Generator
    mission_id: UUID
    run_id: UUID
    record: Recorder
    _due: dict[str, _Due] = field(default_factory=dict)
    dynamic_fix_outages: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        o = self.opts
        self._due = {
            "geometric": _Due(o.geometric_period_s, 0.5),
            "structural": _Due(o.structural_period_s, 0.25),
            "environmental": _Due(o.environmental_period_s, 1.0),
            "survey": _Due(o.survey_period_s, 1.5),
            "fix": _Due(o.fix.period_s, 0.0),
        }

    # ------------------------------------------------------------------ entry point
    def collect(self, t: float, stamp: TimeStamp, true_pose: Pose, est: Pose | None) -> list[Observation]:
        trace = self.ids.new()
        out: list[Observation] = []
        if self._due["fix"].due(t):
            out += self._fix(t, stamp, true_pose, est, trace)
        if self._due["geometric"].due(t):
            out += self._geometric(t, stamp, true_pose, est, trace)
        if self._due["structural"].due(t):
            out += self._structural(t, stamp, true_pose, est, trace)
        if self.t2e is not None and self._due["environmental"].due(t):
            out += self._ecological(self.sensors.environmental, true_pose, est, trace)
        if self.t2e is not None and self._due["survey"].due(t):
            out += self._ecological(self.sensors.survey, true_pose, est, trace)
        return sorted(out, key=lambda ob: ob.observation_id.int)

    def _ctx(
        self,
        sensor: SensorSpec,
        stamp: TimeStamp,
        true: Pose,
        est: Pose | None,
        trace: UUID,
        deg: dict[str, float],
    ) -> SensingContext:
        return SensingContext(
            mission_id=self.mission_id,
            run_id=self.run_id,
            trace_id=trace,
            sensor=sensor,
            true_pose=true,
            estimated_pose=est,
            timestamp=stamp,
            degradation=deg,
        )

    # ------------------------------------------------------------------ channels
    def _fix(
        self, t: float, stamp: TimeStamp, true: Pose, est: Pose | None, trace: UUID
    ) -> list[Observation]:
        if any(a <= t < b for a, b in (*self.opts.fix.outages, *self.dynamic_fix_outages)):
            self.record("fix_outage", {"t_s": t})
            return []
        noisy = np.asarray(true.position_m) + self.opts.fix.sigma_m * self.rng.standard_normal(3)
        return [
            Observation(
                observation_id=self.ids.new(),
                mission_id=self.mission_id,
                run_id=self.run_id,
                trace_id=trace,
                sensor_id=self.sensors.fix_sensor_id,
                modality=Modality.STRUCTURED,
                timestamp=stamp,
                sensor_frame=WORLD,
                robot_pose_estimate=est,
                inline_values=(float(noisy[0]), float(noisy[1]), float(noisy[2])),
                inline_units="m",
                sensor_context={
                    "kind": POSITION_FIX_KIND,
                    "frame_id": WORLD,
                    "sigma_m": self.opts.fix.sigma_m,
                    "source": "SYNTHETIC_USBL_LIKE",
                },
            )
        ]

    def _geometric(
        self, t: float, stamp: TimeStamp, true: Pose, est: Pose | None, trace: UUID
    ) -> list[Observation]:
        deg = dict(self.opts.geometric_degradation)
        if any(a <= t < b for a, b in self.opts.geometric_dropout):
            deg["fault"] = 1.0
            self.record("geometric_dropout", {"t_s": t})
        out: list[Observation] = []
        if deg.get("fault", 0.0) < 1.0:
            self.record(
                "geometric_capture",
                {
                    "t_s": t,
                    "position_m": list(true.position_m),
                    "orientation_wxyz": list(true.orientation_wxyz),
                },
            )
        for spec in self.sensors.geometric:
            out += [
                s.observation
                for s in self.t2s.generate_observation(self._ctx(spec, stamp, true, est, trace, deg))
            ]
        return out

    def _structural(
        self, t: float, stamp: TimeStamp, true: Pose, est: Pose | None, trace: UUID
    ) -> list[Observation]:
        spec = self.sensors.structural
        rot, origin = _sensor_pose(true, spec)
        out: list[Observation] = []
        contra = self.opts.contradiction
        for tgt in self.targets:
            vis = self.t2s.visibility(spec, true, tgt.points, tgt.normals)
            mask = np.asarray(vis.visible, dtype=bool)
            if tgt.is_patch:
                frac = float(mask.mean()) if len(mask) else 0.0
            else:  # fraction of the component surface that faces the sensor and is actually seen
                facing = np.einsum("ij,ij->i", tgt.normals, vis.view_dirs) > 0.0
                frac = float(mask.sum() / max(int(facing.sum()), 1))
            if tgt.is_patch:
                self.record("patch_visibility", {"t_s": t, "visible_fraction": frac})
            if frac <= 0.0:
                continue
            point = tgt.points[mask].mean(axis=0)
            deg = {
                f"visibility:{tgt.world_id}": frac,
                **self._quality(point),
                **self.opts.structural.degradation,
            }
            readings = [(spec, deg)]
            if contra.enabled and t >= contra.start_s:
                aux = {
                    **deg,
                    "contradiction": contra.contradiction,
                    "sensor_bias_m": contra.sensor_bias_m,
                    "corruption": max(contra.corruption, deg.get("corruption", 0.0)),
                }
                readings.append((self.sensors.structural_aux, aux))
            for sensor, d in readings:
                for sample in self.t2t.generate_observation(self._ctx(sensor, stamp, true, est, trace, d)):
                    obs = self._with_geometry(sample.observation, rot, origin, point)
                    assert sample.supervision is not None
                    self.record(
                        "structural_label",
                        {
                            "t_s": t,
                            "observation_id": str(obs.observation_id),
                            "world_id": str(tgt.world_id),
                            "visible_fraction": frac,
                            "patch": tgt.is_patch,
                            "aux": sensor is not spec,
                        },
                    )
                    out.append(obs)
        return out

    def _quality(self, point: np.ndarray) -> dict[str, float]:
        if self.t2e is None:
            return {}
        m = self.t2e.observability_modifiers(
            FramedPoint(frame_id=WORLD, xyz_m=tuple(float(v) for v in point))
        )
        full = self.opts.structural.turbidity_ntu_full_scale
        return {
            "turbidity": float(np.clip(float(m["turbidity_ntu"]) / full, 0.0, 1.0)),
            "biofouling_cover": float(np.clip(float(m["biofouling_cover"]), 0.0, 1.0)),
        }

    def _with_geometry(
        self, obs: Observation, rot: np.ndarray, origin: np.ndarray, point: np.ndarray
    ) -> Observation:
        local = rot.T @ (point - origin)
        s = self.opts.structural
        rng = float(np.linalg.norm(local)) + s.range_sigma_m * float(self.rng.standard_normal())
        bearing = math.atan2(local[1], local[0]) + s.angle_sigma_rad * float(self.rng.standard_normal())
        elev = math.atan2(local[2], math.hypot(local[0], local[1])) + s.angle_sigma_rad * float(
            self.rng.standard_normal()
        )
        ctx = dict(obs.sensor_context)
        ctx.update(
            measured_range_m=max(rng, 0.0),
            measured_bearing_rad=bearing,
            measured_elevation_rad=elev,
            range_sigma_m=s.range_sigma_m,
            angle_sigma_rad=s.angle_sigma_rad,
        )
        return obs.model_copy(update={"sensor_context": ctx})

    def _ecological(self, sensor: SensorSpec, true: Pose, est: Pose | None, trace: UUID) -> list[Observation]:
        assert self.t2e is not None
        ctx = self._ctx(sensor, self.t2e.now(), true, est, trace, {})
        return [s.observation for s in self.t2e.generate_observation(ctx)]


def _sensor_pose(pose: Pose, sensor: SensorSpec) -> tuple[np.ndarray, np.ndarray]:
    r_wb = quat_to_matrix(pose.orientation_wxyz)
    r_bs = quat_to_matrix(sensor.mount_pose.orientation_wxyz)
    origin = np.asarray(pose.position_m) + r_wb @ np.asarray(sensor.mount_pose.position_m)
    return r_wb @ r_bs, origin
