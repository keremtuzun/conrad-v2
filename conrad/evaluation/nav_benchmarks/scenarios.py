"""Scenario loading for NAV-001..008: geometry (SDF), currents, fault schedule, synthetic position fixes.

Evaluation-side code: it may build truth-side artefacts (SDF, renderer fed with the true pose).
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from conrad.robotics.estimation.interface import POSITION_FIX_KIND
from conrad.schemas.frames import WORLD, Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.timebase import TimeStamp
from conrad.settings import REPO_ROOT

BENCHMARK_CONFIG = REPO_ROOT / "configs" / "sim" / "nav_benchmarks.yaml"
BENCHMARK_IDS = tuple(f"NAV-00{i}" for i in range(1, 9))


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_scenario(benchmark_id: str, path: Path = BENCHMARK_CONFIG) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if benchmark_id not in data["scenarios"]:
        raise KeyError(f"unknown benchmark {benchmark_id!r}; known: {sorted(data['scenarios'])}")
    return _merge(data["defaults"], data["scenarios"][benchmark_id])


def _segment_distance(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance of points p[N,3] to a polyline poly[M,3]."""
    best = np.full(len(p), np.inf)
    for a, b in itertools.pairwise(poly):
        ab = b - a
        t = np.clip(((p - a) @ ab) / float(ab @ ab), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return best


@dataclass
class ScenarioGeometry:
    obstacles: list[dict[str, Any]]
    pipeline: np.ndarray | None

    def sdf(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        d = np.full(len(p), 1e6)
        for ob in self.obstacles:
            if ob["type"] == "vertical_cylinder":
                c = np.asarray(ob["center_xy"], dtype=np.float64)
                d = np.minimum(d, np.linalg.norm(p[:, :2] - c, axis=1) - float(ob["radius_m"]))
            elif ob["type"] == "pipe":
                if self.pipeline is None:
                    raise ValueError("pipe obstacle needs a PIPELINE_FOLLOW polyline")
                d = np.minimum(d, _segment_distance(p, self.pipeline) - float(ob["radius_m"]))
            elif ob["type"] == "seabed":
                d = np.minimum(d, p[:, 2] - float(ob["z_m"]))
            else:
                raise ValueError(f"unknown obstacle type {ob['type']!r}")
        return d

    @property
    def empty(self) -> bool:
        return not self.obstacles


def geometry_of(scenario: dict[str, Any]) -> ScenarioGeometry:
    goal = scenario["goal"]
    poly = np.asarray(goal["polyline"], dtype=np.float64) if goal["primitive"] == "PIPELINE_FOLLOW" else None
    return ScenarioGeometry(list(scenario.get("obstacles", [])), poly)


class SyntheticPositionFixRenderer:
    """USBL-like WORLD position fix from the true pose (+ Gaussian noise), with scheduled outages."""

    def __init__(
        self,
        mission_id: Any,
        run_id: Any,
        ids: IdFactory,
        rng: np.random.Generator,
        period_s: float,
        sigma_m: float,
        outages: list[list[float]],
    ) -> None:
        self._mission, self._run, self._ids, self._rng = mission_id, run_id, ids, rng
        self.period_s, self.sigma_m = period_s, sigma_m
        self.outages = [(float(a), float(b)) for a, b in outages]
        self._next_s = 0.0
        self.trace_id = ids.new()
        self.sensor_id = ids.new()
        self.delivered = 0

    def __call__(
        self, sensor_name: str, true_pose: Pose, estimated_pose: Pose | None, timestamp: TimeStamp
    ) -> Observation | None:
        t = timestamp.seconds
        if sensor_name != "sonar" or t + 1e-9 < self._next_s:
            return None
        self._next_s = t + self.period_s
        if any(a <= t < b for a, b in self.outages):
            return None
        noisy = np.asarray(true_pose.position_m) + self.sigma_m * self._rng.standard_normal(3)
        self.delivered += 1
        return Observation(
            observation_id=self._ids.new(),
            mission_id=self._mission,
            run_id=self._run,
            trace_id=self.trace_id,
            sensor_id=self.sensor_id,
            modality=Modality.STRUCTURED,
            timestamp=timestamp,
            sensor_frame=WORLD,
            robot_pose_estimate=estimated_pose,
            inline_values=(float(noisy[0]), float(noisy[1]), float(noisy[2])),
            inline_units="m",
            sensor_context={
                "kind": POSITION_FIX_KIND,
                "frame_id": WORLD,
                "sigma_m": self.sigma_m,
                "source": "SYNTHETIC_USBL_LIKE",
            },
        )
