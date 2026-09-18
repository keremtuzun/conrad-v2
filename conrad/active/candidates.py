"""Candidate viewpoint generation and feasibility filtering (ch16/ch18 'Candidate generation', ch33 MCBR).

Candidates are rings / hemispheres around the target region at sensor-valid standoff ranges, one per
(sensor, configuration, standoff, azimuth, elevation). Everything the generator and the filter know
about the world arrives through injected callables computed from BELIEF (e.g. Model 2S occupancy):

    is_free(points[N,3]) -> bool[N]
    predicted_visibility(pose, target_region) -> float in [0, 1]
    navigation_cost(from_pose, to_pose) -> ResourceCost | None   (None = no route in the belief map)

Feasibility and safety filtering happens BEFORE ranking and every rejection keeps its reason codes.

implementation_status: FROZEN_CONTRACT (filter-before-rank) / EXPERIMENTAL_CANDIDATE (sampler)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np
from pydantic import Field

from conrad.active.config import MCBRConfig
from conrad.schemas.base import ConradModel
from conrad.schemas.decision import InformationNeed, ResourceCost
from conrad.schemas.frames import Pose, SpatialSupport, Vec3, quat_from_euler

IsFree = Callable[[np.ndarray], np.ndarray]
PredictedVisibility = Callable[[Pose, SpatialSupport], float]
NavigationCost = Callable[[Pose, Pose], ResourceCost | None]

R_POSE_NOT_FREE = "POSE_NOT_FREE"
R_UNREACHABLE = "UNREACHABLE_IN_BELIEF_MAP"
R_VISIBILITY = "VISIBILITY_BELOW_MIN"
R_RISK = "RISK_LIMIT_EXCEEDED"
R_ENERGY = "ENERGY_BUDGET_EXCEEDED"
R_DEADLINE = "DEADLINE_EXCEEDED"
R_BOUNDARY = "OUTSIDE_MISSION_BOUNDARY"
R_MODALITY = "MODALITY_NOT_ALTERNATE"
R_SENSOR_UNAVAILABLE = "SENSOR_UNAVAILABLE"


class SensorOption(ConradModel):
    """What MCBR may select: (sensor, configuration). Quality numbers are configuration, not measurements."""

    sensor_id: UUID
    modality: str
    min_range_m: float = Field(gt=0)
    max_range_m: float = Field(gt=0)
    duration_s: float = Field(default=2.0, gt=0)
    power_w: float = Field(default=5.0, ge=0)
    measurement_quality: float = Field(
        default=0.7, ge=0, le=1, description="ability to beat measurement noise"
    )
    discrimination: dict[str, float] = Field(
        default_factory=dict,
        description="'H1|H2' (sorted) or 'default' -> separability of outcomes in [0, 1]",
    )
    configurations: tuple[dict[str, Any], ...] = ({},)
    available: bool = True

    def separability(self, hypotheses: tuple[str, ...]) -> float:
        if len(hypotheses) < 2:
            return self.discrimination.get("default", 0.0)
        pairs = [
            self.discrimination.get("|".join(sorted((a, b))), self.discrimination.get("default", 0.0))
            for i, a in enumerate(hypotheses)
            for b in hypotheses[i + 1 :]
        ]
        return min(pairs)


@dataclass(frozen=True)
class RawCandidate:
    pose: Pose
    sensor: SensorOption
    configuration: dict[str, Any]
    standoff_m: float
    azimuth_rad: float
    elevation_rad: float
    index: int


@dataclass(frozen=True)
class MissionBounds:
    min_m: Vec3
    max_m: Vec3
    energy_budget_j: float | None = None
    now_ns: int = 0


def look_at(position: Vec3, target: Vec3, frame_id: str) -> Pose:
    dx, dy, dz = (target[i] - position[i] for i in range(3))
    yaw = math.atan2(dy, dx)
    pitch = -math.atan2(dz, math.hypot(dx, dy))
    return Pose(frame_id=frame_id, position_m=position, orientation_wxyz=quat_from_euler(0.0, pitch, yaw))


class ViewpointGenerator:
    def __init__(self, config: MCBRConfig) -> None:
        self.config = config

    def generate(self, region: SpatialSupport, sensors: tuple[SensorOption, ...]) -> list[RawCandidate]:
        c = self.config
        raw: list[tuple[SensorOption, dict[str, Any], float, float, float]] = []
        surface = max(region.half_extent_m)
        for sensor in sensors:
            span = max(sensor.max_range_m - sensor.min_range_m, 0.0)
            for cfg in sensor.configurations:
                for f in c.standoff_fractions:
                    standoff = sensor.min_range_m + f * span
                    for elevation in c.elevations_rad:
                        for k in range(c.n_azimuth):
                            raw.append((sensor, cfg, standoff, 2 * math.pi * k / c.n_azimuth, elevation))
        if len(raw) > c.max_candidates:  # deterministic even thinning, never random
            step = len(raw) / c.max_candidates
            raw = [raw[int(i * step)] for i in range(c.max_candidates)]
        out = []
        for i, (sensor, cfg, standoff, az, el) in enumerate(raw):
            r = standoff + surface
            pos = (
                region.center_m[0] + r * math.cos(el) * math.cos(az),
                region.center_m[1] + r * math.cos(el) * math.sin(az),
                region.center_m[2] + r * math.sin(el),
            )
            out.append(
                RawCandidate(
                    look_at(pos, region.center_m, region.frame_id), sensor, dict(cfg), standoff, az, el, i
                )
            )
        return out


class FeasibilityFilter:
    def __init__(self, config: MCBRConfig) -> None:
        self.config = config

    def reasons(
        self,
        candidate: RawCandidate,
        free: bool,
        visibility: float,
        cost: ResourceCost | None,
        need: InformationNeed,
        bounds: MissionBounds | None,
    ) -> tuple[str, ...]:
        out: list[str] = []
        if not candidate.sensor.available:
            out.append(R_SENSOR_UNAVAILABLE)
        if not free:
            out.append(R_POSE_NOT_FREE)
        if bounds is not None and any(
            not (bounds.min_m[i] <= candidate.pose.position_m[i] <= bounds.max_m[i]) for i in range(3)
        ):
            out.append(R_BOUNDARY)
        if cost is None:
            out.append(R_UNREACHABLE)
        else:
            if cost.risk > self.config.risk_limit:
                out.append(R_RISK)
            if (
                bounds is not None
                and bounds.energy_budget_j is not None
                and cost.energy_j > bounds.energy_budget_j
            ):
                out.append(R_ENERGY)
            if (
                bounds is not None
                and need.deadline_ns is not None
                and bounds.now_ns + int(cost.time_s * 1e9) > need.deadline_ns
            ):
                out.append(R_DEADLINE)
        if visibility < self.config.min_visibility:
            out.append(R_VISIBILITY)
        if need.constraints.get("require_alternate_modality"):
            allowed = need.constraints.get("alternate_modalities") or []
            if candidate.sensor.modality not in allowed:
                out.append(R_MODALITY)
        return tuple(out)
