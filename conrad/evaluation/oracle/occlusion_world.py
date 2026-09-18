"""Abstract 3D occlusion world for ACTIVE-MCBR-E001. TRUTH PLANE (evaluation only). SYNTHETIC_ONLY.

A cylindrical target at the origin has ``n_sectors`` surface sectors with hidden defect values; the
mission-relevant weld sectors face +x. Spherical occluders block lines of sight. A binary hidden
hypothesis (H1 surface anomaly vs H2 material loss) can be discriminated well by SONAR and poorly by
RGB. OOD turbidity makes RGB useless. The planner never sees this object: it only sees the belief
built by the experiment, and the oracle scores what the chosen observation ACTUALLY did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TARGET_RADIUS_M = 1.0


@dataclass(frozen=True)
class SensorModel:
    modality: str
    min_range_m: float
    max_range_m: float
    noise_std: float
    range_noise_per_m: float
    hypothesis_accuracy: float  # P(outcome points to the true hypothesis)
    ood_noise_multiplier: float = 1.0


@dataclass
class OcclusionWorld:
    n_sectors: int
    defects: np.ndarray  # [S] hidden defect values in [0, 1]
    weld_sectors: tuple[int, ...]
    occluders: list[tuple[np.ndarray, float]]  # (centre, radius)
    hypothesis_h2: bool
    turbid: bool
    sensors: dict[str, SensorModel] = field(default_factory=dict)

    def sector_normal(self, s: int) -> np.ndarray:
        a = 2 * math.pi * s / self.n_sectors
        return np.array([math.cos(a), math.sin(a), 0.0])

    def sector_point(self, s: int) -> np.ndarray:
        return TARGET_RADIUS_M * self.sector_normal(s)


def sample_world(rng: np.random.Generator, n_sectors: int = 9, n_occluders: int = 3) -> OcclusionWorld:
    defects = rng.uniform(0.0, 1.0, n_sectors)
    weld = tuple(s for s in range(n_sectors) if math.cos(2 * math.pi * s / n_sectors) > 0.7)
    occluders = []
    for _ in range(n_occluders):
        r = rng.uniform(2.0, 4.0)
        a = rng.uniform(-math.pi, math.pi)
        occluders.append(
            (np.array([r * math.cos(a), r * math.sin(a), rng.uniform(-0.5, 0.5)]), rng.uniform(0.4, 0.8))
        )
    sensors = {
        "RGB": SensorModel("RGB", 0.5, 4.0, 0.12, 0.05, 0.55, ood_noise_multiplier=6.0),
        "SONAR": SensorModel("SONAR", 1.0, 8.0, 0.18, 0.01, 0.85),
    }
    return OcclusionWorld(
        n_sectors, defects, weld, occluders, bool(rng.random() < 0.5), bool(rng.random() < 0.5), sensors
    )


def segment_blocked(p: np.ndarray, q: np.ndarray, occluders: list[tuple[np.ndarray, float]]) -> bool:
    d = q - p
    length2 = float(d @ d)
    for c, r in occluders:
        t = 0.0 if length2 == 0 else float(np.clip((c - p) @ d / length2, 0.0, 1.0))
        if float(np.linalg.norm(p + t * d - c)) < r:
            return True
    return False


def visible_sectors(
    world: OcclusionWorld,
    position: np.ndarray,
    sensor: SensorModel,
    occluders: list[tuple[np.ndarray, float]],
) -> list[int]:
    out = []
    for s in range(world.n_sectors):
        pt = world.sector_point(s)
        v = position - pt
        dist = float(np.linalg.norm(v))
        if not (sensor.min_range_m <= dist <= sensor.max_range_m):
            continue
        if float(world.sector_normal(s) @ v) / max(dist, 1e-9) < math.cos(math.radians(70)):
            continue
        if not segment_blocked(position, pt, occluders):
            out.append(s)
    return out


@dataclass
class ObservationOutcome:
    sectors: list[int]
    values: list[float]
    noise_std: list[float]
    hypothesis_vote_h2: bool | None
    hypothesis_accuracy: float


def observe(
    world: OcclusionWorld, position: np.ndarray, modality: str, rng: np.random.Generator
) -> ObservationOutcome:
    """Execute an observation against TRUTH (true occluders, true defects, true hypothesis)."""
    sensor = world.sensors[modality]
    sectors = visible_sectors(world, position, sensor, world.occluders)
    values, stds = [], []
    for s in sectors:
        dist = float(np.linalg.norm(position - world.sector_point(s)))
        std = sensor.noise_std + sensor.range_noise_per_m * dist
        if world.turbid and modality == "RGB":
            std *= sensor.ood_noise_multiplier
        stds.append(std)
        values.append(float(world.defects[s] + rng.normal(0.0, std)))
    vote = None
    acc = sensor.hypothesis_accuracy
    if world.turbid and modality == "RGB":
        acc = 0.5
    if any(s in world.weld_sectors for s in sectors):
        vote = world.hypothesis_h2 if rng.random() < acc else not world.hypothesis_h2
    return ObservationOutcome(sectors, values, stds, vote, acc)


def belief_error(world: OcclusionWorld, mean: np.ndarray, p_h2: float, mission_only: bool) -> float:
    """Actual hidden-state error of a belief (never uncertainty). Mission = weld sectors + hypothesis."""
    idx = list(world.weld_sectors) if mission_only else list(range(world.n_sectors))
    err = float(np.mean(np.abs(mean[idx] - world.defects[idx])))
    return err + abs(p_h2 - (1.0 if world.hypothesis_h2 else 0.0))
