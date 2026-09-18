"""OCPWE: (World, DesiredInformationCondition) -> ObservationSchedule (ch15, T2S-TRAJ-01). TRUTH PLANE.

Explicit controls: target coverage, occlusion preference, viewpoint diversity, modality availability,
redundancy, pose uncertainty (true vs reported pose) and sensor degradation. Styles: coverage, redundant,
poor, information_seeking (discriminating lives in counterfactual.py).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np

from conrad.schemas.frames import WORLD, Pose, euler_from_quat, quat_from_euler, quat_to_matrix
from conrad.schemas.timebase import stamp
from conrad.schemas.world import SensorSpec
from conrad.twins.base import SensingContext, TwinSample
from conrad.twins.twin2s.coverage import CoverageAccumulator
from conrad.twins.twin2s.sdf import Arr
from conrad.twins.twin2s.twin import Twin2S
from conrad.twins.twin2s.visibility import VisibilityReason, sample_surface

STYLES = ("coverage", "redundant", "poor", "information_seeking")


@dataclass(frozen=True)
class InformationConditions:
    target_entity_ids: tuple[UUID, ...]
    style: str = "coverage"
    target_coverage: float = 0.8
    n_views: int = 6
    redundancy: int = 1
    min_view_separation_rad: float = 0.0
    prefer_occluded: bool = False
    modalities: tuple[str, ...] = ("DEPTH_RANGE", "SONAR", "RGB")
    planning_modality: str = "DEPTH_RANGE"
    pose_sigma_m: float = 0.0
    pose_sigma_rad: float = 0.0
    degradation: dict[str, dict[str, float]] = field(default_factory=dict)
    standoffs_m: tuple[float, ...] = (2.0, 3.5)
    heights_m: tuple[float, ...] = (0.2, 1.5)
    n_azimuths: int = 8
    n_surface_samples: int = 300
    dt_s: float = 1.0


@dataclass(frozen=True)
class ScheduledView:
    time_s: float
    true_pose: Pose
    estimated_pose: Pose
    sensors: tuple[SensorSpec, ...]
    degradation: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ObservationSchedule:
    views: tuple[ScheduledView, ...]
    conditions: InformationConditions
    achieved: dict[str, float]


def look_at_pose(position: Arr, target: Arr) -> Pose:
    """Robot pose at ``position`` whose +X boresight points at ``target`` (sensor mounts are unrotated)."""
    d = np.asarray(target, dtype=np.float64) - position
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(-d[2], math.hypot(d[0], d[1]))
    return Pose(
        frame_id=WORLD,
        position_m=(float(position[0]), float(position[1]), float(position[2])),
        orientation_wxyz=quat_from_euler(0.0, pitch, yaw),
    )


def perturb_pose(true: Pose, sigma_m: float, sigma_rad: float, rng: np.random.Generator) -> Pose:
    """Reported pose P_obs = P_true + eps with a declared covariance (sigma 0 -> covariance zero, not unknown)."""
    r, p, y = euler_from_quat(true.orientation_wxyz)
    dp = rng.normal(0.0, sigma_m, 3) if sigma_m > 0 else np.zeros(3)
    da = rng.normal(0.0, sigma_rad, 3) if sigma_rad > 0 else np.zeros(3)
    cov = np.diag([sigma_m**2] * 3 + [sigma_rad**2] * 3).ravel()
    pos = np.asarray(true.position_m) + dp
    return Pose(
        frame_id=true.frame_id,
        position_m=(float(pos[0]), float(pos[1]), float(pos[2])),
        orientation_wxyz=quat_from_euler(r + da[0], p + da[1], y + da[2]),
        covariance_6x6=tuple(float(c) for c in cov),
    )


def _targets(
    twin: Twin2S, ids: tuple[UUID, ...], n: int, rng: np.random.Generator
) -> tuple[Arr, Arr, list[Arr]]:
    pts, nrm, foci = [], [], []
    per = max(n // max(len(ids), 1), 8)
    for eid in ids:
        p, q = sample_surface(twin.world, twin.world.index_of(eid), per, rng)
        if len(p):
            pts.append(p)
            nrm.append(q)
            foci.append(p.mean(axis=0))
    if not pts:
        raise ValueError("target entities expose no observable surface")
    return np.concatenate(pts), np.concatenate(nrm), foci


def candidate_poses(twin: Twin2S, foci: list[Arr], cond: InformationConditions) -> list[Pose]:
    """Viewpoints on rings around every target focus, looking at that focus."""
    out = []
    lo, hi = np.asarray(twin.world.bounds_min), np.asarray(twin.world.bounds_max)
    for focus in foci:
        for s in cond.standoffs_m:
            for h in cond.heights_m:
                for k in range(cond.n_azimuths):
                    a = 2.0 * math.pi * k / cond.n_azimuths
                    pos = focus + np.array([s * math.cos(a), s * math.sin(a), h])
                    if np.any(pos < lo) or np.any(pos > hi) or twin.world.sdf(pos[None, :])[0] < 0.3:
                        continue  # outside the world or inside / too close to geometry
                    out.append(look_at_pose(pos, focus))
    return out


def plan_schedule(
    twin: Twin2S, sensors: tuple[SensorSpec, ...], cond: InformationConditions, rng: np.random.Generator
) -> ObservationSchedule:
    if cond.style not in STYLES:
        raise ValueError(f"unknown OCPWE style {cond.style!r}")
    pts, nrm, foci = _targets(twin, cond.target_entity_ids, cond.n_surface_samples, rng)
    planner = next(s for s in sensors if s.modality == cond.planning_modality)
    cands = candidate_poses(twin, foci, cond)
    if not cands:
        raise ValueError("no feasible candidate viewpoint inside the world bounds")
    vis = [twin.visibility(planner, p, pts, nrm) for p in cands]
    good = np.stack([v.visible & (v.score >= twin.cfg.observed.min_quality) for v in vis])
    occl = np.array([float((v.reason == VisibilityReason.OCCLUDED).mean()) for v in vis])
    fwd = [np.asarray(quat_to_matrix(p.orientation_wxyz)[:, 0]) for p in cands]
    chosen: list[int] = []
    covered = np.zeros(len(pts), dtype=bool)
    order = rng.permutation(len(cands))
    if cond.style == "redundant":
        chosen = [int(order[np.argmax(good[order].sum(1))])]
    elif cond.style == "poor":
        score = good.sum(1) - (100.0 * occl if cond.prefer_occluded else 0.0)
        chosen = [int(i) for i in order[np.argsort(score[order], kind="stable")][: cond.n_views]]
    else:
        sep = max(cond.min_view_separation_rad, 0.5 if cond.style == "information_seeking" else 0.0)
        while len(chosen) < cond.n_views and covered.mean() < cond.target_coverage:
            gain = (good & ~covered).sum(1).astype(float)
            if sep > 0:
                for i in range(len(cands)):
                    for j in chosen:
                        cosang = fwd[i] @ fwd[j] / (np.linalg.norm(fwd[i]) * np.linalg.norm(fwd[j]))
                        if math.acos(float(np.clip(cosang, -1, 1))) < sep:
                            gain[i] = -1.0
            if cond.prefer_occluded:
                gain = gain * (1.0 + occl)
            # overshoot control: do not jump far past the requested coverage
            over = np.maximum((covered | good).mean(1) - cond.target_coverage, 0.0)
            gain = np.where(over > 0.1, gain * 0.01, gain)
            gain[chosen] = -1.0
            best = int(order[np.argmax(gain[order])])
            if gain[best] <= 0:
                break
            chosen.append(best)
            covered |= good[best]
    active = tuple(s for s in sensors if s.modality in cond.modalities)
    views, t = [], 0.0
    for i in chosen:
        reps = cond.n_views if cond.style == "redundant" else cond.redundancy
        for _ in range(reps):
            t += cond.dt_s
            est = perturb_pose(cands[i], cond.pose_sigma_m, cond.pose_sigma_rad, rng)
            views.append(ScheduledView(t, cands[i], est, active, dict(cond.degradation)))
    acc = CoverageAccumulator(pts, twin.cfg.observed)
    for v in views:
        acc.add(twin.visibility(planner, v.true_pose, pts, nrm), v.time_s, planner.modality)
    return ObservationSchedule(tuple(views), cond, acc.summary())


def run_schedule(
    twin: Twin2S,
    schedule: ObservationSchedule,
    mission_id: UUID,
    run_id: UUID,
    trace_id: UUID,
    advance_time: bool = True,
) -> list[TwinSample]:
    """Execute a schedule: every active sensor at every view. Twin time advances to each view time."""
    out: list[TwinSample] = []
    for seq, v in enumerate(schedule.views):
        if advance_time and v.time_s > twin.world.time_s:
            twin.step(v.time_s - twin.world.time_s)
        ts = stamp(v.time_s, twin.cfg.clock_domain, sequence_index=seq)
        for sensor in v.sensors:
            ctx = SensingContext(
                mission_id,
                run_id,
                trace_id,
                sensor,
                v.true_pose,
                v.estimated_pose,
                ts,
                dict(v.degradation.get(sensor.modality, {})),
            )
            out.extend(twin.generate_observation(ctx))
    return out
