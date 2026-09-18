"""Shared plumbing for the 2S experiments (TRUTH PLANE: may use Twin2S for truth and metrics).

The model under test only ever receives Observations (estimated poses, payloads in the ObjectStore) and
the robot's SensorSpecs. Truth (occupancy, observability of query cells) is computed here, after the fact.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
import platform
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import yaml

from conrad.domains.spatial.baselines import MaxLikelihoodSurfaceFill, PlainOccupancyGrid
from conrad.domains.spatial.config import SpatialConfig, ignore_pose_config, spatial_config
from conrad.domains.spatial.model import Model2S
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Observation
from conrad.schemas.timebase import stamp
from conrad.schemas.world import Scenario, SensorSpec
from conrad.sim.scenarios.pipeline_inspection import build_pipeline_inspection_scenario
from conrad.twins.twin2s.evaluation import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    occupancy_error_split,
    occupancy_iou,
    occupancy_truth_at,
    query_points_between,
    unsupported_confidence,
)
from conrad.twins.twin2s.ocpwe import InformationConditions, ObservationSchedule, plan_schedule, run_schedule
from conrad.twins.twin2s.twin import Twin2S
from conrad.twins.twin2s.visibility import VisibilityOracle, VisibilityReason

DEFAULT_SEEDS = (2026201, 2026202, 2026203)
TWIN_ID_OFFSET = 700_001  # twins never share the scenario IdFactory stream
MODEL_ID_OFFSET = 900_007
GEOMETRIC = ("DEPTH_RANGE", "SONAR")
ModelFactory = Callable[[IdFactory, ObjectStore, SpatialConfig], Model2S]
MODELS: dict[str, ModelFactory] = {
    "uahsm": lambda ids, st, cfg: Model2S(ids, st, cfg),
    "uahsm_no_pose_cov": lambda ids, st, cfg: Model2S(ids, st, ignore_pose_config(cfg)),
    "plain_grid": lambda ids, st, cfg: PlainOccupancyGrid(ids, st, cfg),
    "ml_fill": lambda ids, st, cfg: MaxLikelihoodSurfaceFill(ids, st, cfg),
}


def load_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    return data


@dataclass
class World:
    seed: int
    scenario: Scenario
    store: ObjectStore
    sensors: tuple[SensorSpec, ...]

    def twin(self, scenario: Scenario | None = None) -> Twin2S:
        tw = Twin2S(IdFactory(self.seed + TWIN_ID_OFFSET), self.store)
        tw.initialize(scenario or self.scenario)
        tw.reset(self.seed)
        return tw

    @property
    def mission_id(self) -> UUID:
        if self.scenario.mission is None:
            raise ValueError("scenario has no MissionSpec")
        return self.scenario.mission.mission_id

    def group(self, name: str) -> tuple[UUID, ...]:
        return tuple(UUID(u) for u in self.scenario.spatial_state.get("groups", {}).get(name, []))


def build_world(seed: int, config: Mapping[str, Any], root: str | None = None) -> World:
    sc = config.get("scenario", {})
    scenario = build_pipeline_inspection_scenario(
        seed,
        IdFactory(seed),
        family=sc.get("family", "pipeline_with_supports"),
        sensor_overrides=sc.get("sensor_overrides"),
    )
    store = ObjectStore(root or tempfile.mkdtemp(prefix="m2s-exp-"))
    return World(seed, scenario, store, scenario.robots[0].sensors)


def conditions(targets: tuple[UUID, ...], base: Mapping[str, Any], **over: Any) -> InformationConditions:
    kw = {**dict(base), **over}
    for key in ("modalities", "standoffs_m", "heights_m"):
        if key in kw:
            kw[key] = tuple(kw[key])
    return InformationConditions(target_entity_ids=targets, **kw)


def observe(
    world: World, twin: Twin2S, cond: InformationConditions, seed: int
) -> tuple[ObservationSchedule, list[Observation]]:
    sched = plan_schedule(twin, world.sensors, cond, np.random.default_rng(seed))
    ids = IdFactory(seed + 13)
    samples = run_schedule(twin, sched, world.mission_id, ids.new(), ids.new())
    return sched, [s.observation for s in samples]  # supervision labels are discarded here


def eval_points(twin: Twin2S, entity_ids: Sequence[UUID], margin_m: float, res: float) -> np.ndarray:
    """Query cell centres (aligned with the map grid) over the entities' bounding box plus a margin."""
    los, his = [], []
    for eid in entity_ids:
        i = twin.world.index_of(eid)
        lo, hi = twin.world.entities[i].primitive.bounds()
        off = twin.world.entity_offset(i)
        los.append(np.asarray(lo) + off)
        his.append(np.asarray(hi) + off)
    lo = np.floor((np.min(los, axis=0) - margin_m) / res) * res
    hi = np.ceil((np.max(his, axis=0) + margin_m) / res) * res
    lo = np.maximum(lo, np.asarray(twin.world.bounds_min))
    hi = np.minimum(hi, np.asarray(twin.world.bounds_max))
    return query_points_between(lo, hi, res)


def observed_mask(twin: Twin2S, sched: ObservationSchedule, points: np.ndarray, res: float) -> np.ndarray:
    """Truth-plane observability of each query cell: VISIBLE (quality-thresholded, within half a cell
    diagonal of the first surface) from at least one scheduled TRUE pose by an active geometric sensor."""
    oracle = VisibilityOracle(twin.world, twin.cfg)
    tol = 0.5 * res * np.sqrt(3.0)
    out = np.zeros(len(points), dtype=bool)
    for v in sched.views:
        for s in v.sensors:
            if s.modality in GEOMETRIC and not v.degradation.get(s.modality, {}).get("fault", 0.0) >= 1.0:
                r = oracle.visibility(s, v.true_pose, points, None, 0.0, tolerance_m=tol)
                out |= r.reason == VisibilityReason.VISIBLE
    return out


def run_model(
    kind: str, world: World, observations: Sequence[Observation], model_cfg: SpatialConfig, t_end: float
) -> Model2S:
    m = MODELS[kind](IdFactory(world.seed + MODEL_ID_OFFSET), world.store, model_cfg)
    m.initialize({"sensors": world.sensors})
    m.ingest_observations(list(observations))
    m.update_beliefs(stamp(t_end, "SIM"))
    return m


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            err += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(err)


def map_metrics(
    model: Model2S,
    points: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    thr: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, float]:
    state = model.occupancy_state(points)
    p = state.probability
    status = [KnowledgeStatus(s) for s in model.occupancy_status(points)]
    claimed = np.array([s is not KnowledgeStatus.UNKNOWN for s in status])
    y = truth.astype(float)
    pred_occ = claimed & (p > 0.5)
    conf = np.maximum(p, 1 - p) >= thr
    wrong = (p > 0.5) != truth
    out: dict[str, float] = {
        **occupancy_error_split(p, truth, observed),
        "iou": occupancy_iou(pred_occ, truth),
        "iou_observed": occupancy_iou(pred_occ[observed], truth[observed]),
        "ece_claimed": ece(p[claimed], y[claimed]),
        "ece_observed_claimed": ece(p[claimed & observed], y[claimed & observed]),
        "confident_claim_rate": float(conf.mean()),
        "confident_error_rate": float(wrong[conf].mean()) if conf.any() else float("nan"),
        "unknown_fraction": float((~claimed).mean()),
        "coverage_estimate": float(state.coverage.mean()),
        "true_observed_fraction": float(observed.mean()),
        "E_coverage": abs(float(state.coverage.mean()) - float(observed.mean())),
        "mean_UA_claimed": float(state.uncertainty[claimed, 0].mean()) if claimed.any() else float("nan"),
        "mean_UE_claimed": float(state.uncertainty[claimed, 1].mean()) if claimed.any() else float("nan"),
        "n_cells": float(len(points)),
        "n_true_occupied": float(truth.sum()),
    }
    uc = unsupported_confidence(p, status, ~observed, thr)
    out.update({f"hidden_{k}": v for k, v in uc.items()})
    return out


def truth_occupancy(twin: Twin2S, points: np.ndarray) -> np.ndarray:
    return np.asarray(occupancy_truth_at(twin, points), dtype=bool)


def model_config(config: Mapping[str, Any]) -> SpatialConfig:
    return spatial_config(config.get("model", {}))


def aggregate(per_seed: Mapping[int, Any]) -> Any:
    """Mean/std/n over seeds for every numeric leaf with the same path."""
    first = next(iter(per_seed.values()))
    if isinstance(first, Mapping):
        keys = sorted({k for v in per_seed.values() for k in v})
        return {k: aggregate({s: v[k] for s, v in per_seed.items() if k in v}) for k in keys}
    vals = np.array(
        [v for v in per_seed.values() if isinstance(v, (int, float)) and np.isfinite(v)], dtype=float
    )
    return {
        "mean": float(vals.mean()) if vals.size else None,
        "std": float(vals.std(ddof=1)) if vals.size > 1 else None,
        "n": int(vals.size),
    }


def write_result(
    experiment_id: str,
    config: Mapping[str, Any],
    seeds: Sequence[int],
    per_seed: dict[int, Any],
    out_dir: str | Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "experiment_id": experiment_id,
        "claim_scope": "SYNTHETIC_ONLY (Twin2S OCPWE worlds); not a physical-world result",
        "seeds": list(seeds),
        "config": dict(config),
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "aggregate": aggregate(per_seed),
        "environment": {"python": platform.python_version(), "machine": platform.machine()},
        **(extra or {}),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{experiment_id}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
