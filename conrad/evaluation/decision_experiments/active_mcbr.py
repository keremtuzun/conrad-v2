"""ACTIVE-MCBR-E001: MCBR vs baselines in an abstract 3D occlusion world (SYNTHETIC_ONLY).

Truth (defects, hypothesis, true occluders, turbidity) lives in ``conrad.evaluation.oracle``. The
planners see only a belief: per-sector Gaussian estimates, a hypothesis probability, a BELIEVED
occluder map (one occluder may be missing, positions jittered) and the observation history. The
score is ACTUAL hidden-state error reduction, not uncertainty reduction.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from conrad.active import MCBRConfig, MCBRPlanner, PlanningRequest, PriorView, SensorOption, make_planners
from conrad.evaluation.decision_experiments.fixtures import CLOCK, make_belief
from conrad.evaluation.oracle.occlusion_world import (
    TARGET_RADIUS_M,
    ObservationOutcome,
    OcclusionWorld,
    belief_error,
    observe,
    sample_world,
    segment_blocked,
    visible_sectors,
)
from conrad.schemas.decision import InformationNeed, PlanStatus, QuestionType, ResourceCost
from conrad.schemas.frames import WORLD, Pose, SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp
from conrad.schemas.uncertainty import CalibrationMetadata, Uncertainty

EXPERIMENT_ID = "ACTIVE-MCBR-E001"
SCENARIOS = ("coverage", "measurement", "contradiction", "ood")
QUESTION = {
    "coverage": QuestionType.EXTEND_COVERAGE,
    "measurement": QuestionType.IMPROVE_MEASUREMENT,
    "contradiction": QuestionType.DISCRIMINATE_HYPOTHESES,
    "ood": QuestionType.CONFIRM_CONDITION,
}


@dataclass
class SectorBelief:
    mean: np.ndarray
    var: np.ndarray
    p_h2: float
    occluders: list[tuple[np.ndarray, float]]
    views: list[PriorView] = field(default_factory=list)
    sonar_seen_weld: bool = False


def _nominal_std(world: OcclusionWorld, modality: str, dist: float) -> float:
    s = world.sensors[modality]  # sensor SPEC, not hidden state: the belief does not know about turbidity
    return s.noise_std + s.range_noise_per_m * dist


def update(
    b: SectorBelief, world: OcclusionWorld, pos: np.ndarray, modality: str, o: ObservationOutcome
) -> None:
    for s, val in zip(o.sectors, o.values, strict=True):
        std = _nominal_std(world, modality, float(np.linalg.norm(pos - world.sector_point(s))))
        v = 1.0 / (1.0 / b.var[s] + 1.0 / std**2)
        b.mean[s] = v * (b.mean[s] / b.var[s] + val / std**2)
        b.var[s] = v
    if o.hypothesis_vote_h2 is not None:
        acc = world.sensors[modality].hypothesis_accuracy  # believed accuracy
        lr = acc / (1 - acc) if o.hypothesis_vote_h2 else (1 - acc) / acc
        odds = b.p_h2 / max(1 - b.p_h2, 1e-9) * lr
        b.p_h2 = float(odds / (1 + odds))
    if modality == "SONAR" and any(s in world.weld_sectors for s in o.sectors):
        b.sonar_seen_weld = True
    b.views.append(PriorView(position_m=(float(pos[0]), float(pos[1]), float(pos[2])), modality=modality))


def belief_uncertainty(b: SectorBelief, world: OcclusionWorld, scenario: str) -> Uncertainty:
    w = list(world.weld_sectors)
    unobserved = float(np.mean(b.var[w] >= 0.9))
    observed = [s for s in w if b.var[s] < 0.9]
    ua = float(np.mean([min(1.0, math.sqrt(b.var[s]) / 0.5) for s in observed])) if observed else 0.0
    uc = 1.0 - abs(2 * b.p_h2 - 1) if scenario == "contradiction" else 0.0
    ue = 0.8 if (scenario == "ood" and not b.sonar_seen_weld) else 0.1
    return Uncertainty(
        aleatoric=ua,
        epistemic=ue,
        contradiction=uc,
        observational=unobserved,
        calibration_metadata=CalibrationMetadata(calibrated=False),
    )


def initial_belief(world: OcclusionWorld, scenario: str, rng: np.random.Generator) -> SectorBelief:
    n = world.n_sectors
    believed = [(c + rng.normal(0, 0.2, 3), r) for c, r in world.occluders]
    if rng.random() < 0.3:
        believed.pop(int(rng.integers(len(believed))))  # the map misses one occluder
    b = SectorBelief(np.full(n, 0.5), np.full(n, 1.0), 0.5, believed)
    weld = set(world.weld_sectors)
    for s in range(n):
        if s not in weld:
            b.mean[s] = float(np.clip(world.defects[s] + rng.normal(0, 0.1), 0, 1))
            b.var[s] = 0.01
    if scenario != "coverage":
        var = {"measurement": 0.3, "contradiction": 0.03, "ood": 0.3}[scenario]
        for s in weld:
            b.mean[s] = float(world.defects[s] + rng.normal(0, math.sqrt(var)))
            b.var[s] = var
        b.views.append(PriorView(position_m=(3.5, 0.0, 0.0), modality="RGB"))
    return b


def _callables(world: OcclusionWorld, b: SectorBelief) -> tuple[Any, Any, Any]:
    free_sensor = type(world.sensors["RGB"])("ANY", 0.0, 100.0, 0.0, 0.0, 0.5)

    def is_free(points: np.ndarray) -> np.ndarray:
        out = []
        for p in points:
            ok = math.hypot(p[0], p[1]) > TARGET_RADIUS_M + 0.3 and abs(p[2]) < 3.0
            ok = ok and all(float(np.linalg.norm(p - c)) > r + 0.3 for c, r in b.occluders)
            out.append(ok)
        return np.array(out, dtype=bool)

    def predicted_visibility(pose: Pose, region: SpatialSupport) -> float:
        pos = np.array(pose.position_m)
        target = [
            s
            for s in range(world.n_sectors)
            if all(
                abs(world.sector_point(s)[i] - region.center_m[i]) <= region.half_extent_m[i] + 1e-6
                for i in range(3)
            )
        ]
        if not target:
            return 0.0
        seen = set(visible_sectors(world, pos, free_sensor, b.occluders))
        return len(seen.intersection(target)) / len(target)

    def navigation_cost(a: Pose, c: Pose) -> ResourceCost | None:
        p, q = np.array(a.position_m), np.array(c.position_m)
        dist = float(np.linalg.norm(q - p))
        if segment_blocked(p, q, b.occluders) or segment_blocked(
            p, q, [(np.zeros(3), TARGET_RADIUS_M + 0.2)]
        ):
            dist *= 1.6  # believed detour
        clearance = min((float(np.linalg.norm(q - cc)) - r for cc, r in b.occluders), default=10.0)
        risk = 0.6 if clearance < 0.5 else 0.05
        return ResourceCost(time_s=dist / 0.5, energy_j=40.0 * dist, risk=risk, travel_m=dist)

    return is_free, predicted_visibility, navigation_cost


def _sensors(ids: IdFactory) -> tuple[SensorOption, ...]:
    return (
        SensorOption(
            sensor_id=ids.new(),
            modality="RGB",
            min_range_m=0.5,
            max_range_m=4.0,
            power_w=5.0,
            measurement_quality=0.6,
            discrimination={"H1|H2": 0.1, "default": 0.1},
        ),
        SensorOption(
            sensor_id=ids.new(),
            modality="SONAR",
            min_range_m=1.0,
            max_range_m=8.0,
            power_w=15.0,
            measurement_quality=0.8,
            discrimination={"H1|H2": 0.7, "default": 0.7},
        ),
    )


def run_episode(
    world: OcclusionWorld,
    scenario: str,
    planner: Any,
    b0: SectorBelief,
    seed: int,
    steps: int,
    ids: IdFactory,
    sensors: tuple[SensorOption, ...],
) -> dict[str, Any]:
    b = copy.deepcopy(b0)
    robot = np.array([-5.0, 0.0, 0.0])
    weld_region = SpatialSupport(
        frame_id=WORLD, center_m=(TARGET_RADIUS_M, 0.0, 0.0), half_extent_m=(0.3, 0.8, 0.3)
    )
    err0 = belief_error(world, b.mean, b.p_h2, True)
    tot0 = belief_error(world, b.mean, b.p_h2, False)
    u0 = sum(belief_uncertainty(b, world, scenario).as_tuple())
    time_s = energy = travel = 0.0
    regrets: list[float] = []
    redundant = n_obs = 0
    status = "PLAN"
    for step in range(steps):
        u = belief_uncertainty(b, world, scenario)
        msg = make_belief(ids, uncertainty=u, support=weld_region, time_s=100.0 + step)
        need = InformationNeed(
            need_id=ids.new(),
            trace_id=ids.new(),
            target_belief_ids=(msg.belief_id,),
            question_type=QUESTION[scenario],
            target_properties=("condition",),
            priority=0.9,
            constraints={"cause": "EPISTEMIC"} if scenario == "ood" else {},
        )
        is_free, vis, nav = _callables(world, b)
        req = PlanningRequest(
            need=need,
            beliefs=[msg],
            robot_pose=Pose(frame_id=WORLD, position_m=tuple(robot)),
            sensors=sensors,
            is_free=is_free,
            predicted_visibility=vis,
            navigation_cost=nav,
            now=stamp(100.0 + step, CLOCK),
            prior_views=tuple(b.views),
            rng=np.random.default_rng(seed * 31 + step),
        )
        result = planner.plan(req)
        step_seed = seed * 1000 + step
        oracle = {}
        for row in result.table:
            if row["feasible"]:
                trial = copy.deepcopy(b)
                pos = np.array(row["position_m"])
                o = observe(world, pos, row["modality"], np.random.default_rng(step_seed))
                update(trial, world, pos, row["modality"], o)
                oracle[row["action_id"]] = belief_error(world, b.mean, b.p_h2, True) - belief_error(
                    world, trial.mean, trial.p_h2, True
                )
        best = max(oracle.values(), default=0.0)
        if result.plan.status is not PlanStatus.PLAN:
            status = result.plan.status.value
            regrets.append(max(0.0, best))  # stopping forgoes the best available improvement
            break
        a = result.plan.primary_action
        assert a is not None
        pos = np.array(a.pose.position_m)
        modality = str(a.sensor_configuration["modality"])
        before = belief_error(world, b.mean, b.p_h2, True)
        update(b, world, pos, modality, observe(world, pos, modality, np.random.default_rng(step_seed)))
        gain = before - belief_error(world, b.mean, b.p_h2, True)
        regrets.append(best - oracle.get(str(a.action_id), gain))
        redundant += gain < 0.01
        n_obs += 1
        time_s += a.expected_cost.time_s
        energy += a.expected_cost.energy_j
        travel += a.expected_cost.travel_m
        robot = pos
    red = err0 - belief_error(world, b.mean, b.p_h2, True)
    return {
        "mission_error_reduction": red,
        "total_error_reduction": tot0 - belief_error(world, b.mean, b.p_h2, False),
        "uncertainty_reduction": u0 - sum(belief_uncertainty(b, world, scenario).as_tuple()),
        "info_per_time": red / time_s if time_s > 0 else 0.0,
        "info_per_kj": 1000.0 * red / energy if energy > 0 else 0.0,
        "redundant_observations": redundant,
        "observations": n_obs,
        "travel_m": travel,
        "time_s": time_s,
        "energy_j": energy,
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "final_status": status,
    }


def run_seed(seed: int, episodes_per_scenario: int, steps: int, mcbr_cfg: MCBRConfig) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ids = IdFactory(seed)
    planners: dict[str, Any] = dict(make_planners(ids, mcbr_cfg))
    # diagnostic arm: same ranking as full MCBR, stop rule disabled (separates ranking from stopping)
    planners["A-B10_mcbr_full_no_stop_rule"] = MCBRPlanner(
        ids, mcbr_cfg, name="A-B10_mcbr_full_no_stop_rule", value_gate=False
    )
    sensors = _sensors(ids)
    rows: dict[str, list[dict[str, Any]]] = {n: [] for n in planners}
    for scenario in SCENARIOS:
        for ep in range(episodes_per_scenario):
            world = sample_world(rng)
            world.turbid = scenario == "ood"
            b0 = initial_belief(world, scenario, rng)
            for name, planner in planners.items():
                r = run_episode(world, scenario, planner, b0, seed + ep, steps, ids, sensors)
                rows[name].append({"scenario": scenario, **r})
    keys = [k for k in rows[next(iter(rows))][0] if k not in ("scenario", "final_status")]
    out: dict[str, Any] = {}
    for name, rs in rows.items():
        out[name] = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        out[name]["by_scenario_mission_error_reduction"] = {
            s: float(np.mean([r["mission_error_reduction"] for r in rs if r["scenario"] == s]))
            for s in SCENARIOS
        }
        out[name]["stopped_early"] = sum(r["final_status"] != "PLAN" for r in rs)
    return out


def run(config: dict[str, Any], seeds: list[int], out_dir: str | Path) -> dict[str, Any]:
    cfg = MCBRConfig(**config.get("mcbr", {}))
    per_seed = {
        str(s): run_seed(s, int(config.get("episodes_per_scenario", 6)), int(config.get("steps", 4)), cfg)
        for s in seeds
    }
    planners = list(next(iter(per_seed.values())))
    metrics = [k for k, v in next(iter(per_seed.values()))[planners[0]].items() if isinstance(v, float)]
    summary = {
        p: {
            m: {
                "mean": float(np.mean([per_seed[s][p][m] for s in per_seed])),
                "std": float(np.std([per_seed[s][p][m] for s in per_seed])),
            }
            for m in metrics
        }
        for p in planners
    }
    ranking = sorted(planners, key=lambda p: -summary[p]["mission_error_reduction"]["mean"])
    result = {
        "experiment_id": EXPERIMENT_ID,
        "data_status": "SYNTHETIC_ONLY",
        "config": config,
        "seeds": seeds,
        "ranking_by_mission_error_reduction": ranking,
        "summary": summary,
        "per_seed": per_seed,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "active_mcbr_e001.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
