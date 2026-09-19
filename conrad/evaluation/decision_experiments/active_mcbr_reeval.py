"""MCBR re-evaluation under immutable partitions and matched budgets (2026-09-19). SYNTHETIC_ONLY.

    ACTIVE-MCBR-SEL001  selection of the production planner on the VALIDATION partition (abstract world)
    ACTIVE-MCBR-E002    final evaluation, abstract occlusion world: FINAL_TEST and OOD_TEST partitions
    ACTIVE-MCBR-E003    final evaluation, integrated SURROGATE mission (python kernel, FLAGSHIP-I4 family)

Seeds come only from ``conrad.evaluation.partitions`` with a declared purpose, so design/tuning code can
never read held-out seeds. Every planner in a comparison gets the same budget: the same maximum number of
observations, the same energy and time caps (enforced by the shared feasibility filter through
``MissionBounds`` / the need deadline, so candidates that would overrun are rejected for every planner),
the same sensors, the same candidate generator and common random numbers for observation noise.

Scores are ACTUAL hidden-state error reductions against the evaluation-only truth, never uncertainty.
Statistical unit: one world (abstract: mean over replicates x scenario types; mission: one run). Paired
differences use ``conrad.evaluation.metrics.paired_seed_comparison`` (percentile bootstrap, 95 %).
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from conrad.active import MCBRConfig, MCBRPlanner, PlanningRequest, SensorOption, make_planners
from conrad.active.candidates import MissionBounds
from conrad.active.predictive import HypothesisBelief, PredictedOutcome, ScalarBelief
from conrad.active.production import PRODUCTION, build_planner, load_frozen, production_planner
from conrad.active.rankers import RankerConfig, ranker_planner
from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.active_mcbr import (
    QUESTION,
    SCENARIOS,
    SectorBelief,
    _callables,
    _sensors,
    belief_uncertainty,
    initial_belief,
    update,
)
from conrad.evaluation.decision_experiments.fixtures import CLOCK, make_belief
from conrad.evaluation.metrics import paired_seed_comparison
from conrad.evaluation.oracle.occlusion_world import (
    OcclusionWorld,
    belief_error,
    observe,
    sample_world,
    segment_blocked,
    visible_sectors,
)
from conrad.schemas.decision import InformationNeed, PlanStatus
from conrad.schemas.frames import WORLD, Pose, SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp

E002 = "ACTIVE-MCBR-E002"
E003 = "ACTIVE-MCBR-E003"
SEL = "ACTIVE-MCBR-SEL001"
ACTION_THRESHOLD = 0.5  # defect severity above which the weld sector needs action (SYNTHETIC_ONLY)
PRIMARY = "mission_error_reduction"
I4_COMPARATORS = ("A-B1_fixed_inspection", "A-B0_random", "A-B2_coverage")
# the eleven candidates required by the re-evaluation brief, plus diagnostic arms
CANDIDATES = (
    "A-B0_random",
    "A-B1_fixed_inspection",
    "A-B2_coverage",
    "A-B3_frontier",
    "A-B4_geometric_nbv",
    "A-B5_entropy_nbv",
    "A-B6_standard_eig",
    "A-B6b_bayes_eig",
    "A-B7_uncertainty_nbv",
    "A-B9_mcbr_no_mission",
    "A-B10_mcbr_full",
    "A-B10_mcbr_full_no_stop_rule",
    "A-B11_mission_conditioned",
    "A-B12_hypothesis_discrimination",
)


@dataclass(frozen=True)
class Budget:
    max_observations: int = 4
    energy_j: float = 900.0
    time_s: float = 90.0

    @classmethod
    def of(cls, raw: dict[str, Any] | None) -> Budget:
        return cls(**(raw or {}))


# ---------------------------------------------------------------------------------------------- belief model
@dataclass
class SectorPredictive:
    """BELIEF-side predictive model of the abstract world (believed occluders, nominal sensor spec)."""

    world: OcclusionWorld  # used for GEOMETRY (sector layout) and nominal sensor SPECS only
    occluders: list[tuple[np.ndarray, float]]
    scalars: tuple[ScalarBelief, ...]
    hypotheses: tuple[HypothesisBelief, ...]
    epistemic: float
    used_modalities: frozenset[str]

    def predict(self, pose: Pose, sensor: SensorOption) -> PredictedOutcome:
        spec = self.world.sensors[sensor.modality]  # nominal spec: the belief does not know about turbidity
        pos = np.asarray(pose.position_m, dtype=np.float64)
        seen = visible_sectors(self.world, pos, spec, self.occluders)
        std = {
            f"s{s}": spec.noise_std
            + spec.range_noise_per_m * float(np.linalg.norm(pos - self.world.sector_point(s)))
            for s in seen
        }
        acc = {"H2": spec.hypothesis_accuracy} if any(s in self.world.weld_sectors for s in seen) else {}
        return PredictedOutcome(noise_std=std, hypothesis_accuracy=acc)


def predictive_for(world: OcclusionWorld, b: SectorBelief, scenario: str) -> SectorPredictive:
    weld = set(world.weld_sectors)
    scal = tuple(
        ScalarBelief(
            key=f"s{s}",
            mean=float(b.mean[s]),
            var=float(b.var[s]),
            threshold=ACTION_THRESHOLD,
            weight=(1.0 / len(weld)) if s in weld else 0.0,
        )
        for s in range(world.n_sectors)
    )
    u = belief_uncertainty(b, world, scenario)
    return SectorPredictive(
        world,
        b.occluders,
        scal,
        (HypothesisBelief("H2", float(b.p_h2)),),
        float(u.epistemic),
        frozenset(v.modality for v in b.views),
    )


def weld_region(world: OcclusionWorld) -> SpatialSupport:
    pts = np.array([world.sector_point(s) for s in world.weld_sectors])
    half_x = max(0.3, 1.0 - float(pts[:, 0].min()) + 0.05)
    half_y = max(0.8, float(np.abs(pts[:, 1]).max()) + 0.16)
    return SpatialSupport(frame_id=WORLD, center_m=(1.0, 0.0, 0.0), half_extent_m=(half_x, half_y, 0.3))


# ---------------------------------------------------------------------------------------------- planners
def build_planners(
    ids: IdFactory,
    mcbr_cfg: MCBRConfig,
    names: tuple[str, ...] | list[str],
    variants: dict[str, dict[str, Any]] | None = None,
    include_production: bool = False,
) -> dict[str, Any]:
    base: dict[str, Any] = dict(make_planners(ids, mcbr_cfg))
    base["A-B10_mcbr_full_no_stop_rule"] = MCBRPlanner(
        ids, mcbr_cfg, name="A-B10_mcbr_full_no_stop_rule", value_gate=False
    )
    out = {n: base[n] for n in names}
    for vname, section in (variants or {}).items():
        out[vname] = build_planner(section, ids, mcbr_cfg, vname)
    if include_production:
        out[PRODUCTION] = production_planner(ids, mcbr_cfg)
    return out


# ---------------------------------------------------------------------------------------------- abstract episode
def _missing_true_occluders(world: OcclusionWorld, b: SectorBelief) -> list[tuple[np.ndarray, float]]:
    return [
        (c, r)
        for c, r in world.occluders
        if all(float(np.linalg.norm(c - bc)) > 0.8 for bc, _ in b.occluders)
    ]


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def run_abstract_episode(
    world: OcclusionWorld,
    scenario: str,
    planner: Any,
    b0: SectorBelief,
    noise_seed: tuple[int, ...],
    budget: Budget,
    ids: IdFactory,
    sensors: tuple[SensorOption, ...],
    diagnostics: bool = False,
) -> dict[str, Any]:
    b = copy.deepcopy(b0)
    robot = np.array([-5.0, 0.0, 0.0])
    region = weld_region(world)
    missing = _missing_true_occluders(world, b)
    err0 = belief_error(world, b.mean, b.p_h2, True)
    tot0 = belief_error(world, b.mean, b.p_h2, False)
    max_dwell = max(s.duration_s for s in sensors)
    max_sensor_j = max(s.power_w * s.duration_s for s in sensors)
    time_s = energy = travel = 0.0
    redundant = n_obs = collisions = stops = 0
    regrets: list[float] = []
    corrs: list[float] = []
    status = "BUDGET_OR_STEPS_EXHAUSTED"
    for step in range(budget.max_observations):
        u = belief_uncertainty(b, world, scenario)
        msg = make_belief(ids, uncertainty=u, support=region, time_s=100.0 + time_s)
        now = stamp(100.0 + time_s, CLOCK)
        remaining_t = budget.time_s - time_s - max_dwell
        remaining_e = budget.energy_j - energy - max_sensor_j
        if remaining_t <= 0 or remaining_e <= 0:
            break
        need = InformationNeed(
            need_id=ids.new(),
            trace_id=ids.new(),
            target_belief_ids=(msg.belief_id,),
            question_type=QUESTION[scenario],
            target_properties=("condition",),
            priority=0.9,
            deadline_ns=now.time_ns + int(remaining_t * 1e9),
            constraints={"cause": "EPISTEMIC"} if scenario == "ood" else {},
        )
        is_free, vis, nav = _callables(world, b)
        req = PlanningRequest(
            need=need,
            beliefs=[msg],
            robot_pose=Pose(frame_id=WORLD, position_m=tuple(float(x) for x in robot)),
            sensors=sensors,
            is_free=is_free,
            predicted_visibility=vis,
            navigation_cost=nav,
            now=now,
            prior_views=tuple(b.views),
            bounds=MissionBounds((-20.0, -20.0, -5.0), (20.0, 20.0, 5.0), remaining_e, now.time_ns),
            rng=np.random.default_rng([*noise_seed, step, 31]),
            predictive=predictive_for(world, b, scenario),
        )
        result = planner.plan(req)
        obs_rng = np.random.default_rng([*noise_seed, step])
        if diagnostics:
            before = belief_error(world, b.mean, b.p_h2, True)
            oracle, scores = [], []
            for row in result.table:
                if row["feasible"]:
                    trial = copy.deepcopy(b)
                    pos = np.array(row["position_m"])
                    o = observe(world, pos, row["modality"], np.random.default_rng([*noise_seed, step]))
                    update(trial, world, pos, row["modality"], o)
                    oracle.append(before - belief_error(world, trial.mean, trial.p_h2, True))
                    scores.append(float(row["score"]))
            best = max(oracle, default=0.0)
            rc = _spearman(scores, oracle)
            if rc is not None:
                corrs.append(rc)
        if result.plan.status is not PlanStatus.PLAN:
            status = result.plan.status.value
            stops += result.plan.status is PlanStatus.NOT_WORTH_COST
            if diagnostics:
                regrets.append(max(0.0, best))
            break
        a = result.plan.primary_action
        assert a is not None
        pos = np.array(a.pose.position_m)
        modality = str(a.sensor_configuration["modality"])
        if segment_blocked(robot, pos, missing) or any(
            float(np.linalg.norm(pos - c)) < r for c, r in world.occluders
        ):
            collisions += 1
        before = belief_error(world, b.mean, b.p_h2, True)
        update(b, world, pos, modality, observe(world, pos, modality, obs_rng))
        gain = before - belief_error(world, b.mean, b.p_h2, True)
        if diagnostics:
            chosen = next(
                (
                    i
                    for i, row in enumerate(r for r in result.table if r["feasible"])
                    if row["action_id"] == str(a.action_id)
                ),
                None,
            )
            regrets.append(best - (oracle[chosen] if chosen is not None else gain))
        redundant += gain < 0.01
        n_obs += 1
        time_s += a.expected_cost.time_s
        energy += a.expected_cost.energy_j
        travel += a.expected_cost.travel_m
        robot = pos
    err1 = belief_error(world, b.mean, b.p_h2, True)
    weld = list(world.weld_sectors)
    classified = all((b.mean[s] > ACTION_THRESHOLD) == (world.defects[s] > ACTION_THRESHOLD) for s in weld)
    hyp_ok = (b.p_h2 > 0.5) == world.hypothesis_h2
    red = err0 - err1
    out = {
        "mission_error_reduction": red,
        "mission_error_final": err1,
        "hidden_state_error_reduction": tot0 - belief_error(world, b.mean, b.p_h2, False),
        "info_per_time": red / time_s if time_s > 0 else 0.0,
        "info_per_kj": 1000.0 * red / energy if energy > 0 else 0.0,
        "redundant_observations": float(redundant),
        "observations": float(n_obs),
        "travel_m": travel,
        "time_s": time_s,
        "energy_j": energy,
        "collisions": float(collisions),
        "mission_success": float(classified and hyp_ok),
        "stopped_not_worth_cost": float(stops),
        "final_status": status,
    }
    if diagnostics:
        out["mean_regret"] = float(np.mean(regrets)) if regrets else 0.0
        out["oracle_rank_corr"] = float(np.mean(corrs)) if corrs else 0.0
    return out


def run_abstract_world(
    world_seed: int,
    family: str,
    planners: dict[str, Any],
    ids: IdFactory,
    budget: Budget,
    replicates: int,
    diagnostics: bool,
) -> dict[str, dict[str, Any]]:
    """All planners on one world: mean over replicates x scenario types (the statistical unit)."""
    world0 = sample_world(np.random.default_rng([world_seed, 0x0C]), family=family)
    sensors = _sensors(ids)
    rows: dict[str, list[dict[str, Any]]] = {n: [] for n in planners}
    for rep in range(replicates):
        for k, scenario in enumerate(SCENARIOS):
            world = copy.deepcopy(world0)
            world.turbid = scenario == "ood"
            rng = np.random.default_rng([world_seed, rep, k, 0xB0])
            b0 = initial_belief(world, scenario, rng)
            for name, planner in planners.items():
                r = run_abstract_episode(
                    world, scenario, planner, b0, (world_seed, rep, k), budget, ids, sensors, diagnostics
                )
                rows[name].append({"scenario": scenario, **r})
    out: dict[str, dict[str, Any]] = {}
    for name, rs in rows.items():
        keys = [k for k, v in rs[0].items() if isinstance(v, float)]
        out[name] = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        out[name]["by_scenario"] = {
            s: float(np.mean([r[PRIMARY] for r in rs if r["scenario"] == s])) for s in SCENARIOS
        }
    return out


def _abstract_worker(args: tuple[Any, ...]) -> tuple[int, dict[str, Any]]:
    seed, family, names, variants, production, mcbr_raw, budget_raw, reps, diag = args
    ids = IdFactory(seed)
    planners = build_planners(ids, MCBRConfig(**mcbr_raw), names, variants, production)
    return seed, run_abstract_world(seed, family, planners, ids, Budget.of(budget_raw), reps, diag)


def evaluate_abstract(
    split: P.Split,
    config: dict[str, Any],
    names: tuple[str, ...] | list[str],
    variants: dict[str, dict[str, Any]] | None = None,
    include_production: bool = False,
    diagnostics: bool = False,
    max_worlds: int | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    seeds = list(split.world_seeds)[: max_worlds or None]
    fams = split.families
    jobs = [
        (
            s,
            fams[s % len(fams)],
            tuple(names),
            variants or {},
            include_production,
            dict(config.get("mcbr", {})),
            dict(config.get("budget", {})),
            split.replicates_per_world,
            diagnostics,
        )
        for s in seeds
    ]
    per_world: dict[int, dict[str, Any]] = {}
    if workers <= 1:
        for j in jobs:
            s, r = _abstract_worker(j)
            per_world[s] = r
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for s, r in ex.map(_abstract_worker, jobs):
                per_world[s] = r
    return {
        "partition": split.partition.value,
        "partition_digest": split.digest,
        "families": list(fams),
        "world_seeds": seeds,
        "replicates_per_world": split.replicates_per_world,
        "per_world": {str(s): per_world[s] for s in seeds},
    }


# ---------------------------------------------------------------------------------------------- statistics
def summarize(per_world: dict[str, dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    planners = list(next(iter(per_world.values())))
    out: dict[str, Any] = {}
    for p in planners:
        out[p] = {}
        for m in metrics:
            vals = [per_world[w][p][m] for w in per_world if m in per_world[w][p]]
            if vals:
                out[p][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return out


def paired(
    per_world: dict[str, dict[str, Any]],
    candidate: str,
    baselines: list[str],
    metrics: dict[str, str],
    seed: int = 20260919,
) -> dict[str, Any]:
    """metric -> direction ('higher'/'lower'); positive benefit = candidate better."""
    out: dict[str, Any] = {}
    for base in baselines:
        if base == candidate:
            continue
        out[base] = {}
        for m, direction in metrics.items():
            b = {int(w): float(per_world[w][base][m]) for w in per_world}
            c = {int(w): float(per_world[w][candidate][m]) for w in per_world}
            pc = paired_seed_comparison(
                b, c, np.random.default_rng(seed), metric=m, direction=direction, n_resamples=4000
            )
            out[base][m] = {
                "benefit_mean": pc.benefit.point,
                "ci95": [pc.benefit.low, pc.benefit.high],
                "n_worlds": pc.benefit.n,
                "fraction_improved": pc.fraction_of_seeds_improved,
                "baseline_mean": pc.baseline_mean,
                "candidate_mean": pc.candidate_mean,
                "repeatable_benefit": pc.repeatable_benefit,
            }
    return out


ABSTRACT_METRICS = {
    "mission_error_reduction": "higher",
    "hidden_state_error_reduction": "higher",
    "mission_error_final": "lower",
    "info_per_time": "higher",
    "info_per_kj": "higher",
    "redundant_observations": "lower",
    "observations": "lower",
    "travel_m": "lower",
    "time_s": "lower",
    "energy_j": "lower",
    "collisions": "lower",
    "mission_success": "higher",
}


def _write(out_dir: Path, name: str, obj: Any) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------------------------- design
def run_design(config: dict[str, Any], seeds: list[int], out_dir: Path) -> dict[str, Any]:
    """Diagnosis and tuning on DEVELOPMENT only (purpose=design): held-out seeds raise inside this scope."""
    with P.purpose_scope(P.Purpose.DESIGN):
        split = P.split("abstract", P.Partition.DEVELOPMENT, P.Purpose.DESIGN)
        res = evaluate_abstract(
            split,
            config,
            list(config.get("planners", CANDIDATES)),
            dict(config.get("variants", {})),
            diagnostics=True,
            max_worlds=config.get("max_worlds"),
            workers=int(config.get("workers", 8)),
        )
    metrics = [*ABSTRACT_METRICS, "mean_regret", "oracle_rank_corr", "stopped_not_worth_cost"]
    summary = summarize(res["per_world"], metrics)
    by_scen = {
        p: {
            s: float(np.mean([res["per_world"][w][p]["by_scenario"][s] for w in res["per_world"]]))
            for s in SCENARIOS
        }
        for p in summary
    }
    result = {
        "purpose": "design",
        **{k: v for k, v in res.items() if k != "per_world"},
        "ranking_by_primary": sorted(summary, key=lambda p: -summary[p][PRIMARY]["mean"]),
        "summary": summary,
        "by_scenario": by_scen,
        "per_world": res["per_world"],
    }
    _write(out_dir, str(config.get("out_name", "design.json")), result)
    return result


# ---------------------------------------------------------------------------------------------- SEL001
def run_selection(config: dict[str, Any], seeds: list[int], out_dir: Path) -> dict[str, Any]:
    """Select on VALIDATION (purpose=selection). ``seeds`` is ignored: seeds come from the partition file."""
    with P.purpose_scope(P.Purpose.SELECTION):
        split = P.split("abstract", P.Partition.VALIDATION, P.Purpose.SELECTION)
        variants = dict(config.get("variants", {}))
        res = evaluate_abstract(
            split,
            config,
            list(config.get("planners", CANDIDATES)),
            variants,
            diagnostics=bool(config.get("diagnostics", True)),
            max_worlds=config.get("max_worlds"),
            workers=int(config.get("workers", 8)),
        )
    metrics = [*ABSTRACT_METRICS, "mean_regret", "oracle_rank_corr", "stopped_not_worth_cost"]
    summary = summarize(res["per_world"], metrics)
    ranking = sorted(summary, key=lambda p: -summary[p][PRIMARY]["mean"])
    winner = ranking[0]
    result = {
        "experiment_id": SEL,
        "data_status": "SYNTHETIC_ONLY",
        "purpose": "selection",
        "primary_metric": PRIMARY,
        "config": config,
        **{k: v for k, v in res.items() if k != "per_world"},
        "ranking_by_primary": ranking,
        "selected": winner,
        "summary": summary,
        "paired_selected_vs_all": paired(res["per_world"], winner, ranking, {PRIMARY: "higher"}),
        "per_world": res["per_world"],
    }
    _write(out_dir, "active_mcbr_sel001.json", result)
    return result


# ---------------------------------------------------------------------------------------------- E002
def run_e002(config: dict[str, Any], seeds: list[int], out_dir: Path) -> dict[str, Any]:
    """Final abstract evaluation of the FROZEN production planner (FINAL_TEST + OOD_TEST)."""
    frozen = load_frozen()
    names = list(config.get("planners", CANDIDATES))
    out: dict[str, Any] = {
        "experiment_id": E002,
        "data_status": "SYNTHETIC_ONLY",
        "evidence_kind": "ABSTRACT_OCCLUSION_WORLD",
        "frozen_planner": frozen,
        "primary_metric": PRIMARY,
        "i4_comparators": list(I4_COMPARATORS),
        "config": config,
        "partitions": {},
    }
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        for part in (P.Partition.FINAL_TEST, P.Partition.OOD_TEST):
            split = P.split("abstract", part, P.Purpose.FINAL_EVALUATION)
            res = evaluate_abstract(
                split,
                config,
                names,
                include_production=True,
                max_worlds=config.get("max_worlds"),
                workers=int(config.get("workers", 8)),
            )
            pw = res["per_world"]
            summary = summarize(pw, [*ABSTRACT_METRICS, "stopped_not_worth_cost"])
            out["partitions"][part.value] = {
                **{k: v for k, v in res.items() if k != "per_world"},
                "n_worlds": len(pw),
                "summary": summary,
                "ranking_by_primary": sorted(summary, key=lambda p: -summary[p][PRIMARY]["mean"]),
                "paired_production_vs": paired(pw, PRODUCTION, list(summary), ABSTRACT_METRICS),
                "per_world": pw,
            }
    _write(out_dir, "active_mcbr_e002.json", out)
    return {k: v for k, v in out.items() if k != "partitions"} | {
        "n_worlds": {k: v["n_worlds"] for k, v in out["partitions"].items()}
    }


# ---------------------------------------------------------------------------------------------- E003 (mission)
MISSION_CONFIG = "configs/sim/mission_test_small.yaml"


def _mission_job(args: tuple[Any, ...]) -> dict[str, Any]:
    seed, family, planner, runtime, keep_root = args
    from uuid import UUID

    from conrad.orchestration.evaluation import evaluate_run_dir
    from conrad.persistence.db import make_engine
    from conrad.persistence.repository import Repository
    from conrad.schemas.world import Domain
    from conrad.settings import load_settings
    from conrad.sim.mission.run import run_scenario

    s = load_settings(MISSION_CONFIG)
    rt = {**dict(s.sim.get("mission", {}).get("runtime", {})), **runtime, "planner": planner}
    upd = {"sim": {**dict(s.sim), "mission": {"world": {"family": family}, "runtime": rt}}}
    s = s.model_copy(update={"run": s.run.model_copy(update={"seed": int(seed)}), **upd})
    root = Path(tempfile.mkdtemp(prefix="mcbr-e003-"))
    out = run_scenario("FLAGSHIP-I4", s, run_id=f"E003-{seed}-{planner}", runs_root=root)
    d = Path(out["run_dir"])
    rep = evaluate_run_dir(d)
    rtm = json.loads((d / "mission" / "runtime_metrics.json").read_text(encoding="utf-8"))
    truth = json.loads((d / "truth" / "truth_record.json").read_text(encoding="utf-8"))
    goals = json.loads((d / "mission" / "trajectories.json").read_text(encoding="utf-8"))["goals"]
    target = UUID(rep["target_registry_id"])
    engine = make_engine(d / "conrad.sqlite")
    try:
        direct_t = sorted(
            r.measurement_time_ns / 1e9
            for r in Repository(engine).all_revisions()
            if r.cell.domain is Domain.TECHNICAL
            and r.cell.registry_entity_id == target
            and r.update_kind.value == "DIRECT"
        )
    finally:
        engine.dispose()
    insp = [g["t_s"] for g in goals if g["purpose"] == "INSPECT" and g["accepted"]]
    ends = [*insp[1:], math.inf] if insp else []
    informative = sum(1 for t0, t1 in zip(insp, ends, strict=True) if any(t0 < t <= t1 for t in direct_t))
    traj = np.asarray([row[1:4] for row in truth["trajectory"]], dtype=np.float64)
    travel = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()) if len(traj) > 1 else 0.0
    ea = rep["target_error_after"] or {}
    eb = rep["target_error_before"] or {}
    norm = []
    raw_after = {}
    for q in ("corrosion_depth_m", "crack_length_m"):
        prior = ea.get(f"{q}.prior_mean_abs_error") or eb.get(f"{q}.prior_mean_abs_error")
        err = ea.get(f"{q}.abs_error")
        err = prior if err is None else err  # UNKNOWN at the end = the prior mean, not a guessed value
        raw_after[q] = err
        norm.append(float(err) / float(prior) if (prior and err is not None) else 1.0)
    hse = float(np.mean(norm))
    energy = float(truth["vehicle"]["energy_used_j"])
    duration = float(rt.get("duration_s", 100.0))
    comm = rtm["communication"]
    after = rep["target_after"] or {}
    row = {
        "seed": int(seed),
        "family": family,
        "planner": planner,
        "planner_name_in_run": (rtm["plans"][0]["planner"] if rtm["plans"] else planner),
        "hidden_state_error_final": hse,
        "hidden_state_error_improvement": 1.0 - hse,  # relative to the prior-mean (UNKNOWN) error
        "mission_error_reduction": 1.0 - hse,  # the target segment IS the mission-relevant belief
        "corrosion_abs_error_m": raw_after["corrosion_depth_m"],
        "crack_abs_error_m": raw_after["crack_length_m"],
        "info_per_time": (1.0 - hse) / duration,
        "info_per_kj": 1000.0 * (1.0 - hse) / energy if energy > 0 else 0.0,
        "observations": float(len(insp)),
        "redundant_observations": float(len(insp) - informative),
        "travel_m": travel,
        "time_s": duration,
        "energy_j": energy,
        "collisions": float(truth["vehicle"]["collisions"]),
        "target_observed": float(after.get("condition.status") == "OBSERVED"),
        "critical_delivered": float(comm["critical_delivered"] >= 1),
        "mission_success": float(
            after.get("condition.status") == "OBSERVED" and comm["critical_delivered"] >= 1
        ),
        "patch_max_visible_fraction": rep["patch_max_visible_fraction"],
        "inspection_goal_t_s": rep["inspection_goal_t_s"],
        "plans": [p["status"] for p in rtm["plans"]],
        "uir": rtm["uir"],
    }
    if keep_root:
        dest = Path(keep_root) / d.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(d, dest)
        row["run_dir"] = str(dest)
    shutil.rmtree(root, ignore_errors=True)
    return row


MISSION_METRICS = {
    "hidden_state_error_improvement": "higher",
    "mission_error_reduction": "higher",
    "hidden_state_error_final": "lower",
    "info_per_time": "higher",
    "info_per_kj": "higher",
    "redundant_observations": "lower",
    "observations": "lower",
    "travel_m": "lower",
    "energy_j": "lower",
    "collisions": "lower",
    "mission_success": "higher",
}


def evaluate_mission(
    split: P.Split, planners: list[str], config: dict[str, Any], keep_root: Path | None
) -> dict[str, Any]:
    seeds = list(split.world_seeds)[: config.get("max_worlds") or None]
    fams = split.families
    runtime = dict(config.get("runtime", {}))
    jobs = [
        (s, fams[s % len(fams)], p, runtime, str(keep_root) if keep_root else None)
        for s in seeds
        for p in planners
    ]
    rows: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=int(config.get("workers", 8))) as ex:
        futs = [ex.submit(_mission_job, j) for j in jobs]
        for f in cf.as_completed(futs):
            rows.append(f.result())
    per_world: dict[str, dict[str, Any]] = {}
    for r in rows:
        per_world.setdefault(str(r["seed"]), {})[r["planner"]] = r
    return {
        "partition": split.partition.value,
        "partition_digest": split.digest,
        "families": list(fams),
        "world_seeds": seeds,
        "replicates_per_world": 1,
        "per_world": per_world,
    }


def run_e003(config: dict[str, Any], seeds: list[int], out_dir: Path) -> dict[str, Any]:
    """Integrated SURROGATE mission (python kernel). Not Unity: the formal Unity run belongs to the integrator."""
    frozen = load_frozen()
    planners = list(config.get("planners", [PRODUCTION, *I4_COMPARATORS, "A-B10_mcbr_full"]))
    keep = out_dir / "runs" if config.get("keep_runs", False) else None
    out: dict[str, Any] = {
        "experiment_id": E003,
        "data_status": "SYNTHETIC_ONLY",
        "evidence_kind": "SURROGATE",
        "evidence_note": "integrated surrogate mission on the python sim kernel (L1); NOT the formal Unity run",
        "scenario": "FLAGSHIP-I4",
        "mission_config": MISSION_CONFIG,
        "frozen_planner": frozen,
        "primary_metric": "hidden_state_error_improvement",
        "i4_comparators": list(I4_COMPARATORS),
        "config": config,
        "partitions": {},
    }
    parts = [P.Partition(p) for p in config.get("partitions", ["final_test", "ood_test"])]
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        for part in parts:
            split = P.split("mission", part, P.Purpose.FINAL_EVALUATION)
            res = evaluate_mission(split, planners, config, keep)
            pw = res["per_world"]
            summary = summarize(pw, [*MISSION_METRICS, "target_observed", "critical_delivered"])
            out["partitions"][part.value] = {
                **{k: v for k, v in res.items() if k != "per_world"},
                "n_worlds": len(pw),
                "summary": summary,
                "paired_production_vs": paired(pw, PRODUCTION, planners, MISSION_METRICS),
                "per_world": pw,
            }
    _write(out_dir, "active_mcbr_e003.json", out)
    return {k: v for k, v in out.items() if k != "partitions"} | {
        "n_worlds": {k: v["n_worlds"] for k, v in out["partitions"].items()}
    }


STAGE_PARTITIONS: dict[str, tuple[str, tuple[P.Partition, ...], P.Purpose]] = {
    "design": ("abstract", (P.Partition.DEVELOPMENT,), P.Purpose.DESIGN),
    "selection": ("abstract", (P.Partition.VALIDATION,), P.Purpose.SELECTION),
    "e002": ("abstract", (P.Partition.FINAL_TEST, P.Partition.OOD_TEST), P.Purpose.FINAL_EVALUATION),
    "e003": ("mission", (P.Partition.FINAL_TEST, P.Partition.OOD_TEST), P.Purpose.FINAL_EVALUATION),
}


def check_seeds(stage: str, seeds: list[int]) -> None:
    """Seeds handed in by a caller must lie inside the stage's partitions (seeds are never free inputs)."""
    domain, parts, purpose = STAGE_PARTITIONS[stage]
    allowed: set[int] = set()
    for part in parts:
        allowed |= set(P.split(domain, part, purpose).world_seeds)
    stray = sorted({int(s) for s in seeds} - allowed)
    if stray:
        raise P.PartitionAccessError(f"stage {stage!r} got seeds outside its partitions: {stray[:10]}")


def run(config: dict[str, Any], seeds: list[int], out_dir: str | Path) -> dict[str, Any]:
    """Dispatch entry: ``config['stage']`` in {design, selection, e002, e003}."""
    stage = str(config.get("stage"))
    check_seeds(stage, seeds)
    fn = {"design": run_design, "selection": run_selection, "e002": run_e002, "e003": run_e003}[stage]
    return fn(config, seeds, Path(out_dir))


__all__ = [
    "E002",
    "E003",
    "I4_COMPARATORS",
    "PRIMARY",
    "SEL",
    "Budget",
    "RankerConfig",
    "build_planners",
    "evaluate_abstract",
    "paired",
    "ranker_planner",
    "run",
    "run_abstract_episode",
    "run_e002",
    "run_e003",
    "run_selection",
]
