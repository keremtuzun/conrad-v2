"""ACTIVE-MCBR-E006 (I4 oracle headroom): how much active-planning headroom does ACTIVE_INSPECTION_OCCLUDED_V1 actually have?

DEVELOPMENT worlds only, SYNTHETIC_ONLY, python sim kernel (L1 surrogate: it never promotes a gate).

Gate I4 is decided and final: the production MCBR planner beats a fixed route and random views and does not
beat a systematic coverage sweep. Before any V4 mechanism is designed, this experiment measures the ceiling.
It replaces the ranking rule with an evaluation-only oracle that SEES TRUTH
(``conrad.evaluation.oracle.i4_view_oracle``) and picks the candidate that would actually reveal most of the
hidden defect, one-step and with two-step lookahead. Everything else is identical for every arm: the same
mission, duration, ``max_plans_per_need`` budget, sensors, candidate generator, feasibility filter and
``PlanningRequest``.

    if the ORACLE cannot beat coverage-only, no ranking rule can, and a V4 that only re-ranks views is not
    worth building. If the oracle beats coverage clearly, the headroom exists and V4 has something to aim at.

The oracle is injected exactly as ACTIVE-MCBR-E005 injects a candidate production planner: this module
rebinds ``conrad.orchestration.deliberation.production_planner`` inside its own worker process. No runtime
module imports it; ``tests/leakage`` forbids the deployment packages from importing ``conrad.evaluation``.

Arms:

    A-B2_coverage        the systematic sweep that gate I4 could not beat (also the truth-harvest pass)
    ORACLE_1STEP         ranks by the defect surface that view alone would newly reveal
    ORACLE_2STEP         ranks by the best two-view union that starts with that view
    ORACLE_1STEP_RATIO   the same truth value under the FROZEN planner's own cost rule, value / (0.05 + cost),
                         so that ranking quality is isolated at the incumbent's cost sensitivity
    ORACLE_1STEP_WEIGHTED        as ORACLE_1STEP, but cells count by cos(incidence) rather than 0/1, which is
                                 the truth-side twin of the production predictive model's own cell weight
    ORACLE_1STEP_WEIGHTED_RATIO  the weighted oracle under the frozen planner's cost rule
    PRODUCTION           the frozen V3 planner, reported for reference

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from conrad.active.config import MCBRConfig
from conrad.active.planner import MCBRPlanner
from conrad.active.production import FROZEN_PATH, PRODUCTION
from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.active_mcbr_reeval import (
    MISSION_CONFIG,
    MISSION_METRICS,
    paired,
    score_mission_run,
    summarize,
)
from conrad.evaluation.oracle.i4_view_oracle import ARMS, TruthView, harvest
from conrad.schemas.ids import IdFactory

EXPERIMENT = "ACTIVE-MCBR-E006"
SCENARIO = "I4-OCCLUDED"
FAMILY_ID = "ACTIVE_INSPECTION_OCCLUDED_V1"
PRIMARY = "hidden_state_error_improvement"
BASELINE = "A-B2_coverage"
ORACLES = tuple(ARMS)
EXTRA_METRICS = ("target_observed", "patch_max_visible_fraction")


def _settings(seed: int, runtime: dict[str, Any], planner: str) -> Any:
    from conrad.settings import load_settings

    s = load_settings(MISSION_CONFIG)
    rt = {**dict(s.sim.get("mission", {}).get("runtime", {})), **runtime, "planner": planner}
    return s.model_copy(
        update={
            "run": s.run.model_copy(update={"seed": int(seed)}),
            "sim": {**dict(s.sim), "mission": {"world": {}, "runtime": rt}},
        }
    )


def _harvest_job(args: tuple[int, dict[str, Any], float]) -> dict[str, Any]:
    """One baseline mission, kept long enough to harvest the world's truth for the oracle arms."""
    seed, runtime, duration = args
    from conrad.sim.mission.run import run_scenario

    root = Path(tempfile.mkdtemp(prefix="i4-oracle-harvest-"))
    try:
        out = run_scenario(
            SCENARIO, _settings(seed, runtime, BASELINE), run_id=f"HARVEST-{seed}", runs_root=root
        )
        run_dir = Path(out["run_dir"])
        row = score_mission_run(run_dir, seed, FAMILY_ID, BASELINE, duration)
        row["planner"] = BASELINE
        return {"seed": seed, "row": row, "truth": harvest(run_dir).to_json()}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _oracle_job(args: tuple[int, str, dict[str, Any], float, dict[str, Any]]) -> dict[str, Any]:
    """One oracle mission. The oracle is bound here, in the evaluation worker, never in a runtime module."""
    seed, arm, runtime, duration, truth_json = args
    import conrad.orchestration.deliberation as deliberation
    from conrad.evaluation.oracle.i4_view_oracle import oracle_planner
    from conrad.sim.mission.run import run_scenario

    truth = TruthView.from_json(truth_json)
    steps, cost_mode, weighted = ARMS[arm]
    original = deliberation.production_planner

    def _inject(ids: IdFactory, cfg: MCBRConfig, path: Path = FROZEN_PATH) -> MCBRPlanner:
        return oracle_planner(ids, cfg, truth, steps, cost_mode, weighted, arm)

    deliberation.production_planner = _inject
    root = Path(tempfile.mkdtemp(prefix="i4-oracle-"))
    try:
        out = run_scenario(
            SCENARIO, _settings(seed, runtime, PRODUCTION), run_id=f"{arm}-{seed}", runs_root=root
        )
        row = score_mission_run(Path(out["run_dir"]), seed, FAMILY_ID, arm, duration)
        row["planner"] = arm
        return {"seed": seed, "row": row}
    finally:
        deliberation.production_planner = original
        shutil.rmtree(root, ignore_errors=True)


def _plain_job(args: tuple[int, str, dict[str, Any], float]) -> dict[str, Any]:
    seed, arm, runtime, duration = args
    from conrad.sim.mission.run import run_scenario

    root = Path(tempfile.mkdtemp(prefix="i4-plain-"))
    try:
        out = run_scenario(SCENARIO, _settings(seed, runtime, arm), run_id=f"{arm}-{seed}", runs_root=root)
        row = score_mission_run(Path(out["run_dir"]), seed, FAMILY_ID, arm, duration)
        row["planner"] = arm
        return {"seed": seed, "row": row}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run(config: dict[str, Any], seeds: list[int], out_dir: str | Path) -> dict[str, Any]:
    """Development worlds only. Any seed outside the declared development split is refused."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime = dict(config.get("runtime", {}))
    duration = float(runtime.get("duration_s", 100.0))
    workers = int(config.get("workers", 8))
    extra = [str(a) for a in config.get("reference_arms", [PRODUCTION])]
    oracles = [str(a) for a in config.get("oracle_arms", ORACLES)]

    with P.purpose_scope(P.Purpose.DESIGN):
        split = P.split(P.I4_MCBR_V4_DOMAIN, P.Partition.DEVELOPMENT, P.Purpose.DESIGN)
        allowed = set(split.world_seeds)
        stray = sorted({int(s) for s in seeds} - allowed)
        if stray:
            raise P.PartitionAccessError(f"seeds outside the development split: {stray[:10]}")
        use = [int(s) for s in seeds] or list(split.world_seeds)[: config.get("max_worlds") or None]

        per_world: dict[str, dict[str, Any]] = {}
        truths: dict[int, dict[str, Any]] = {}
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in cf.as_completed([ex.submit(_harvest_job, (s, runtime, duration)) for s in use]):
                r = fut.result()
                per_world.setdefault(str(r["seed"]), {})[BASELINE] = r["row"]
                truths[int(r["seed"])] = r["truth"]
            jobs = [
                ex.submit(_oracle_job, (s, a, runtime, duration, truths[s])) for s in use for a in oracles
            ] + [ex.submit(_plain_job, (s, a, runtime, duration)) for s in use for a in extra]
            for fut in cf.as_completed(jobs):
                r = fut.result()
                per_world.setdefault(str(r["seed"]), {})[r["row"]["planner"]] = r["row"]

    pw = {str(s): per_world[str(s)] for s in use if str(s) in per_world}
    arms = [BASELINE, *oracles, *extra]
    result = {
        "experiment_id": str(config.get("experiment_id", EXPERIMENT)),
        "data_status": "SYNTHETIC_ONLY",
        "evidence_kind": "SURROGATE",
        "evidence_note": "python sim kernel; it measures headroom on DEVELOPMENT worlds and promotes no gate",
        "world_family": FAMILY_ID,
        "scenario": SCENARIO,
        "mission_config": MISSION_CONFIG,
        "partition_file": "configs/eval/partitions_i4_mcbr_v4.yaml",
        "partition": split.partition.value,
        "partition_digest": split.digest,
        "world_seeds": use,
        "n_worlds": len(pw),
        "arms": arms,
        "baseline": BASELINE,
        "primary_metric": PRIMARY,
        "question": config.get("question"),
        "decision_rule": config.get("decision_rule"),
        "config": config,
        "summary": summarize(pw, [*MISSION_METRICS, *EXTRA_METRICS]),
        "paired_vs_coverage": {
            a: paired(pw, a, [BASELINE], dict(MISSION_METRICS))[BASELINE] for a in arms if a != BASELINE
        },
        "per_world": pw,
    }
    result["headroom"] = headroom(pw, arms)
    name = str(config.get("out_name", "i4_oracle_headroom.json"))
    (out / name).write_text(json.dumps(result, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return {k: v for k, v in result.items() if k not in ("per_world", "config")}


def headroom(pw: dict[str, dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    """The all-or-nothing decomposition: how often each arm reads the defect at all, and what it gets."""
    out: dict[str, Any] = {}
    for a in arms:
        rows = [w[a] for w in pw.values() if a in w]
        read = [r for r in rows if r[PRIMARY] > 1e-9]
        out[a] = {
            "n": len(rows),
            "worlds_with_the_defect_read": len(read),
            "fraction_read": len(read) / len(rows) if rows else None,
            "mean_when_read": float(np.mean([r[PRIMARY] for r in read])) if read else None,
            "mean": float(np.mean([r[PRIMARY] for r in rows])) if rows else None,
            "observations": float(np.mean([r["observations"] for r in rows])) if rows else None,
        }
    base = out.get(BASELINE, {})
    for a in arms:
        if a == BASELINE or not base.get("fraction_read"):
            continue
        out[a]["extra_worlds_read_vs_coverage"] = (
            out[a]["worlds_with_the_defect_read"] - base["worlds_with_the_defect_read"]
        )
    return out


__all__ = ["EXPERIMENT", "ORACLES", "PRIMARY", "headroom", "run"]
