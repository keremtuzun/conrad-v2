"""ACTIVE-MCBR-E004: integrated SURROGATE mission for gate I4 with the belief-side predictive model. SYNTHETIC_ONLY.

The mission runtime now hands MCBR a belief-side predictive model (``conrad.orchestration.mission_predictive``):
Model2T surface-cell coverage, locus and variances, the declared Model2T sensor datasheet, Model2S ray-cast
occlusion and the shared navigation cost. The production planner is the frozen
``configs/active/mcbr_frozen_v2.yaml`` (same validation-selected ranker as v1, plus the frozen predictive model).

Stages (``config['stage']``):

    design   DEVELOPMENT worlds only (``unity_gate`` development 7810000-7810019 and, optionally, ``mission``
             development/validation), purpose design. Arms may override the predictive model config; that
             override is applied inside each worker process and exists only on this evaluation path.
    e004     the pre-declared I4 worlds of ``unity_gate`` final_test (``config['i4_worlds']``), purpose
             final_evaluation, run once. Planners are compared on the SAME worlds with the SAME mission duration
             (time budget), the same ``max_plans_per_need`` (observation budget), the same single structural
             sensor, the same candidate generator and feasibility filter and the same PlanningRequest (energy and
             travel are measured). Paired percentile bootstrap over worlds (95 %, 4000 resamples).

Decision rule (declared in ``configs/eval/active_mcbr_e004.yaml`` before the run): PRODUCTION "beats" a
baseline on a metric iff the lower bound of the paired 95 % CI of the benefit is > 0.

Scores are ACTUAL hidden-state errors against the evaluation-only truth (never uncertainty).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
from pathlib import Path
from typing import Any

from conrad.active.production import PRODUCTION, load_frozen
from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.active_mcbr_reeval import (
    MISSION_CONFIG,
    MISSION_METRICS,
    _mission_job,
    paired,
    summarize,
)

E004 = "ACTIVE-MCBR-E004"
I4_COMPARATORS = ("A-B1_fixed_inspection", "A-B0_random", "A-B2_coverage")
PRIMARY = "hidden_state_error_improvement"
FAMILY = "straight_pipeline"


def _arm_job(args: tuple[Any, ...]) -> dict[str, Any]:
    seed, arm, planner, pcfg, runtime = args
    if pcfg != "FROZEN":  # design arms only: override the frozen predictive model inside this worker
        import conrad.orchestration.mission_predictive as mp
        from conrad.active.surface_predictive import SurfacePredictiveConfig

        override = None if pcfg is None else SurfacePredictiveConfig(**pcfg)
        mp.production_predictive_config = lambda: override
    row = _mission_job((seed, FAMILY, planner, runtime, None))
    row["planner"] = arm
    row["planner_config"] = planner
    return row


def _evaluate(
    seeds: list[int], arms: dict[str, dict[str, Any]], config: dict[str, Any], checkpoint: Path | None = None
) -> dict[str, Any]:
    """Every (world, arm) once. ``checkpoint`` (jsonl) keeps finished jobs, so a crashed run (e.g. out of memory)
    resumes without re-running or re-scoring anything; it is never read for any other purpose."""
    runtime = dict(config.get("runtime", {}))
    per_world: dict[str, dict[str, Any]] = {}
    if checkpoint is not None and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            per_world.setdefault(str(r["seed"]), {})[r["planner"]] = r
    jobs = [
        (s, a, v["planner"], v.get("predictive", "FROZEN"), runtime)
        for s in seeds
        for a, v in arms.items()
        if a not in per_world.get(str(s), {})
    ]
    with cf.ProcessPoolExecutor(max_workers=int(config.get("workers", 8))) as ex:
        for fut in cf.as_completed([ex.submit(_arm_job, j) for j in jobs]):
            r = fut.result()
            per_world.setdefault(str(r["seed"]), {})[r["planner"]] = r
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    return {str(s): per_world[str(s)] for s in seeds}


def _write(out_dir: Path, name: str, obj: Any) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")


def declared_i4_worlds(config: dict[str, Any]) -> list[int]:
    """The pre-declared I4 worlds; they must be unity_gate final_test seeds and not another gate's role."""
    seeds = [int(s) for s in config["i4_worlds"]]
    split = P.split(P.UNITY_GATES_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION)
    roles = set(P.load_unity_gates()["raw"].get("roles", {}).values())
    stray = sorted(set(seeds) - set(split.world_seeds))
    if stray or set(seeds) & roles or len(set(seeds)) != len(seeds):
        raise P.PartitionAccessError(
            f"I4 worlds must be distinct unity_gate final_test non-role seeds: {seeds}"
        )
    return seeds


def run_design(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    arms = dict(config["arms"])
    seeds: list[int] = []
    with P.purpose_scope(P.Purpose.DESIGN):
        for spec in config.get(
            "design_splits", [{"domain": P.UNITY_GATES_DOMAIN, "partition": "development"}]
        ):
            s = P.split(spec["domain"], spec["partition"], P.Purpose.DESIGN)
            seeds += list(s.world_seeds)[: spec.get("max_worlds") or None]
    pw = _evaluate(seeds, arms, config)
    ref = str(config.get("reference_arm", next(iter(arms))))
    metrics = [*MISSION_METRICS, "target_observed"]
    result = {
        "experiment_id": E004 + "-DESIGN",
        "purpose": "design",
        "data_status": "SYNTHETIC_ONLY",
        "world_seeds": seeds,
        "config": config,
        "summary": summarize(pw, metrics),
        "paired_reference_vs": paired(pw, ref, list(arms), {PRIMARY: "higher", "info_per_kj": "higher"}),
        "per_world": pw,
    }
    _write(out_dir, str(config.get("out_name", "design.json")), result)
    return {k: v for k, v in result.items() if k != "per_world"}


def run_e004(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    frozen = load_frozen()
    planners = list(config.get("planners", [PRODUCTION, *I4_COMPARATORS, "A-B10_mcbr_full"]))
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        seeds = declared_i4_worlds(config)
        digest = P.load_unity_gates()["digest"]
        pw = _evaluate(seeds, {p: {"planner": p} for p in planners}, config, out_dir / "checkpoint.jsonl")
    metrics = dict(MISSION_METRICS)
    summary = summarize(pw, [*metrics, "target_observed", "critical_delivered"])
    part = {
        "partition": "final_test",
        "partition_file": "configs/eval/partitions_unity_gates.yaml",
        "partition_digest": digest,
        "families": [FAMILY],
        "world_seeds": seeds,
        "n_worlds": len(pw),
        "summary": summary,
        "paired_production_vs": paired(pw, PRODUCTION, planners, metrics),
        "per_world": pw,
    }
    out = {
        "experiment_id": E004,
        "data_status": "SYNTHETIC_ONLY",
        "evidence_kind": "SURROGATE",
        "evidence_note": "integrated surrogate mission on the python sim kernel (L1); NOT the formal Unity run",
        "scenario": "FLAGSHIP-I4",
        "mission_config": MISSION_CONFIG,
        "frozen_planner": frozen,
        "primary_metric": PRIMARY,
        "i4_comparators": list(I4_COMPARATORS),
        "decision_rule": config.get("decision_rule"),
        "config": config,
        "partitions": {"final_test": part},
    }
    _write(out_dir, "active_mcbr_e004.json", out)
    return {k: v for k, v in out.items() if k != "partitions"} | {"n_worlds": len(pw)}


def run(config: dict[str, Any], seeds: list[int], out_dir: str | Path) -> dict[str, Any]:
    """Dispatch entry. ``seeds`` must equal the declared I4 worlds (stage e004) or be empty (design)."""
    stage = str(config.get("stage"))
    if stage == "design":
        return run_design(config, Path(out_dir))
    if stage == "e004":
        if seeds and [int(s) for s in seeds] != [int(s) for s in config["i4_worlds"]]:
            raise P.PartitionAccessError("ACTIVE-MCBR-E004 runs only on its declared I4 worlds")
        return run_e004(config, Path(out_dir))
    raise ValueError(f"unknown stage {stage!r}")


__all__ = ["E004", "I4_COMPARATORS", "PRIMARY", "declared_i4_worlds", "run", "run_design", "run_e004"]
