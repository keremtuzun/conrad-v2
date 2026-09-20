"""ACTIVE-MCBR-E005: gate I4 on the world family ACTIVE_INSPECTION_OCCLUDED_V1. SYNTHETIC_ONLY.

``docs/audits/I4_WORLD_FAMILY.md`` declares the family, the splits, the planner set and the metrics BEFORE
any run. The family makes the ch25 premise true by construction: the defect sits on the side of the target
the transit lane does not see, and two unregistered rack panels leave one angular window, so several candidate
observations exist and only a few of them resolve the ambiguity.

Stages (``config['stage']``):

    design      DEVELOPMENT worlds, purpose design. Diagnoses the family and compares the planner set.
    selection   VALIDATION worlds, purpose selection. Picks the production planner. Nothing else may read it.
    e005        FINAL_TEST worlds, purpose final_evaluation, run once.
    e005_ood    OOD_TEST worlds (held-out structural family), purpose final_evaluation, run once.

Every planner receives the identical ``PlanningRequest`` (predictive model included), the identical candidate
generator and feasibility filter, and matched budgets: the same mission duration and time budget, the same
``max_plans_per_need`` observation budget, the same single structural sensor and sensor capability. Travel
allowance, energy and travel are measured on the same scale for every planner and reported.

Scores are ACTUAL hidden-state errors against the evaluation-only truth, never uncertainty.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from conrad.active.production import PRODUCTION, load_frozen
from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.active_mcbr_reeval import (
    MISSION_CONFIG,
    MISSION_METRICS,
    paired,
    score_mission_run,
    summarize,
)

E005 = "ACTIVE-MCBR-E005"
FAMILY_ID = "ACTIVE_INSPECTION_OCCLUDED_V1"
SCENARIO = "I4-OCCLUDED"
OOD_SCENARIO = "I4-OCCLUDED-OOD"
PRIMARY = "hidden_state_error_improvement"
I4_COMPARATORS = ("A-B1_fixed_inspection", "A-B0_random", "A-B2_coverage")
#: the complete comparison required by the brief: random, fixed nominal route, coverage-only, geometric NBV,
#: entropy/uncertainty NBV, a conventional expected-information-gain baseline, the old MCBR and PRODUCTION.
PLANNERS: tuple[str, ...] = (
    PRODUCTION,
    "A-B1_fixed_inspection",
    "A-B0_random",
    "A-B2_coverage",
    "A-B4_geometric_nbv",
    "A-B5_entropy_nbv",
    "A-B5b_entropy_nbv_predictive",
    "A-B7_uncertainty_nbv",
    "A-B6b_bayes_eig",
    "A-B10_mcbr_full",
)
STAGES: dict[str, tuple[P.Partition, P.Purpose]] = {
    "design": (P.Partition.DEVELOPMENT, P.Purpose.DESIGN),
    "selection": (P.Partition.VALIDATION, P.Purpose.SELECTION),
    "e005": (P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION),
    "e005_ood": (P.Partition.OOD_TEST, P.Purpose.FINAL_EVALUATION),
}
EXTRA_METRICS = ("target_observed", "critical_delivered", "patch_max_visible_fraction")


def select_validation_winner(
    per_world: dict[str, dict[str, Any]], arms: tuple[str, ...] | list[str]
) -> dict[str, Any]:
    """Apply the predeclared validation rule without reading any final or OOD result."""
    candidates = [a for a in arms if a not in I4_COMPARATORS]
    comparisons = {a: paired(per_world, a, list(I4_COMPARATORS), MISSION_METRICS) for a in candidates}
    eligible = [
        a for a in candidates if all(comparisons[a][b][PRIMARY]["ci95"][0] > 0.0 for b in I4_COMPARATORS)
    ]
    means = {a: float(np.mean([per_world[w][a][PRIMARY] for w in per_world])) for a in candidates}
    winner = max(eligible, key=lambda a: (means[a], a)) if eligible else None
    return {
        "candidates": candidates,
        "required_comparators": list(I4_COMPARATORS),
        "pairwise": comparisons,
        "primary_means": means,
        "eligible": eligible,
        "winner": winner,
        "decision": (
            "freeze winner as production MCBR before any final run"
            if winner is not None
            else "no arm beat fixed, random and coverage with a positive paired CI; freeze nothing"
        ),
    }


def _job(args: tuple[Any, ...]) -> dict[str, Any]:
    """One (world, arm) mission on the python sim kernel. ``variant`` overrides the planner section only."""
    seed, scenario, arm, planner, variant, runtime = args
    from conrad.settings import load_settings
    from conrad.sim.mission.run import run_scenario

    s = load_settings(MISSION_CONFIG)
    rt = {**dict(s.sim.get("mission", {}).get("runtime", {})), **runtime, "planner": planner}
    s = s.model_copy(
        update={
            "run": s.run.model_copy(update={"seed": int(seed)}),
            "sim": {**dict(s.sim), "mission": {"world": {}, "runtime": rt}},
        }
    )
    if variant is not None:  # design / selection arms only: a candidate production planner section
        import conrad.orchestration.deliberation as D
        import conrad.orchestration.mission_predictive as MP
        from conrad.active.production import build_planner
        from conrad.active.surface_predictive import SurfacePredictiveConfig

        section = {k: v for k, v in dict(variant).items() if k != "mission_predictive"}
        original, original_cfg = D.production_planner, MP.production_predictive_config
        D.production_planner = lambda ids, cfg, path=None: build_planner(  # type: ignore[assignment]
            section, ids, cfg, f"VARIANT[{arm}]"
        )
        if "mission_predictive" in variant:
            raw = variant["mission_predictive"]
            override = None if raw is None else SurfacePredictiveConfig(**raw)
            MP.production_predictive_config = lambda: override
        try:
            return _run_and_score(scenario, s, seed, arm, planner, rt, run_scenario)
        finally:
            D.production_planner = original
            MP.production_predictive_config = original_cfg
    return _run_and_score(scenario, s, seed, arm, planner, rt, run_scenario)


def _run_and_score(
    scenario: str, s: Any, seed: int, arm: str, planner: str, rt: dict[str, Any], run_scenario: Any
) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix="mcbr-e005-"))
    try:
        out = run_scenario(scenario, s, run_id=f"E005-{seed}-{arm}", runs_root=root)
        d = Path(out["run_dir"])
        row = score_mission_run(d, int(seed), scenario, planner, float(rt.get("duration_s", 100.0)))
        row["planner"] = arm
        row["planner_config"] = planner
        row["scenario"] = scenario
        return row
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _evaluate(
    seeds: list[int],
    scenario: str,
    arms: dict[str, dict[str, Any]],
    config: dict[str, Any],
    checkpoint: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Every (world, arm) once. ``checkpoint`` (jsonl) lets a crashed run resume without re-scoring."""
    runtime = dict(config.get("runtime", {}))
    per_world: dict[str, dict[str, Any]] = {}
    if checkpoint is not None and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            per_world.setdefault(str(r["seed"]), {})[r["planner"]] = r
    jobs = [
        (s, scenario, a, v["planner"], v.get("variant"), runtime)
        for s in seeds
        for a, v in arms.items()
        if a not in per_world.get(str(s), {})
    ]
    with cf.ProcessPoolExecutor(max_workers=int(config.get("workers", 8))) as ex:
        for fut in cf.as_completed([ex.submit(_job, j) for j in jobs]):
            r = fut.result()
            per_world.setdefault(str(r["seed"]), {})[r["planner"]] = r
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    return {str(s): per_world[str(s)] for s in seeds if str(s) in per_world}


def _write(out_dir: Path, name: str, obj: Any) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")


def arms_of(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    arms = {p: {"planner": p} for p in config.get("planners", PLANNERS)}
    for name, section in (config.get("variants") or {}).items():
        arms[name] = {"planner": PRODUCTION, "variant": section}
    return arms


def run_stage(stage: str, config: dict[str, Any], seeds: list[int], out_dir: Path) -> dict[str, Any]:
    partition, purpose = STAGES[stage]
    scenario = OOD_SCENARIO if partition is P.Partition.OOD_TEST else SCENARIO
    arms = arms_of(config)
    with P.purpose_scope(purpose):
        split = P.split(P.I4_OCCLUDED_DOMAIN, partition, purpose)
        allowed = set(split.world_seeds)
        stray = sorted({int(s) for s in seeds} - allowed)
        if stray:
            raise P.PartitionAccessError(f"stage {stage!r} got seeds outside its partition: {stray[:10]}")
        use = [int(s) for s in seeds] or list(split.world_seeds)[: config.get("max_worlds") or None]
        pw = _evaluate(use, scenario, arms, config, out_dir / f"checkpoint_{stage}.jsonl")
    reference = str(config.get("reference_arm", PRODUCTION))
    metrics = dict(MISSION_METRICS)
    result = {
        "experiment_id": f"{E005}-{stage.upper()}",
        "data_status": "SYNTHETIC_ONLY",
        "evidence_kind": "FORMAL" if config.get("evidence_kind") == "FORMAL" else "SURROGATE",
        "evidence_note": "integrated mission on the python sim kernel (L1); the formal path is Unity",
        "world_family": FAMILY_ID,
        "scenario": scenario,
        "mission_config": MISSION_CONFIG,
        "stage": stage,
        "partition": partition.value,
        "partition_file": "configs/eval/partitions_i4_occluded.yaml",
        "partition_digest": split.digest,
        "families": list(split.families),
        "world_seeds": use,
        "n_worlds": len(pw),
        "frozen_planner": load_frozen(),
        "primary_metric": PRIMARY,
        "i4_comparators": list(I4_COMPARATORS),
        "decision_rule": config.get("decision_rule"),
        "reference_arm": reference,
        "config": config,
        "summary": summarize(pw, [*metrics, *EXTRA_METRICS]),
        "paired_reference_vs": paired(pw, reference, list(arms), metrics),
        "per_world": pw,
    }
    if stage == "selection":
        result["selection"] = select_validation_winner(pw, list(arms))
    _write(out_dir, str(config.get("out_name", f"active_mcbr_e005_{stage}.json")), result)
    return {k: v for k, v in result.items() if k not in ("per_world", "config")}


def run(config: dict[str, Any], seeds: list[int], out_dir: str | Path) -> dict[str, Any]:
    stage = str(config.get("stage"))
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; known {sorted(STAGES)}")
    return run_stage(stage, config, [int(s) for s in seeds], Path(out_dir))


__all__ = [
    "E005",
    "FAMILY_ID",
    "PLANNERS",
    "PRIMARY",
    "SCENARIO",
    "arms_of",
    "run",
    "run_stage",
    "select_validation_winner",
]
