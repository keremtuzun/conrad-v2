"""ACTIVE-MCBR-E007 (MCBR V4): does the mission fly the view MCBR chose, and does that change the read rate?

SYNTHETIC_ONLY, python sim kernel (L1 SURROGATE: it promotes no gate and changes no recorded result).
Gate I4 stays FAIL exactly as ``artifacts/gates/I4/evidence_formal.json`` records it.

``ACTIVE-MCBR-E006`` measured the ceiling of view SELECTION on this family and found it almost exhausted:
a truth-seeing oracle reads the defect in 16 of 40 development worlds, the frozen V3 planner in 15,
coverage-only in 11. What that experiment did not bound is whether a chosen view is ever flown, and which
candidates the feasibility filter offers. This experiment measures both, on the same DEVELOPMENT worlds
first and on VALIDATION worlds for selection, and compares V4 execution variants against V3 and coverage.

Every arm gets the same mission, duration, ``max_plans_per_need`` budget, sensors and metric. The arms
differ only in the mission runtime configuration named in the config file (``arms``), so the comparison is
matched by construction.

Reported per world, beside the mission metrics: the fraction of accepted views actually flown inside the
declared pose and aim tolerance, the realised-versus-commanded aim error, the empty-plan rate, views before
detection, time to detection, whether the defect was read, redundant observations, travel, mission time,
energy and planning latency.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.active_mcbr_reeval import (
    MISSION_CONFIG,
    MISSION_METRICS,
    paired,
    score_mission_run,
    summarize,
)
from conrad.orchestration.view_execution import ABANDON_REASONS, ALL_REASONS

EXPERIMENT = "ACTIVE-MCBR-E007"
SCENARIO = "I4-OCCLUDED"
FAMILY_ID = "ACTIVE_INSPECTION_OCCLUDED_V1"
PRIMARY = "hidden_state_error_improvement"
BASELINE = "A-B2_coverage"

EXECUTION_METRICS = {
    "fraction_views_flown": "higher",
    "fraction_views_reached": "higher",
    "mean_closest_approach_m": "lower",
    "mean_view_direction_error_rad": "lower",
    "mean_boresight_error_rad": "lower",
    "empty_plan_rate": "lower",
    "views_before_detection": "lower",
    "time_to_detection_s": "lower",
    "defect_read": "higher",
    "accepted_views": "lower",
    "planning_latency_ms": "lower",
}
REPORT_METRICS = (
    "hidden_state_error_improvement",
    "defect_read",
    "fraction_views_flown",
    "fraction_views_reached",
    "mean_view_direction_error_rad",
    "mean_boresight_error_rad",
    "empty_plan_rate",
    "accepted_views",
    "views_before_detection",
    "time_to_detection_s",
    "redundant_observations",
    "travel_m",
    "time_s",
    "energy_j",
    "planning_latency_ms",
    "target_observed",
    "patch_max_visible_fraction",
)


def _settings(seed: int, runtime: dict[str, Any], arm_runtime: dict[str, Any]) -> Any:
    """Mission settings for one arm: the shared runtime block, then the arm's own overrides."""
    from conrad.settings import load_settings

    s = load_settings(MISSION_CONFIG)
    rt = {**dict(s.sim.get("mission", {}).get("runtime", {})), **runtime, **arm_runtime}
    return s.model_copy(
        update={
            "run": s.run.model_copy(update={"seed": int(seed)}),
            "sim": {**dict(s.sim), "mission": {"world": {}, "runtime": rt}},
        }
    )


def execution_row(run_dir: Path, base: dict[str, Any]) -> dict[str, Any]:
    """Deployment-side execution metrics of one run bundle, merged into a mission-metric row."""
    rtm = json.loads((run_dir / "mission" / "runtime_metrics.json").read_text(encoding="utf-8"))
    views = json.loads((run_dir / "mission" / "view_execution.json").read_text(encoding="utf-8"))
    summary = rtm.get("view_execution", {})
    plans = rtm.get("plans", [])
    planned = [p for p in plans if p.get("status") == "PLAN"]
    counts = dict.fromkeys(ALL_REASONS, 0)
    for v in views:
        if v.get("outcome") in counts:
            counts[v["outcome"]] += 1
    flown = [v for v in views if v.get("outcome") == "FLOWN"]
    read_t = base.get("first_direct_revision_s")
    idx = None
    if read_t is not None:
        starts = [v["t_start_s"] for v in views]
        idx = float(sum(1 for t in starts if t <= read_t) - 1) if starts else None
    row = dict(base)
    row.update(
        {
            "accepted_views": float(summary.get("accepted_views") or 0),
            "views_flown": float(summary.get("flown") or 0),
            "fraction_views_flown": float(summary.get("fraction_flown") or 0.0),
            "fraction_views_reached": float(summary.get("fraction_reached_tolerance") or 0.0),
            "mean_closest_approach_m": _f(summary.get("mean_closest_approach_m")),
            "mean_view_direction_error_rad": _f(summary.get("mean_view_direction_error_rad")),
            "mean_boresight_error_rad": _f(summary.get("mean_boresight_error_rad")),
            "plans_total": float(len(plans)),
            "plans_planned": float(len(planned)),
            # An empty plan is a plan that produced no action for a SUPPLY reason. NEED_SATISFIED is not
            # an empty plan: it is the planner correctly declining to spend a view it does not need.
            "empty_plan_rate": float(
                sum(1 for p in plans if p.get("status") in ("NO_FEASIBLE_OBSERVATION", "NOT_WORTH_COST"))
                / len(plans)
            )
            if plans
            else 0.0,
            "plans_need_satisfied": float(sum(1 for p in plans if p.get("status") == "NEED_SATISFIED")),
            "plans_not_worth_cost": float(sum(1 for p in plans if p.get("status") == "NOT_WORTH_COST")),
            "mean_feasible_candidates": float(np.mean([p.get("feasible", 0) for p in plans]))
            if plans
            else 0.0,
            "plans_with_no_feasible_candidate": float(
                sum(1 for p in plans if p.get("status") == "NO_FEASIBLE_OBSERVATION")
            ),
            "defect_read": float(base[PRIMARY] > 1e-9),
            "views_before_detection": float(idx) if idx is not None else float(len(views)),
            "time_to_detection_s": float(read_t) if read_t is not None else float(base["time_s"]),
            "dwell_time_in_tolerance_s": float(np.mean([v["time_within_tolerance_s"] for v in views]))
            if views
            else 0.0,
            "flown_views_with_revision": float(len(flown)),
            "goal_rejections": float(len(rtm.get("goal_rejections", []))),
            "planning_latency_ms": float(np.mean([p.get("latency_ms", 0.0) for p in plans]))
            if plans
            else 0.0,
            "mission_wall_clock_s": float(base.get("mission_wall_clock_s", float("nan"))),
            "outcome_counts": counts,
        }
    )
    return row


def _f(v: Any) -> float:
    return float("nan") if v is None else float(v)


def _job(args: tuple[int, str, dict[str, Any], dict[str, Any], float, str | None]) -> dict[str, Any]:
    seed, arm, runtime, arm_runtime, duration, keep = args
    from conrad.sim.mission.run import run_scenario

    root = Path(tempfile.mkdtemp(prefix="i4-exec-"))
    try:
        t0 = time.perf_counter()
        out = run_scenario(
            SCENARIO, _settings(seed, runtime, arm_runtime), run_id=f"{arm}-{seed}", runs_root=root
        )
        wall_s = time.perf_counter() - t0
        d = Path(out["run_dir"])
        base = score_mission_run(d, seed, FAMILY_ID, arm, duration)
        base["first_direct_revision_s"] = _first_direct_revision_s(d)
        base["mission_wall_clock_s"] = wall_s
        row = execution_row(d, base)
        row["planner"] = arm
        if keep:
            dest = Path(keep) / d.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(d, dest)
        return {"seed": seed, "row": row}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _first_direct_revision_s(d: Path) -> float | None:
    from uuid import UUID

    from conrad.orchestration.evaluation import evaluate_run_dir
    from conrad.persistence.db import make_engine
    from conrad.persistence.repository import Repository
    from conrad.schemas.world import Domain

    rep = evaluate_run_dir(d)
    target = UUID(rep["target_registry_id"])
    engine = make_engine(d / "conrad.sqlite")
    try:
        ts = sorted(
            r.measurement_time_ns / 1e9
            for r in Repository(engine).all_revisions()
            if r.cell.domain is Domain.TECHNICAL
            and r.cell.registry_entity_id == target
            and r.update_kind.value == "DIRECT"
        )
    finally:
        engine.dispose()
    return float(ts[0]) if ts else None


def run(config: dict[str, Any], seeds: list[int], out_dir: str | Path) -> dict[str, Any]:
    """Runs on the partition named by the config; a seed outside that partition is refused."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime = dict(config.get("runtime", {}))
    duration = float(runtime.get("duration_s", 100.0))
    workers = int(config.get("workers", 10))
    arms: dict[str, dict[str, Any]] = dict(config["arms"])
    partition = P.Partition(str(config.get("partition", "development")))
    purpose = P.Purpose(str(config.get("purpose", "design")))
    keep = config.get("keep_runs")

    with P.purpose_scope(purpose):
        split = P.split(P.I4_MCBR_V4_DOMAIN, partition, purpose)
        allowed = set(split.world_seeds)
        stray = sorted({int(s) for s in seeds} - allowed)
        if stray:
            raise P.PartitionAccessError(f"seeds outside the {partition.value} split: {stray[:10]}")
        use = [int(s) for s in seeds] or list(split.world_seeds)[: config.get("max_worlds") or None]
        per_world: dict[str, dict[str, Any]] = {}
        jobs = [(s, a, runtime, arms[a], duration, keep) for s in use for a in arms]
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in cf.as_completed([ex.submit(_job, j) for j in jobs]):
                r = fut.result()
                per_world.setdefault(str(r["seed"]), {})[r["row"]["planner"]] = r["row"]

    names = list(arms)
    pw = {str(s): per_world[str(s)] for s in use if len(per_world.get(str(s), {})) == len(names)}
    metrics = {**MISSION_METRICS, **EXECUTION_METRICS}
    baseline = str(config.get("baseline", BASELINE))
    result: dict[str, Any] = {
        "experiment_id": str(config.get("experiment_id", EXPERIMENT)),
        "data_status": "SYNTHETIC_ONLY",
        "evidence_kind": "SURROGATE",
        "evidence_note": "python sim kernel; it promotes no gate and changes no recorded I4 result",
        "world_family": FAMILY_ID,
        "scenario": SCENARIO,
        "mission_config": MISSION_CONFIG,
        "partition_file": "configs/eval/partitions_i4_mcbr_v4.yaml",
        "partition": split.partition.value,
        "partition_digest": split.digest,
        "purpose": purpose.value,
        "world_seeds": use,
        "n_worlds": len(pw),
        "arms": names,
        "arm_runtime": arms,
        "baseline": baseline,
        "primary_metric": PRIMARY,
        "question": config.get("question"),
        "decision_rule": config.get("decision_rule"),
        "config": config,
        "summary": summarize(pw, [*metrics, *REPORT_METRICS]),
        "execution_breakdown": breakdown(pw, names),
        "read_table": read_table(pw, names),
        "per_world": pw,
    }
    name = str(config.get("out_name", "i4_view_execution.json"))
    path = out / name
    # The runs are the expensive part: persist them before any statistics can fail on them.
    path.write_text(json.dumps(result, indent=1, sort_keys=True, default=str), encoding="utf-8")
    result["paired_vs_baseline"] = {a: _paired(pw, a, baseline, metrics) for a in names if a != baseline}
    reference = [str(a) for a in config.get("reference_arms", [])]
    result["paired_vs_reference"] = {
        ref: {a: _paired(pw, a, ref, metrics) for a in names if a != ref} for ref in reference if ref in names
    }
    result["metrics_not_paired"] = sorted(set(metrics) - _finite_metrics(pw, names, metrics))
    path.write_text(json.dumps(result, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return {k: v for k, v in result.items() if k not in ("per_world", "config")}


def _finite_metrics(pw: dict[str, dict[str, Any]], arms: list[str], metrics: dict[str, str]) -> set[str]:
    """Metrics that are finite in every world of every arm; a metric with a NaN is reported, not imputed."""
    ok = set()
    for m in metrics:
        vals = [w[a].get(m) for w in pw.values() for a in arms if a in w]
        if vals and all(isinstance(v, int | float) and math.isfinite(float(v)) for v in vals):
            ok.add(m)
    return ok


def _paired(
    pw: dict[str, dict[str, Any]], candidate: str, base: str, metrics: dict[str, str]
) -> dict[str, Any]:
    usable = _finite_metrics(pw, [candidate, base], metrics)
    return paired(pw, candidate, [base], {m: d for m, d in metrics.items() if m in usable})[base]


def breakdown(pw: dict[str, dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    """Why accepted views were not flown, per arm, summed over worlds."""
    out: dict[str, Any] = {}
    for a in arms:
        rows = [w[a] for w in pw.values() if a in w]
        counts = dict.fromkeys(ALL_REASONS, 0)
        for r in rows:
            for k, v in r.get("outcome_counts", {}).items():
                counts[k] = counts.get(k, 0) + int(v)
        total = sum(counts.values())
        out[a] = {
            "accepted_views": total,
            "counts": counts,
            "fractions": {k: (v / total if total else None) for k, v in counts.items()},
            "not_flown": total - counts.get("FLOWN", 0),
            "fraction_flown": counts.get("FLOWN", 0) / total if total else None,
            "abandon_reasons": {k: counts.get(k, 0) for k in ABANDON_REASONS},
            "plans_total": sum(r["plans_total"] for r in rows),
            "plans_with_an_action": sum(r["plans_planned"] for r in rows),
            "plans_no_feasible_candidate": sum(r["plans_with_no_feasible_candidate"] for r in rows),
            "plans_need_satisfied": sum(r["plans_need_satisfied"] for r in rows),
            "plans_not_worth_cost": sum(r["plans_not_worth_cost"] for r in rows),
            "mean_feasible_candidates": float(np.mean([r["mean_feasible_candidates"] for r in rows]))
            if rows
            else None,
            "goal_rejections": sum(r["goal_rejections"] for r in rows),
        }
    return out


def read_table(pw: dict[str, dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    """The all-or-nothing decomposition the family actually has: how many worlds read the defect."""
    out: dict[str, Any] = {}
    for a in arms:
        rows = [w[a] for w in pw.values() if a in w]
        read = [r for r in rows if r[PRIMARY] > 1e-9]
        out[a] = {
            "n": len(rows),
            "worlds_read": len(read),
            "fraction_read": len(read) / len(rows) if rows else None,
            "mean_when_read": float(np.mean([r[PRIMARY] for r in read])) if read else None,
            "mean": float(np.mean([r[PRIMARY] for r in rows])) if rows else None,
            "views_flown": sum(r["views_flown"] for r in rows),
            "accepted_views": sum(r["accepted_views"] for r in rows),
            "mean_time_to_detection_s": float(
                np.mean([r["time_to_detection_s"] for r in read] or [float("nan")])
            )
            if read
            else None,
            "mean_views_before_detection": float(
                np.mean([r["views_before_detection"] for r in read] or [float("nan")])
            )
            if read
            else None,
        }
    return out


def table_markdown(result: dict[str, Any], metrics: tuple[str, ...] = REPORT_METRICS) -> str:
    """The development / validation table of one run, for the write-up."""
    arms = list(result["arms"])
    lines = ["| arm | " + " | ".join(metrics) + " |", "|" + "---|" * (len(metrics) + 1)]
    for a in arms:
        s = result["summary"].get(a, {})
        cells = []
        for m in metrics:
            v = s.get(m, {}).get("mean")
            cells.append("n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4g}")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


__all__ = [
    "EXECUTION_METRICS",
    "EXPERIMENT",
    "PRIMARY",
    "REPORT_METRICS",
    "breakdown",
    "execution_row",
    "read_table",
    "run",
    "table_markdown",
]
