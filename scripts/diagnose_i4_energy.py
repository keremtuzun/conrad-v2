"""Development-only I4 energy/cost trace. Never supplies truth or future costs to runtime."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.active_mcbr_reeval import score_mission_run
from conrad.evaluation.decision_experiments.i4_view_execution import _first_direct_revision_s, _settings
from conrad.sim.mission.run import prepare


def diagnose(seed: int, out_dir: Path, arm: str = "V4_full") -> dict[str, Any]:
    historical = 8002000 <= seed < 8002040
    with P.purpose_scope(P.Purpose.DESIGN):
        fresh = seed in P.split(P.I4_ENERGY_V1_DOMAIN, P.Partition.DEVELOPMENT, P.Purpose.DESIGN).world_seeds
    if not historical and not fresh:
        raise ValueError("diagnostic accepts only I4 development seeds")
    arms: dict[str, dict[str, Any]] = {
        "V4_full": {"planner": "V4", "view_execution": {"enabled": True}},
        "V4_short_leg": {
            "planner": "V4",
            "trajectory_short_leg_fix": True,
            "view_execution": {"enabled": True},
        },
        "V4_route_cost": {
            "planner": "V4",
            "view_execution": {"enabled": True, "route_aware_navigation_cost": True},
        },
    }
    if arm not in arms:
        raise ValueError(f"unknown diagnostic arm: {arm}")
    runtime = {
        "duration_s": 100.0,
        "control_period_s": 0.1,
        "max_plans_per_need": 4,
        "model2e_enabled": False,
    }
    settings = _settings(seed, runtime, arms[arm])
    out_dir.mkdir(parents=True, exist_ok=True)
    session = prepare("I4-OCCLUDED", settings, run_id=f"I4-ENERGY-DIAG-s{seed}-{arm}", runs_root=out_dir)
    intervals: list[dict[str, Any]] = []
    hw = session.world.hardware
    before = hw.get_power_state()
    if before is None or before.energy_used_j is None:
        raise RuntimeError("power/energy meter unavailable")
    prev_e = float(before.energy_used_j)
    prev_p = np.asarray(session.runtime.estimated_pose().position_m, dtype=float)
    truth = hw.truth_access()
    prev_true_p = np.asarray(truth.true_state().position_world_m, dtype=float)
    plan_costs: dict[str, dict[str, Any]] = {}
    for _ in range(round(session.rcfg.duration_s / session.rcfg.control_period_s)):
        session.step()
        power = hw.get_power_state()
        if power is None or power.energy_used_j is None:
            raise RuntimeError("power/energy meter disappeared")
        t = hw.now_ns() / 1e9
        energy = float(power.energy_used_j)
        delta = energy - prev_e
        if delta < -1e-8 or not math.isfinite(delta):
            raise RuntimeError("non-monotone energy meter")
        prev_e = energy
        total_w = (
            None
            if power.current_a is None or power.voltage_v is None
            else float(power.current_a * power.voltage_v)
        )
        propulsion_w = (
            None
            if power.voltage_v is None
            else sum(
                float(thr.current_a) * float(power.voltage_v)
                for thr in hw.get_thruster_state()
                if thr.current_a is not None
            )
        )
        propulsion_j = (
            None
            if total_w is None or propulsion_w is None or total_w <= 0
            else delta * min(1.0, max(0.0, propulsion_w / total_w))
        )
        pos = np.asarray(session.runtime.estimated_pose().position_m, dtype=float)
        true_p = np.asarray(truth.true_state().position_world_m, dtype=float)
        active = session.runtime.executive.active
        view = session.runtime.executive.views.open_record
        phase = (
            "inspection_approach"
            if view is not None
            else (
                "inspection_other"
                if active is not None and active.purpose == "INSPECT"
                else ("idle" if active is None else active.purpose.lower())
            )
        )
        if view is not None and view.t_within_position_tolerance_s is not None:
            phase = "inspection_dwell"
        intervals.append(
            {
                "t_s": t,
                "phase": phase,
                "plan_id": None if view is None else view.plan_id,
                "energy_j": delta,
                "distance_m": float(np.linalg.norm(pos - prev_p)),
                "true_distance_m": float(np.linalg.norm(true_p - prev_true_p)),
                "power_w": total_w,
                "propulsion_energy_j_est": propulsion_j,
                "base_energy_j_est": None if propulsion_j is None else delta - propulsion_j,
            }
        )
        prev_p = pos
        prev_true_p = true_p
        for adopted in session.runtime.deliberation.plans:
            plan = adopted.plan
            if str(plan.plan_id) in plan_costs or plan.primary_action is None:
                continue
            cost = plan.primary_action.expected_cost
            dwell_s = plan.primary_action.duration_s
            sensor_w = session.runtime.deliberation.sensor_option.power_w
            plan_costs[str(plan.plan_id)] = {
                "predicted_time_s": cost.time_s,
                "predicted_energy_j": cost.energy_j,
                "predicted_distance_m": cost.travel_m,
                "predicted_approach_time_s": cost.time_s - dwell_s,
                "predicted_navigation_energy_j": cost.energy_j - sensor_w * dwell_s,
                "predicted_sensor_energy_j": sensor_w * dwell_s,
                "dwell_s": dwell_s,
            }
    result = session.finish()
    first_direct_revision_s = _first_direct_revision_s(session.run_dir)
    mission_metrics = score_mission_run(
        session.run_dir, seed, "ACTIVE_INSPECTION_OCCLUDED_V1", arm, session.rcfg.duration_s
    )
    views = json.loads((session.run_dir / "mission" / "view_execution.json").read_text(encoding="utf-8"))
    by_phase: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "energy_j": 0.0,
            "propulsion_energy_j_est": 0.0,
            "base_energy_j_est": 0.0,
            "distance_m": 0.0,
            "true_distance_m": 0.0,
            "time_s": 0.0,
        }
    )
    dt = session.rcfg.control_period_s
    for row in intervals:
        bucket = by_phase[row["phase"]]
        bucket["energy_j"] += row["energy_j"]
        if row["propulsion_energy_j_est"] is not None:
            bucket["propulsion_energy_j_est"] += row["propulsion_energy_j_est"]
            bucket["base_energy_j_est"] += row["base_energy_j_est"]
        bucket["distance_m"] += row["distance_m"]
        bucket["true_distance_m"] += row["true_distance_m"]
        bucket["time_s"] += dt
    view_rows: list[dict[str, Any]] = []
    for view in views:
        start, end = view["t_start_s"], view["t_end_s"]
        matching = [r for r in intervals if r["plan_id"] == view["plan_id"] and start <= r["t_s"] <= end]
        approach = [r for r in matching if r["phase"] == "inspection_approach"]
        pred = plan_costs.get(view["plan_id"], {})
        view_rows.append(
            {
                "plan_id": view["plan_id"],
                "outcome": view["outcome"],
                "start_s": start,
                "end_s": end,
                "reached": view["reached_tolerance"],
                "flown": view["outcome"] == "FLOWN",
                "predicted": pred,
                "realized_elapsed_s": end - start,
                "realized_approach_time_s": None
                if view["t_within_position_tolerance_s"] is None
                else view["t_within_position_tolerance_s"] - start,
                "realized_distance_m": sum(r["distance_m"] for r in matching),
                "realized_true_distance_m": sum(r["true_distance_m"] for r in matching),
                "realized_energy_j": sum(r["energy_j"] for r in matching),
                "realized_approach_distance_m": sum(r["distance_m"] for r in approach),
                "realized_approach_true_distance_m": sum(r["true_distance_m"] for r in approach),
                "realized_approach_energy_j": sum(r["energy_j"] for r in approach),
                "arrival_s": view["t_within_position_tolerance_s"],
                "closest_approach_m": view["closest_approach_m"],
                "target_revision_during_view": first_direct_revision_s is not None
                and start <= first_direct_revision_s <= end,
            }
        )
    body = {
        "evidence_kind": "DEVELOPMENT_DIAGNOSTIC",
        "execution_path": "Python kernel surrogate",
        "partition": "historical_v4_development" if historical else "i4_energy_v1_development",
        "seed": seed,
        "arm": arm,
        "run_dir": str(session.run_dir),
        "mission_energy_j": result["report"]["vehicle"]["energy_used_j"]
        if "vehicle" in result["report"]
        else prev_e,
        "meter_end_j": prev_e,
        "mission_metrics": mission_metrics,
        "power_split_note": "Estimated from same-tick reported thruster and total currents; phase total uses exact meter delta",
        "first_direct_revision_s": first_direct_revision_s,
        "energy_after_first_direct_revision_j": None
        if first_direct_revision_s is None
        else sum(r["energy_j"] for r in intervals if r["t_s"] > first_direct_revision_s),
        "phases": dict(by_phase),
        "views": view_rows,
        "intervals": intervals,
    }
    (session.run_dir / "reports" / "i4_energy_diagnostic.json").write_text(
        json.dumps(body, indent=1, sort_keys=True), encoding="utf-8"
    )
    return {k: v for k, v in body.items() if k != "intervals"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, nargs="?")
    parser.add_argument("--all-development", action="store_true")
    parser.add_argument("--all-new-development", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arm", default="V4_full")
    parser.add_argument("--out", type=Path, default=Path("artifacts/experiments/I4-ENERGY-DIAG"))
    args = parser.parse_args()
    if args.all_development or args.all_new_development:
        if args.seed is not None:
            parser.error("provide either a seed or --all-development")
        if not 1 <= args.workers <= 4:
            parser.error("--workers must be 1..4")
        seeds = range(8100000, 8100040) if args.all_new_development else range(8002000, 8002040)
        jobs = [
            s
            for s in seeds
            if not (
                args.out / f"I4-ENERGY-DIAG-s{s}-{args.arm}" / "reports" / "i4_energy_diagnostic.json"
            ).exists()
        ]
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(diagnose, s, args.out, args.arm): s for s in jobs}
            for future in cf.as_completed(futures):
                row = future.result()
                print(
                    f"completed {row['seed']}: {len(row['views'])} views, {row['meter_end_j']:.1f} J",
                    flush=True,
                )
    elif args.seed is not None:
        print(json.dumps(diagnose(args.seed, args.out, args.arm), indent=1, default=dict))
    else:
        parser.error("seed or --all-development required")
