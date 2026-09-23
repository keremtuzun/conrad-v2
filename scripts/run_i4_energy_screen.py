"""Screen predeclared I4 cost/filter ablations on ten new development worlds."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path
from typing import Any

from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.i4_view_execution import _job

RUNTIME: dict[str, Any] = {
    "duration_s": 100.0,
    "control_period_s": 0.1,
    "max_plans_per_need": 4,
    "model2e_enabled": False,
}
ROOT = Path("artifacts/experiments/I4-ENERGY-DIAG")


def arms(calibration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    t = float(calibration["time_offset_s"])
    e = float(calibration["energy_offset_j"])
    base = {"planner": "V4", "view_execution": {"enabled": True}}
    return {
        "V4_time": {
            **base,
            "i4_cost_calibration": {"enabled": True, "time_offset_s": t},
        },
        "V4_energy": {
            **base,
            "i4_cost_calibration": {"enabled": True, "energy_offset_j": e},
        },
        "V4_horizon": {
            **base,
            "view_execution": {"enabled": True, "budget_from_mission_duration": True},
            "v4": {"budget_reserve_s": t},
        },
        "V4_combined": {
            **base,
            "trajectory_short_leg_fix": True,
            "i4_cost_calibration": {"enabled": True, "time_offset_s": t, "energy_offset_j": e},
            "view_execution": {"enabled": True, "budget_from_mission_duration": True},
        },
    }


def run(out: Path, workers: int) -> None:
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1..4")
    calibration = json.loads((ROOT / "development_cost_calibration.json").read_text(encoding="utf-8"))
    if calibration["fit_seeds"] != list(range(8100000, 8100020)):
        raise ValueError("calibration provenance mismatch")
    variants = arms(calibration)
    with P.purpose_scope(P.Purpose.DESIGN):
        allowed = P.split(P.I4_ENERGY_V1_DOMAIN, P.Partition.DEVELOPMENT, P.Purpose.DESIGN).world_seeds
    seeds = [seed for seed in allowed if 8100000 <= seed < 8100010]
    if seeds != list(range(8100000, 8100010)):
        raise ValueError("unexpected screen seeds")
    jobs = [
        (seed, arm, RUNTIME, runtime, 100.0, None)
        for seed in seeds
        for arm, runtime in variants.items()
        if not (out / str(seed) / f"{arm}.json").exists()
    ]
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, job): (job[0], job[1]) for job in jobs}
        for future in cf.as_completed(futures):
            result = future.result()
            seed, arm = futures[future]
            row = result["row"]
            if result["seed"] != seed or row["planner"] != arm:
                raise ValueError("ablation result provenance mismatch")
            path = out / str(seed) / f"{arm}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row, indent=1, sort_keys=True), encoding="utf-8")
            print(f"completed {seed} {arm}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/experiments/I4-ENERGY-ABLATIONS/development_screen")
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.out, args.workers)
