"""Run the unchanged I4 simple-view baselines on the new development partition."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path
from typing import Any

from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.i4_view_execution import _job

ARMS = ("A-B1_fixed_inspection", "A-B0_random", "A-B2_coverage")
RUNTIME: dict[str, Any] = {
    "duration_s": 100.0,
    "control_period_s": 0.1,
    "max_plans_per_need": 4,
    "model2e_enabled": False,
}


def run(out: Path, workers: int, max_worlds: int) -> None:
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1..4")
    if not 1 <= max_worlds <= 40:
        raise ValueError("max_worlds must be 1..40")
    with P.purpose_scope(P.Purpose.DESIGN):
        seeds = P.split(P.I4_ENERGY_V1_DOMAIN, P.Partition.DEVELOPMENT, P.Purpose.DESIGN).world_seeds[
            :max_worlds
        ]
    jobs = []
    for seed in seeds:
        for arm in ARMS:
            path = out / str(seed) / f"{arm}.json"
            if not path.exists():
                jobs.append((seed, arm, RUNTIME, {"planner": arm}, 100.0, None))
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, job): (job[0], job[1]) for job in jobs}
        for future in cf.as_completed(futures):
            result = future.result()
            seed, arm = futures[future]
            row = result["row"]
            if result["seed"] != seed or row["planner"] != arm:
                raise ValueError("baseline result provenance mismatch")
            path = out / str(seed) / f"{arm}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row, indent=1, sort_keys=True), encoding="utf-8")
            print(f"completed {seed} {arm}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/experiments/I4-ENERGY-ABLATIONS/development")
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-worlds", type=int, default=10)
    args = parser.parse_args()
    run(args.out, args.workers, args.max_worlds)
