"""Summarize measured I4 development traces without opening held-out worlds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def errors(rows: list[tuple[float, float]]) -> dict[str, Any]:
    diff = np.asarray([actual - predicted for predicted, actual in rows], dtype=float)
    if not len(diff):
        return {"n": 0}
    return {
        "n": len(diff),
        "bias": float(diff.mean()),
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "q05": float(np.quantile(diff, 0.05)),
        "q50": float(np.quantile(diff, 0.5)),
        "q95": float(np.quantile(diff, 0.95)),
        "underprediction_fraction": float(np.mean(diff > 0)),
    }


def summarize(
    root: Path, require_complete: bool = False, partition: str = "historical", arm: str = "V4_full"
) -> dict[str, Any]:
    lo = 8002000 if partition == "historical" else 8100000
    hi = lo + 40
    paths = sorted(root.glob(f"I4-ENERGY-DIAG-s*-{arm}/reports/i4_energy_diagnostic.json"))
    docs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    docs = [doc for doc in docs if lo <= int(doc["seed"]) < hi]
    seeds = [int(doc["seed"]) for doc in docs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate or non-development seed in diagnostics")
    if require_complete and seeds != list(range(lo, hi)):
        raise ValueError(f"development diagnostics incomplete: {len(seeds)}/40")
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    phases: dict[str, list[float]] = defaultdict(list)
    outcomes: dict[str, dict[str, float]] = defaultdict(
        lambda: {"views": 0, "energy_j": 0.0, "distance_m": 0.0, "time_s": 0.0}
    )
    censored: list[dict[str, Any]] = []
    post_read: list[float] = []
    startup_energy: list[float] = []
    mission_metrics: dict[str, list[float]] = defaultdict(list)
    for doc in docs:
        for key, value in doc.get("mission_metrics", {}).items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                mission_metrics[key].append(float(value))
        intervals = doc["intervals"]
        total = sum(float(row["energy_j"]) for row in intervals)
        startup_energy.append(float(doc["meter_end_j"]) - total)
        if doc["energy_after_first_direct_revision_j"] is not None:
            post_read.append(float(doc["energy_after_first_direct_revision_j"]))
        for phase, row in doc["phases"].items():
            phases[phase].append(float(row["energy_j"]))
        for view in doc["views"]:
            outcome = str(view["outcome"])
            bucket = outcomes[outcome]
            bucket["views"] += 1
            bucket["energy_j"] += float(view["realized_energy_j"])
            bucket["distance_m"] += float(view["realized_distance_m"])
            bucket["time_s"] += float(view["realized_elapsed_s"])
            predicted = view["predicted"]
            if not predicted:
                continue
            if outcome != "FLOWN":
                censored.append(
                    {
                        "seed": doc["seed"],
                        "outcome": outcome,
                        "predicted_time_s": predicted["predicted_time_s"],
                        "elapsed_until_abandonment_s": view["realized_elapsed_s"],
                        "predicted_energy_j": predicted["predicted_energy_j"],
                        "energy_until_abandonment_j": view["realized_energy_j"],
                    }
                )
                continue
            for metric, pred, actual in (
                ("time_s", "predicted_time_s", "realized_elapsed_s"),
                ("estimated_distance_m", "predicted_distance_m", "realized_distance_m"),
                ("true_distance_m", "predicted_distance_m", "realized_true_distance_m"),
                ("energy_j", "predicted_energy_j", "realized_energy_j"),
                ("approach_time_s", "predicted_approach_time_s", "realized_approach_time_s"),
                ("approach_estimated_distance_m", "predicted_distance_m", "realized_approach_distance_m"),
                ("approach_true_distance_m", "predicted_distance_m", "realized_approach_true_distance_m"),
                ("approach_energy_j", "predicted_navigation_energy_j", "realized_approach_energy_j"),
            ):
                if predicted.get(pred) is not None and view.get(actual) is not None:
                    pairs[metric].append((float(predicted[pred]), float(view[actual])))
    if any(not math.isclose(energy, 2.5, abs_tol=1e-5) for energy in startup_energy):
        raise ValueError(f"startup energy does not reconcile to 0.1 s hotel load: {startup_energy}")
    return {
        "evidence_kind": "DEVELOPMENT_DIAGNOSTIC",
        "partition": partition,
        "arm": arm,
        "execution_path": "Python kernel surrogate",
        "seeds": seeds,
        "n_worlds": len(seeds),
        "meter_startup_energy_j": startup_energy,
        "completed_view_cost_error": {key: errors(value) for key, value in pairs.items()},
        "censored_views": censored,
        "phase_energy_j": {
            key: {"n": len(value), "mean": statistics.mean(value), "sum": sum(value)}
            for key, value in phases.items()
        },
        "view_outcomes": outcomes,
        "mission_metrics": {
            key: {"n": len(value), "mean": statistics.mean(value)}
            for key, value in mission_metrics.items()
        },
        "post_first_direct_revision_energy_j": {
            "n": len(post_read),
            "mean": statistics.mean(post_read) if post_read else None,
            "sum": sum(post_read),
        },
        "wasted_energy_j": None,
        "wasted_energy_note": "No-value status cannot be established from a target direct revision alone",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/experiments/I4-ENERGY-DIAG"))
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--partition", choices=("historical", "new"), default="historical")
    parser.add_argument("--arm", choices=("V4_full", "V4_short_leg"), default="V4_full")
    args = parser.parse_args()
    result = summarize(args.root, args.require_complete, args.partition, args.arm)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / f"development_summary_{args.partition}_{args.arm}.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "censored_views"}, indent=1))
