"""Paired, development-only I4 ablation screen; no formal gate decision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_i4_energy_screen import arms as variant_arms

DIAG = Path("artifacts/experiments/I4-ENERGY-DIAG")
ABL = Path("artifacts/experiments/I4-ENERGY-ABLATIONS")
ARMS = (
    "V4_full",
    "V4_short_leg",
    "V4_time",
    "V4_energy",
    "V4_horizon",
    "V4_combined",
    "A-B1_fixed_inspection",
    "A-B0_random",
    "A-B2_coverage",
)
METRICS = (
    "hidden_state_error_improvement",
    "target_observed",
    "info_per_time",
    "info_per_kj",
    "energy_j",
    "travel_m",
    "observations",
    "redundant_observations",
    "accepted_views",
    "views_flown",
    "ABANDONED_MISSION_END",
    "collisions",
)


def read_row(seed: int, arm: str) -> dict[str, Any]:
    if arm in ("V4_full", "V4_short_leg"):
        path = DIAG / f"I4-ENERGY-DIAG-s{seed}-{arm}" / "reports" / "i4_energy_diagnostic.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["seed"] != seed or doc["arm"] != arm:
            raise ValueError(f"diagnostic provenance mismatch: {path}")
        row = dict(doc["mission_metrics"])
        row["accepted_views"] = len(doc["views"])
        row["views_flown"] = sum(view["outcome"] == "FLOWN" for view in doc["views"])
        row["ABANDONED_MISSION_END"] = sum(
            view["outcome"] == "ABANDONED_MISSION_END" for view in doc["views"]
        )
        return row
    folder = "development" if arm.startswith("A-") else "development_screen"
    path = ABL / folder / str(seed) / f"{arm}.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    if row["seed"] != seed or row["planner"] != arm:
        raise ValueError(f"ablation provenance mismatch: {path}")
    row["ABANDONED_MISSION_END"] = row["outcome_counts"].get("ABANDONED_MISSION_END", 0)
    return row


def summarize() -> dict[str, Any]:
    seeds = list(range(8100000, 8100010))
    calibration = json.loads((DIAG / "development_cost_calibration.json").read_text(encoding="utf-8"))
    rows = {arm: [read_row(seed, arm) for seed in seeds] for arm in ARMS}
    means = {
        arm: {metric: float(np.mean([float(row[metric]) for row in values])) for metric in METRICS}
        for arm, values in rows.items()
    }
    paired: dict[str, Any] = {}
    for arm, values in rows.items():
        if arm == "V4_full":
            continue
        diffs = np.array([values[i]["info_per_kj"] - rows["V4_full"][i]["info_per_kj"] for i in range(10)])
        rng = np.random.default_rng(20260919)
        boot = diffs[rng.integers(0, 10, size=(4000, 10))].mean(axis=1)
        paired[arm] = {
            "info_per_kj_mean_difference": float(diffs.mean()),
            "info_per_kj_95pct_development_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "defect_read_difference": means[arm]["target_observed"] - means["V4_full"]["target_observed"],
        }
    return {
        "evidence_kind": "DEVELOPMENT_SCREEN",
        "execution_path": "Python kernel surrogate",
        "seeds": seeds,
        "arms": list(ARMS),
        "arm_runtime": {
            **variant_arms(calibration),
            **{arm: {"planner": arm} for arm in ARMS if arm.startswith("A-")},
            "V4_full": {"planner": "V4", "view_execution": {"enabled": True}},
            "V4_short_leg": {
                "planner": "V4",
                "trajectory_short_leg_fix": True,
                "view_execution": {"enabled": True},
            },
        },
        "calibration_offsets": {
            "time_offset_s": calibration["time_offset_s"],
            "energy_offset_j": calibration["energy_offset_j"],
        },
        "means": means,
        "paired_vs_v4": paired,
        "wasted_energy_j": None,
        "note": "Ten-world development screen only; no formal or validation claim",
    }


if __name__ == "__main__":
    result = summarize()
    path = ABL / "development_screen_summary.json"
    path.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=1))
