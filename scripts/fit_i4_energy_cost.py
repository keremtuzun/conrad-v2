"""Fit a two-offset I4 cost correction on the first 20 new development worlds only."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from scripts.summarize_i4_energy import errors

ROOT = Path("artifacts/experiments/I4-ENERGY-DIAG")
ARM = "V4_short_leg"
CALIBRATION = range(8100000, 8100020)
DEVELOPMENT_CHECK = range(8100020, 8100040)


def views(seeds: range) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        path = ROOT / f"I4-ENERGY-DIAG-s{seed}-{ARM}" / "reports" / "i4_energy_diagnostic.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["seed"] != seed or doc["arm"] != ARM or doc["partition"] != "i4_energy_v1_development":
            raise ValueError(f"unexpected provenance in {path}")
        rows.extend(view for view in doc["views"] if view["outcome"] == "FLOWN" and view["predicted"])
    return rows


def cost_errors(rows: list[dict[str, Any]], time_offset_s: float, energy_offset_j: float) -> dict[str, Any]:
    return {
        "time_s": errors(
            [
                (view["predicted"]["predicted_time_s"] + time_offset_s, view["realized_elapsed_s"])
                for view in rows
            ]
        ),
        "energy_j": errors(
            [
                (view["predicted"]["predicted_energy_j"] + energy_offset_j, view["realized_energy_j"])
                for view in rows
            ]
        ),
    }


if __name__ == "__main__":
    fit = views(CALIBRATION)
    check = views(DEVELOPMENT_CHECK)
    if len(fit) < 10 or len(check) < 10:
        raise ValueError("too few completed views for transparent offset calibration")
    # Predeclared low-complexity rule: fit the median residual, rounded upward to one
    # tenth of a second / hundred joules. The path-length slopes remain V4's 1/cruise
    # and 40 J/m, ensuring a monotone, deployment-side predictor.
    time_resid = statistics.median(
        view["realized_elapsed_s"] - view["predicted"]["predicted_time_s"] for view in fit
    )
    energy_resid = statistics.median(
        view["realized_energy_j"] - view["predicted"]["predicted_energy_j"] for view in fit
    )
    import math

    time_offset_s = max(0.0, math.ceil(time_resid * 10) / 10)
    energy_offset_j = max(0.0, math.ceil(energy_resid / 100) * 100.0)
    result = {
        "evidence_kind": "DEVELOPMENT_CALIBRATION",
        "execution_path": "Python kernel surrogate",
        "source_arm": ARM,
        "fit_seeds": list(CALIBRATION),
        "check_seeds": list(DEVELOPMENT_CHECK),
        "rule": "median completed-view residual, rounded upward to 0.1 s and 100 J",
        "time_offset_s": time_offset_s,
        "energy_offset_j": energy_offset_j,
        "fit_before": cost_errors(fit, 0.0, 0.0),
        "fit_after": cost_errors(fit, time_offset_s, energy_offset_j),
        "development_check_before": cost_errors(check, 0.0, 0.0),
        "development_check_after": cost_errors(check, time_offset_s, energy_offset_j),
    }
    out = ROOT / "development_cost_calibration.json"
    out.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=1))
