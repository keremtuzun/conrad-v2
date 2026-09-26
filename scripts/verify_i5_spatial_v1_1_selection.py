"""Verify an I5 Spatial V1.1 validation result against its frozen YAML selection rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    rule = config["selection_rule"]
    rows = result["per_run"]
    expected = set(product(config["seeds"], config["scenarios"], config["arms"]))
    observed = [(row["seed"], row["scenario"], row["arm"]) for row in rows]
    verdicts = result["verdicts"]
    primary = "egdc_structured"
    nominal = "I5-NOMINAL"
    non_nominal = [scenario for scenario in config["scenarios"] if scenario != nominal]
    primary_rows = [row for row in rows if row["arm"] == primary]
    false_intact = sum(
        bool(row["spatial_v1"]["false_intact_on_covered_resolvable_defect"])
        for row in primary_rows
        if row.get("spatial_v1") is not None
    )
    checks = {
        "config_exact": result["config"] == config,
        "partition_exact": result["partition"] == "validation"
        and result["partition_domain"] == config["partition_domain"],
        "complete_unique_grid": len(observed) == len(expected)
        and len(set(observed)) == len(observed)
        and set(observed) == expected,
        "primary_nominal_warrants": verdicts["per_scenario_warrant_reached"][nominal]
        >= rule["primary_nominal_warrants_min"],
        "non_nominal_warrants": all(
            verdicts["per_scenario_warrant_reached"][scenario]
            >= rule["non_nominal_warrants_per_scenario_min"]
            for scenario in non_nominal
        ),
        "non_nominal_correct_given_warrant": all(
            (verdicts["per_scenario_correct_given_warrant_rate"][scenario] or 0.0)
            >= rule["non_nominal_correct_given_warrant_rate_min"]
            for scenario in non_nominal
        ),
        "false_intact": false_intact <= rule["false_intact_on_covered_resolvable_defect_max"],
        "decision_violations": verdicts["violations_total"] <= rule["decision_violations_max"],
        "nominal_over_escalations": verdicts["nominal_over_escalations"]
        <= rule["nominal_over_escalations_max"],
        "traceability": verdicts["traceable_fraction"] >= rule["traceability_min"],
        "uir": verdicts["uir"] <= rule["uir_max"],
        "competitiveness": verdicts["competitive_outcomes"],
        "spatial_runtime": verdicts["spatial_v1_runtime_selected"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "measured": {
            "rows": len(rows),
            "unique_rows": len(set(observed)),
            "primary_nominal_warrants": verdicts["per_scenario_warrant_reached"][nominal],
            "non_nominal_warrants": {
                scenario: verdicts["per_scenario_warrant_reached"][scenario]
                for scenario in non_nominal
            },
            "non_nominal_correct_given_warrant": {
                scenario: verdicts["per_scenario_correct_given_warrant_rate"][scenario]
                for scenario in non_nominal
            },
            "false_intact": false_intact,
            "decision_violations": verdicts["violations_total"],
            "nominal_over_escalations": verdicts["nominal_over_escalations"],
            "traceability": verdicts["traceable_fraction"],
            "uir": verdicts["uir"],
            "competitive_outcomes": verdicts["competitive"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    report = {
        "config_sha256": _sha256(args.config),
        "result_sha256": _sha256(args.result),
        **verify(config, result),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
