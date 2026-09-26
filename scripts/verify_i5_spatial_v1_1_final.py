"""Verify the sealed I5 Spatial V1.1 surrogate final from raw mission rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import yaml

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
PRIMARY_ARM = "egdc_structured"
NOMINAL = "I5-NOMINAL"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _row_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return int(row["seed"]), str(row["scenario"]), str(row["arm"])


def _primary_rows(rows: list[dict[str, Any]], scenario: str | None = None) -> list[dict[str, Any]]:
    return [
        row for row in rows if row["arm"] == PRIMARY_ARM and (scenario is None or row["scenario"] == scenario)
    ]


def _correct_given_warrant(rows: list[dict[str, Any]]) -> float | None:
    warranted = [row for row in rows if bool(row["driving"]["warrant_reached"])]
    if not warranted:
        return None
    return sum(bool(row["driving"]["correct"]) for row in warranted) / len(warranted)


def _pooled(rows: list[dict[str, Any]], arm: str) -> dict[str, int]:
    arm_rows = [row for row in rows if row["arm"] == arm]
    return {
        "task_success": sum(bool(row["outcome"]["task_success"]) for row in arm_rows),
        "safety_events": sum(int(row["outcome"]["safety_events"]) for row in arm_rows),
        "violations": sum(int(row["driving"]["violations_total"]) for row in arm_rows),
    }


def checkpoint_from_manifest(result: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the sealed declaration while retaining a digest of omitted duplicate rows."""
    declaration = {
        "experiment_id": result["experiment_id"],
        "partition": result["partition"],
        "partition_domain": result["partition_domain"],
        "seeds": result["seeds"],
        "scenarios": list(result["scenarios"]),
        "arms": result["arms"],
        "config": result["config"],
        "source": manifest["source"],
    }
    return {
        "declaration_sha256": manifest["declaration_sha256"],
        "declaration": declaration,
        "completed_rows": result["per_run"],
    }


def verify(
    config: dict[str, Any],
    result: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    expected_source_commit: str,
    checkpoint_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed report derived from raw rows and sealed checkpoint data."""
    rule = config["final_rule"]
    rows = list(result["per_run"])
    checkpoint_rows = list(checkpoint["completed_rows"])
    expected = set(product(config["seeds"], config["scenarios"], config["arms"]))
    observed = [_row_key(row) for row in rows]
    primary = _primary_rows(rows)
    source = checkpoint["declaration"]["source"]
    declaration = checkpoint["declaration"]
    declaration_hash = _canonical_sha256(declaration)

    scenario_warrants = {
        scenario: sum(bool(row["driving"]["warrant_reached"]) for row in _primary_rows(rows, scenario))
        for scenario in config["scenarios"]
    }
    scenario_correct_given_warrant = {
        scenario: _correct_given_warrant(_primary_rows(rows, scenario)) for scenario in config["scenarios"]
    }
    false_intact = sum(
        bool(row["spatial_v1"]["false_intact_on_covered_resolvable_defect"])
        for row in primary
        if row.get("spatial_v1") is not None
    )
    violations = sum(int(row["driving"]["violations_total"]) for row in primary)
    nominal_over_escalations = sum(
        int(row["driving"]["over_escalations"]) for row in _primary_rows(rows, NOMINAL)
    )
    decisions = sum(int(row["driving"]["n_decisions"]) for row in primary)
    traceable = sum(int(row["driving"]["traceable_decisions"]) for row in primary)
    relied = sum(int(row["driving"]["uir"]["relied_world_claims"]) for row in primary)
    unsupported = sum(int(row["driving"]["uir"]["relied_unsupported_claims"]) for row in primary)
    traceability = traceable / decisions if decisions else 0.0
    uir = unsupported / relied if relied else 0.0
    pooled = {arm: _pooled(rows, arm) for arm in config["arms"]}
    competitive = all(
        pooled[PRIMARY_ARM]["task_success"] >= pooled[arm]["task_success"]
        and pooled[PRIMARY_ARM]["safety_events"] <= pooled[arm]["safety_events"]
        and pooled[PRIMARY_ARM]["violations"] <= pooled[arm]["violations"]
        for arm in config["arms"]
        if arm != PRIMARY_ARM
    )

    checkpoint_by_key = {_row_key(row): row for row in checkpoint_rows}
    result_by_key = {_row_key(row): row for row in rows}
    manifest_rows_exact = checkpoint_manifest is None or (
        checkpoint_manifest["completed_rows"] == len(rows)
        and checkpoint_manifest["completed_rows_canonical_sha256"] == _canonical_sha256(rows)
        and len(checkpoint_manifest["checkpoint_sha256"]) == 64
    )
    checks = {
        "config_exact": result["config"] == config,
        "identity_exact": result["experiment_id"] == config["experiment_id"],
        "surrogate_boundary": result["data_status"] == "SYNTHETIC_ONLY"
        and "SURROGATE" in result["evidence_class"]
        and "not Unity" in result["evidence_class"],
        "partition_exact": result["partition"] == "final_test"
        and result["partition_domain"] == config["partition_domain"],
        "result_axes_exact": result["seeds"] == config["seeds"]
        and result["arms"] == config["arms"]
        and set(result["scenarios"]) == set(config["scenarios"]),
        "complete_unique_grid": len(rows) == int(rule["expected_rows"])
        and len(observed) == len(expected)
        and len(set(observed)) == len(observed)
        and set(observed) == expected,
        "checkpoint_complete_unique_grid": len(checkpoint_rows) == int(rule["expected_rows"])
        and len(checkpoint_by_key) == len(checkpoint_rows)
        and set(checkpoint_by_key) == expected,
        "checkpoint_rows_exact": checkpoint_by_key == result_by_key,
        "checkpoint_manifest_rows_exact": manifest_rows_exact,
        "checkpoint_declaration_hash": declaration_hash
        == checkpoint["declaration_sha256"]
        == result["checkpoint"]["declaration_sha256"],
        "checkpoint_declaration_exact": declaration["experiment_id"] == config["experiment_id"]
        and declaration["partition"] == "final_test"
        and declaration["partition_domain"] == config["partition_domain"]
        and declaration["seeds"] == config["seeds"]
        and declaration["scenarios"] == config["scenarios"]
        and declaration["arms"] == config["arms"]
        and declaration["config"] == config,
        "source_commit": source["git_commit"] == expected_source_commit,
        "source_clean": source["source_diff_sha256"] == EMPTY_SHA256,
        "spatial_runtime": len(primary) == len(config["seeds"]) * len(config["scenarios"])
        and all(row.get("spatial_v1") is not None for row in primary),
        "primary_nominal_warrants": scenario_warrants[NOMINAL] >= int(rule["primary_nominal_warrants_min"]),
        "each_scenario_warrants": all(
            count >= int(rule["each_scenario_warrants_min"]) for count in scenario_warrants.values()
        ),
        "each_scenario_correct_given_warrant": all(
            rate is not None and rate >= float(rule["each_scenario_correct_given_warrant_rate_min"])
            for rate in scenario_correct_given_warrant.values()
        ),
        "false_intact": false_intact <= int(rule["false_intact_on_covered_resolvable_defect_max"]),
        "decision_violations": violations <= int(rule["decision_violations_max"]),
        "nominal_over_escalations": nominal_over_escalations <= int(rule["nominal_over_escalations_max"]),
        "traceability": traceability >= float(rule["traceability_min"]),
        "uir": uir <= float(rule["uir_max"]),
        "competitiveness": competitive,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "measured": {
            "rows": len(rows),
            "unique_rows": len(set(observed)),
            "source": source,
            "scenario_warrants": scenario_warrants,
            "scenario_correct_given_warrant": scenario_correct_given_warrant,
            "false_intact": false_intact,
            "decision_violations": violations,
            "nominal_over_escalations": nominal_over_escalations,
            "traceability": traceability,
            "uir": uir,
            "pooled": pooled,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path)
    checkpoint_group.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    manifest = None
    if args.checkpoint is not None:
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        checkpoint_sha256 = _sha256(args.checkpoint)
    else:
        manifest = json.loads(args.checkpoint_manifest.read_text(encoding="utf-8"))
        checkpoint = checkpoint_from_manifest(result, manifest)
        checkpoint_sha256 = manifest["checkpoint_sha256"]
    report = {
        "config_sha256": _sha256(args.config),
        "result_sha256": _sha256(args.result),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_manifest_sha256": (
            _sha256(args.checkpoint_manifest) if args.checkpoint_manifest is not None else None
        ),
        **verify(
            config,
            result,
            checkpoint,
            expected_source_commit=args.expected_source_commit,
            checkpoint_manifest=manifest,
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
