"""I4 failure triage from existing formal and diagnostic evidence.

This module does not rerun Unity or promote any gate. It extracts the current
formal I4 failure axis and summarizes whether existing development diagnostics
show a plausible non-final-test repair path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conrad.settings import REPO_ROOT


DEFAULT_FORMAL = Path("artifacts/gates/I4/evidence_formal.json")
DEFAULT_ORACLE = Path("artifacts/experiments/ACTIVE-MCBR-E006/i4_oracle_headroom.json")
DEFAULT_WEIGHTED_ORACLE = Path("artifacts/experiments/ACTIVE-MCBR-E006-W/i4_oracle_headroom_weighted.json")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a JSON object")
    return data


def _criterion_statuses(formal: dict[str, Any]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in formal.get("criteria", []):
        if isinstance(row, dict):
            out[str(row.get("criterion"))] = {
                "status": str(row.get("status")),
                "measured": str(row.get("measured", "")),
            }
    return out


def _headroom(path: str | Path, arms: tuple[str, ...]) -> dict[str, Any]:
    data = _load(path)
    h = data.get("headroom") if isinstance(data.get("headroom"), dict) else {}
    return {
        "path": str(_resolve(path)),
        "experiment_id": data.get("experiment_id"),
        "partition": data.get("partition"),
        "n_worlds": data.get("n_worlds"),
        "arms": {arm: h.get(arm) for arm in arms if arm in h},
    }


def build_i4_triage(
    *,
    formal: str | Path = DEFAULT_FORMAL,
    oracle: str | Path = DEFAULT_ORACLE,
    weighted_oracle: str | Path = DEFAULT_WEIGHTED_ORACLE,
) -> dict[str, Any]:
    formal_data = _load(formal)
    criteria = _criterion_statuses(formal_data)
    failing = {name: row for name, row in criteria.items() if row["status"] == "FAIL"}
    blockers = [
        {
            "blocker_id": "I4-FORMAL-FAIL-01",
            "scope": "INTERNAL",
            "criterion": name,
            "detail": row["measured"],
        }
        for name, row in failing.items()
    ]
    plain = _headroom(oracle, ("A-B2_coverage", "PRODUCTION", "ORACLE_1STEP", "ORACLE_2STEP"))
    weighted = _headroom(
        weighted_oracle,
        ("A-B2_coverage", "ORACLE_1STEP_WEIGHTED", "ORACLE_1STEP_WEIGHTED_RATIO"),
    )
    weighted_arms = weighted["arms"]
    coverage = weighted_arms.get("A-B2_coverage") or {}
    best_weighted_name = None
    best_weighted_extra = None
    for arm, vals in weighted_arms.items():
        if arm == "A-B2_coverage" or not isinstance(vals, dict):
            continue
        extra = vals.get("extra_worlds_read_vs_coverage")
        if best_weighted_extra is None or (extra is not None and extra > best_weighted_extra):
            best_weighted_name = arm
            best_weighted_extra = extra
    recommendation = {
        "next_internal_action": "design_i4_v5_weighted_ratio_development_repair",
        "reason": (
            "formal I4 fails only the information/time/energy criterion; existing development diagnostics "
            "show weighted-ratio oracle headroom over coverage, so the next repair should target value/cost "
            "and execution efficiency on development/validation partitions before any new formal final-test run"
        ),
        "do_not_do": [
            "do not relabel existing formal I4 evidence as PASS",
            "do not reread or tune on formal I4 final-test per-world outcomes",
            "do not promote oracle evidence; it is truth-side diagnostic only",
        ],
        "best_existing_weighted_oracle": {
            "arm": best_weighted_name,
            "extra_worlds_read_vs_coverage": best_weighted_extra,
            "coverage_worlds_read": coverage.get("worlds_with_the_defect_read"),
        },
    }
    return {
        "schema_version": "1.0.0",
        "gate_id": "I4-TRIAGE",
        "status": "DESIGNED",
        "decision": "REPAIR_REQUIRED",
        "formal_evidence": str(_resolve(formal)),
        "formal_git_commit": formal_data.get("git_commit"),
        "criteria": {name: {"status": row["status"]} for name, row in criteria.items()},
        "blockers": blockers,
        "diagnostics": {
            "oracle_headroom": plain,
            "weighted_oracle_headroom": weighted,
        },
        "recommendation": recommendation,
    }


def write_i4_triage(report: dict[str, Any], output: str | Path) -> Path:
    path = _resolve(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
