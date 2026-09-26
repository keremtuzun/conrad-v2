from __future__ import annotations

import json
from pathlib import Path

from conrad.evaluation.i4_triage import build_i4_triage, write_i4_triage


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_i4_triage_preserves_formal_failure_and_names_weighted_repair(tmp_path: Path) -> None:
    formal = _write(
        tmp_path / "formal.json",
        {
            "git_commit": "abc",
            "criteria": [
                {"criterion": "closed loop", "status": "PASS", "measured": "ok"},
                {"criterion": "beats simple views on information/time/energy", "status": "FAIL", "measured": "ci < 0"},
            ],
        },
    )
    oracle = _write(
        tmp_path / "oracle.json",
        {
            "experiment_id": "E006",
            "partition": "development",
            "n_worlds": 2,
            "headroom": {
                "A-B2_coverage": {"worlds_with_the_defect_read": 1},
                "PRODUCTION": {"worlds_with_the_defect_read": 1},
            },
        },
    )
    weighted = _write(
        tmp_path / "weighted.json",
        {
            "experiment_id": "E006-W",
            "partition": "development",
            "n_worlds": 2,
            "headroom": {
                "A-B2_coverage": {"worlds_with_the_defect_read": 1},
                "ORACLE_1STEP_WEIGHTED_RATIO": {
                    "worlds_with_the_defect_read": 2,
                    "extra_worlds_read_vs_coverage": 1,
                },
            },
        },
    )
    report = build_i4_triage(formal=formal, oracle=oracle, weighted_oracle=weighted)
    assert report["decision"] == "REPAIR_REQUIRED"
    assert report["blockers"][0]["criterion"] == "beats simple views on information/time/energy"
    assert report["recommendation"]["next_internal_action"] == "design_i4_v5_weighted_ratio_development_repair"
    assert report["recommendation"]["best_existing_weighted_oracle"]["arm"] == "ORACLE_1STEP_WEIGHTED_RATIO"
    written = write_i4_triage(report, tmp_path / "triage.json")
    assert json.loads(written.read_text())["gate_id"] == "I4-TRIAGE"
