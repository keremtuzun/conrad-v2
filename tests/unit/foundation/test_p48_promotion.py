from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from conrad.foundation.pretraining.p48_promotion import (
    evaluate_p48_promotion,
    write_p48_promotion_report,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"qualified-checkpoint")
    checkpoint_id = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    p47 = _write_json(
        tmp_path / "p47.json",
        {
            "status": "VALIDATED-RUN",
            "decision": "GO",
            "blockers": [],
            "report_digest": "p47-r02-digest",
        },
    )
    report = _write_json(
        tmp_path / "report.json",
        {
            "experiment_id": "P4.8-U1-SONAR-RESEARCH",
            "formal_p4_8": True,
            "rehearsal_only": False,
            "gate": {
                "p47_report_digest": "p47-r02-digest",
                "clean_tracked_git": True,
                "compute": {"cuda_available": True},
                "measured_cuda_vram_gb": [22],
            },
            "data": {
                "train": {
                    "partition": "PRETRAIN_REAL",
                    "sample_ids": ["train-a", "train-b"],
                    "normalization_fit_partitions": ["PRETRAIN_REAL"],
                    "normalization_excluded_partitions": ["VALIDATION", "FINAL_TEST", "OOD_TEST"],
                    "rendered_sonar_limitation": (
                        "SubPipe sonar is rendered side-scan imagery, not raw acoustic backscatter."
                    ),
                },
                "validation": {
                    "partition": "VALIDATION",
                    "sample_ids": ["val-a", "val-b"],
                    "normalization_fit_partitions": ["PRETRAIN_REAL"],
                    "normalization_excluded_partitions": ["VALIDATION", "FINAL_TEST", "OOD_TEST"],
                    "rendered_sonar_limitation": (
                        "SubPipe sonar is rendered side-scan imagery, not raw acoustic backscatter."
                    ),
                },
                "final_test_used": False,
                "ood_test_used": False,
                "limitation": "SubPipe sonar is rendered side-scan imagery, not raw acoustic backscatter.",
            },
            "checkpoint": str(checkpoint),
            "checkpoint_id": checkpoint_id,
            "metrics": {"optimizer_steps": 100_000},
            "representation_health": {
                "finite": True,
                "collapse_score": 0.01,
                "effective_rank": 70.0,
                "formal_rank_guard": "PASS",
            },
            "reload": {"matches": True, "max_abs_diff": 0.0},
        },
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "stage_id": "U1-SONAR-RESEARCH",
                "promotable": True,
                "optimizer_steps": 100_000,
            }
        ),
        encoding="utf-8",
    )
    state = _write_json(tmp_path / "run_state.json", {"status": "COMPLETED", "comparability": "COMPARABLE"})
    return {"checkpoint": checkpoint, "p47": p47, "report": report, "config": config, "state": state}


def test_p48_promotion_accepts_only_complete_qualified_run(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    result = evaluate_p48_promotion(
        run_report=paths["report"],
        resolved_config=paths["config"],
        run_state=paths["state"],
        p47_report=paths["p47"],
    )
    assert result["status"] == "VALIDATED-RUN"
    assert result["decision"] == "PROMOTE"
    assert result["blockers"] == []
    assert result["promotion_record"]["next_stage"] == "P4.9"
    written = write_p48_promotion_report(result, tmp_path / "gate" / "promotion.json")
    assert json.loads(written.read_text())["decision"] == "PROMOTE"


def test_p48_promotion_rejects_budget_pilot_even_when_completed(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    config = yaml.safe_load(paths["config"].read_text())
    config.update(
        {
            "stage_id": "P4.8C-U1-SONAR-BUDGET-PILOT",
            "promotable": False,
            "optimizer_steps": 1000,
        }
    )
    paths["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    report = json.loads(paths["report"].read_text())
    report["metrics"]["optimizer_steps"] = 1000
    report["representation_health"].update(
        {"formal_rank_guard": "NOT_EVALUABLE", "effective_rank": 7.0}
    )
    paths["report"].write_text(json.dumps(report), encoding="utf-8")

    result = evaluate_p48_promotion(
        run_report=paths["report"],
        resolved_config=paths["config"],
        run_state=paths["state"],
        p47_report=paths["p47"],
    )
    blockers = {blocker["blocker_id"] for blocker in result["blockers"]}
    assert result["decision"] == "DO-NOT-PROMOTE"
    assert "P48-PROMOTION-CONFIG-02" in blockers
    assert "P48-PROMOTION-STEPS-01" in blockers
    assert "P48-PROMOTION-HEALTH-01" in blockers


def test_p48_promotion_rejects_checkpoint_digest_mismatch(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    paths["checkpoint"].write_bytes(b"tampered")
    result = evaluate_p48_promotion(
        run_report=paths["report"],
        resolved_config=paths["config"],
        run_state=paths["state"],
        p47_report=paths["p47"],
    )
    assert result["decision"] == "DO-NOT-PROMOTE"
    assert {blocker["blocker_id"] for blocker in result["blockers"]} == {
        "P48-PROMOTION-CHECKPOINT-02"
    }
