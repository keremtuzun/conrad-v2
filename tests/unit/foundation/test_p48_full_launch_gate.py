from __future__ import annotations

import copy
import json
from pathlib import Path

from conrad.foundation.pretraining.p48_gate import evaluate_p48_full_launch, write_p48_full_launch_report


def _p47_report() -> dict:
    return {
        "schema_version": "1.0.0",
        "gate_id": "P4.7",
        "status": "VALIDATED-RUN",
        "decision": "GO",
        "blockers": [],
        "report_digest": "digest-p47",
    }


def _rehearsal_report() -> dict:
    return {
        "component": "foundation.osfm.u1_sonar",
        "experiment_id": "P4.8A-U1-SONAR-REAL-REHEARSAL",
        "formal_p4_8": False,
        "rehearsal_only": True,
        "gate": {
            "p47_report_digest": "digest-p47",
            "compute": {"cuda_available": True, "cuda_devices": ["NVIDIA L4"]},
            "measured_cuda_vram_gb": [22],
            "clean_tracked_git": True,
        },
        "data": {
            "train": {
                "partition": "PRETRAIN_REAL",
                "sample_ids": ["train-a", "train-b"],
                "normalization_fit_partitions": ["PRETRAIN_REAL"],
                "normalization_excluded_partitions": ["VALIDATION", "FINAL_TEST", "OOD_TEST"],
                "rendered_sonar_limitation": "SubPipe sonar is rendered side-scan imagery, not raw acoustic backscatter.",
            },
            "validation": {
                "partition": "VALIDATION",
                "sample_ids": ["val-a", "val-b"],
                "normalization_fit_partitions": ["PRETRAIN_REAL"],
                "normalization_excluded_partitions": ["VALIDATION", "FINAL_TEST", "OOD_TEST"],
                "rendered_sonar_limitation": "SubPipe sonar is rendered side-scan imagery, not raw acoustic backscatter.",
            },
            "final_test_used": False,
            "ood_test_used": False,
            "limitation": "SubPipe sonar is rendered side-scan imagery, not raw acoustic backscatter.",
        },
        "representation_health": {"finite": True},
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_p48_full_launch_gate_fails_closed_until_full_run_path_exists(tmp_path: Path) -> None:
    p47 = _write(tmp_path / "p47.json", _p47_report())
    rehearsal = _write(tmp_path / "rehearsal.json", _rehearsal_report())
    report = evaluate_p48_full_launch(rehearsal, p47)
    assert report["status"] == "VALIDATED-RUN"
    assert report["decision"] == "NO-GO"
    assert [b["blocker_id"] for b in report["blockers"]] == ["P48-FULL-RUN-IMPLEMENTATION-01"]
    written = write_p48_full_launch_report(report, tmp_path / "gate" / "preflight.json")
    assert json.loads(written.read_text())["gate_id"] == "P4.8-FULL-LAUNCH"


def test_p48_full_launch_gate_fails_closed_on_missing_rehearsal(tmp_path: Path) -> None:
    p47 = _write(tmp_path / "p47.json", _p47_report())
    report = evaluate_p48_full_launch(tmp_path / "missing.json", p47)
    blockers = {b["blocker_id"] for b in report["blockers"]}
    assert report["decision"] == "NO-GO"
    assert "P48A-EVIDENCE-01" in blockers
    assert "P48-FULL-RUN-IMPLEMENTATION-01" in blockers


def test_p48_full_launch_gate_detects_partition_and_metadata_leakage(tmp_path: Path) -> None:
    p47 = _write(tmp_path / "p47.json", _p47_report())
    bad = copy.deepcopy(_rehearsal_report())
    bad["data"]["validation"]["sample_ids"] = ["train-a"]
    bad["data"]["final_test_used"] = True
    bad["data"]["train"]["normalization_excluded_partitions"] = ["FINAL_TEST", "OOD_TEST"]
    bad["data"]["limitation"] = "sonar"
    bad["data"]["train"]["rendered_sonar_limitation"] = "sonar"
    bad["data"]["validation"]["rendered_sonar_limitation"] = "sonar"
    rehearsal = _write(tmp_path / "rehearsal.json", bad)
    report = evaluate_p48_full_launch(rehearsal, p47)
    blockers = {b["blocker_id"] for b in report["blockers"]}
    assert "P48-DATA-LEAK-01" in blockers
    assert "P48-DATA-OVERLAP-01" in blockers
    assert "P48-STATS-TRAIN-ONLY-01" in blockers
    assert "P48-SONAR-CLAIM-01" in blockers
