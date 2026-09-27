from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conrad.cli import commands  # noqa: F401  (registers CLI subcommands)
from conrad.cli.app import app
from conrad.oceansense.status import evaluate_p5_p10_status, status_digest, write_p5_p10_status


def test_status_gate_is_implemented_but_not_validated_without_qualified_osfm() -> None:
    report = evaluate_p5_p10_status(test_result="PASS")
    assert report["status"] == "IMPLEMENTED"
    assert report["decision"] == "IMPLEMENTED"
    assert report["qualified_checkpoint"]["verified"] is False
    assert report["blockers"] == [
        {
            "blocker_id": "P5-P10-OSFM-01",
            "scope": "EXTERNAL",
            "detail": "qualified OSFM-S-PRETRAIN-V1 metadata path not supplied",
        }
    ]
    assert all(phase["status"] == "IMPLEMENTED" for phase in report["phases"].values())


def test_status_gate_fails_closed_when_tests_are_not_passed() -> None:
    report = evaluate_p5_p10_status(test_result="NOT_RUN")
    assert report["status"] == "DESIGNED"
    assert report["decision"] == "DESIGNED"
    assert {b["blocker_id"] for b in report["blockers"]} == {"P5-P10-TESTS-01", "P5-P10-OSFM-01"}


def test_status_gate_allows_validated_run_only_with_qualified_checkpoint_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "osfm_s_pretrain_v1.json"
    metadata.write_text(
        json.dumps(
            {
                "checkpoint_label": "OSFM-S-PRETRAIN-V1",
                "status": "VALIDATED-RUN",
                "decision": "PROMOTE",
            }
        ),
        encoding="utf-8",
    )
    report = evaluate_p5_p10_status(test_result="PASS", qualified_checkpoint_metadata=metadata)
    assert report["status"] == "VALIDATED-RUN"
    assert report["decision"] == "VALIDATED-RUN"
    assert report["blockers"] == []
    assert all(phase["validated_run_allowed"] for phase in report["phases"].values())


@pytest.mark.parametrize(
    ("metadata", "expected_detail"),
    (
        (
            {"checkpoint_label": "OSFM-S-PRETRAIN-V0", "status": "VALIDATED-RUN", "decision": "PROMOTE"},
            "checkpoint label",
        ),
        (
            {"checkpoint_label": "OSFM-S-PRETRAIN-V1", "status": "IMPLEMENTED", "decision": "PROMOTE"},
            "checkpoint status",
        ),
        (
            {"checkpoint_label": "OSFM-S-PRETRAIN-V1", "status": "VALIDATED-RUN", "decision": "HOLD"},
            "checkpoint decision",
        ),
    ),
)
def test_status_gate_rejects_unqualified_checkpoint_metadata(
    tmp_path: Path,
    metadata: dict[str, str],
    expected_detail: str,
) -> None:
    metadata_path = tmp_path / "candidate_osfm_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = evaluate_p5_p10_status(test_result="PASS", qualified_checkpoint_metadata=metadata_path)
    assert report["status"] == "IMPLEMENTED"
    assert report["decision"] == "IMPLEMENTED"
    assert report["qualified_checkpoint"]["verified"] is False
    assert expected_detail in report["qualified_checkpoint"]["detail"]
    assert report["blockers"] == [
        {
            "blocker_id": "P5-P10-OSFM-01",
            "scope": "EXTERNAL",
            "detail": report["qualified_checkpoint"]["detail"],
        }
    ]
    assert all(not phase["validated_run_allowed"] for phase in report["phases"].values())


def test_status_report_write_is_machine_readable_and_digestible(tmp_path: Path) -> None:
    out = tmp_path / "status.json"
    report = evaluate_p5_p10_status(test_result="PASS")
    written = write_p5_p10_status(report, out)
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["gate_id"] == "P5-P10-DOWNSTREAM-CONTRACTS"
    assert len(status_digest(out)) == 64


def test_cli_writes_p5_p10_status_artifact(tmp_path: Path) -> None:
    out = tmp_path / "status.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            "oceansense-p5-p10-status",
            "--test-result",
            "PASS",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["decision"] == "IMPLEMENTED"
    assert loaded["blockers"][0]["blocker_id"] == "P5-P10-OSFM-01"
