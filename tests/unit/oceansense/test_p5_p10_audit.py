from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conrad.cli import commands  # noqa: F401  (registers CLI subcommands)
from conrad.cli.app import app
from conrad.oceansense.audit import evaluate_p5_p10_requirement_audit, write_p5_p10_requirement_audit


def test_requirement_audit_records_all_p5_p10_rows_without_qualified_osfm() -> None:
    report = evaluate_p5_p10_requirement_audit(tests_passed=True)
    assert report["audit_id"] == "P5-P10-REQUIREMENT-AUDIT"
    assert report["status"] == "IMPLEMENTED"
    assert {row["phase"] for row in report["requirements"]} == {"P5", "P6", "P7", "P8", "P9", "P10", "P5-P10"}
    assert all(row["final_validation_dependency"] == "OSFM-S-PRETRAIN-V1" for row in report["requirements"])
    assert all(not row["validated_run_allowed"] for row in report["requirements"])
    assert report["remaining_blockers"] == [
        {
            "blocker_id": "P5-P10-OSFM-01",
            "scope": "EXTERNAL",
            "detail": "qualified OSFM-S-PRETRAIN-V1 metadata path not supplied",
        }
    ]


def test_requirement_audit_fails_closed_when_contract_tests_have_not_passed() -> None:
    report = evaluate_p5_p10_requirement_audit(tests_passed=False)
    assert report["status"] == "DESIGNED"
    assert all(row["status"] == "DESIGNED" for row in report["requirements"])
    assert report["remaining_blockers"][0]["blocker_id"] == "P5-P10-OSFM-01"


def test_requirement_audit_allows_validated_run_only_with_qualified_checkpoint_metadata(tmp_path: Path) -> None:
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
    report = evaluate_p5_p10_requirement_audit(tests_passed=True, qualified_checkpoint_metadata=metadata)
    assert report["status"] == "VALIDATED-RUN"
    assert report["remaining_blockers"] == []
    assert all(row["status"] == "VALIDATED-RUN" for row in report["requirements"])
    assert all(row["validated_run_allowed"] for row in report["requirements"])


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
def test_requirement_audit_rejects_unqualified_checkpoint_metadata(
    tmp_path: Path,
    metadata: dict[str, str],
    expected_detail: str,
) -> None:
    metadata_path = tmp_path / "candidate_osfm_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = evaluate_p5_p10_requirement_audit(tests_passed=True, qualified_checkpoint_metadata=metadata_path)
    assert report["status"] == "IMPLEMENTED"
    assert report["qualified_checkpoint_verified"] is False
    assert expected_detail in report["qualified_checkpoint"]["detail"]
    assert report["remaining_blockers"] == [
        {
            "blocker_id": "P5-P10-OSFM-01",
            "scope": "EXTERNAL",
            "detail": report["qualified_checkpoint"]["detail"],
        }
    ]
    assert all(row["status"] == "IMPLEMENTED" for row in report["requirements"])
    assert all(not row["validated_run_allowed"] for row in report["requirements"])


def test_requirement_audit_write_and_cli(tmp_path: Path) -> None:
    written = write_p5_p10_requirement_audit(
        evaluate_p5_p10_requirement_audit(tests_passed=True),
        tmp_path / "audit.json",
    )
    assert json.loads(written.read_text(encoding="utf-8"))["status"] == "IMPLEMENTED"

    cli_out = tmp_path / "cli_audit.json"
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "oceansense-p5-p10-audit",
            "--tests-passed",
            "--output",
            str(cli_out),
        ],
    )
    assert result.exit_code == 0
    loaded = json.loads(cli_out.read_text(encoding="utf-8"))
    assert loaded["audit_id"] == "P5-P10-REQUIREMENT-AUDIT"
    assert loaded["status"] == "IMPLEMENTED"
