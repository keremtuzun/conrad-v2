from __future__ import annotations

import json
from pathlib import Path

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
            "detail": "final integrated validation requires verified OSFM-S-PRETRAIN-V1",
        }
    ]


def test_requirement_audit_fails_closed_when_contract_tests_have_not_passed() -> None:
    report = evaluate_p5_p10_requirement_audit(tests_passed=False)
    assert report["status"] == "DESIGNED"
    assert all(row["status"] == "DESIGNED" for row in report["requirements"])
    assert report["remaining_blockers"][0]["blocker_id"] == "P5-P10-OSFM-01"


def test_requirement_audit_allows_validated_run_only_with_qualified_checkpoint() -> None:
    report = evaluate_p5_p10_requirement_audit(tests_passed=True, qualified_checkpoint_verified=True)
    assert report["status"] == "VALIDATED-RUN"
    assert report["remaining_blockers"] == []
    assert all(row["status"] == "VALIDATED-RUN" for row in report["requirements"])
    assert all(row["validated_run_allowed"] for row in report["requirements"])


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
