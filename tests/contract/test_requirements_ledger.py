"""The requirements ledger is generated, current, and never silently drops a requirement (prompt s7)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _builder():
    spec = importlib.util.spec_from_file_location("ledger", ROOT / "scripts" / "build_requirements_ledger.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ledger_rows_are_unique_and_statuses_legal() -> None:
    rows = _builder().parse()
    ids = [r["requirement_id"] for r in rows]
    assert len(ids) == len(set(ids)) and len(ids) >= 80
    legal = {
        "MISSING",
        "PARTIAL",
        "IMPLEMENTED",
        "TESTED",
        "INTEGRATED",
        "EVALUATED",
        "VALIDATED",
        "BLOCKED_EXTERNAL",
        "BLOCKED_OPEN_DECISION",
    }
    assert {r["implementation_status"] for r in rows} <= legal


def test_committed_ledger_covers_every_source_requirement() -> None:
    committed = json.loads(
        (ROOT / "artifacts" / "spec" / "requirements-ledger.json").read_text(encoding="utf-8")
    )
    committed_ids = {r["requirement_id"] for r in committed["requirements"]}
    source_ids = {r["requirement_id"] for r in _builder().parse()}
    assert source_ids == committed_ids, "regenerate with scripts/build_requirements_ledger.py"


def test_tested_requirements_point_at_existing_code_and_tests() -> None:
    for row in _builder().parse():
        if row["implementation_status"] in ("TESTED", "INTEGRATED"):
            assert not row["missing_paths"], row["requirement_id"]


def test_blocked_rows_name_their_external_dependency() -> None:
    for row in _builder().parse():
        if row["implementation_status"] == "BLOCKED_EXTERNAL":
            assert row["external_dependencies"], row["requirement_id"]
