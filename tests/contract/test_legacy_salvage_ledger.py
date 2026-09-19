"""The legacy salvage ledger is complete and consistent (prompt s4-s5): nothing ported without a destination."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ALLOWED = {
    "REUSE_WITHOUT_SEMANTIC_CHANGE",
    "PORT_AND_ADAPT",
    "REIMPLEMENT_FROM_CONCEPT",
    "REFERENCE_ONLY",
    "OBSOLETE",
    "REJECT_ARCHITECTURAL_CONFLICT",
}


def test_salvage_ledger_entries_are_classified_and_consistent() -> None:
    data = json.loads(
        (ROOT / "artifacts/migration/digital-twin-salvage-ledger.json").read_text(encoding="utf-8")
    )
    entries = data["entries"] if isinstance(data, dict) else data
    assert len(entries) >= 20
    for e in entries:
        assert e["classification"] in ALLOWED, e
        ported = bool(e.get("ported"))
        porting = e["classification"] in {"REUSE_WITHOUT_SEMANTIC_CHANGE", "PORT_AND_ADAPT"}
        assert ported == bool(e.get("destination_path")), e["legacy_path"]
        if ported:
            assert porting and (ROOT / e["destination_path"]).exists()
    assert (ROOT / "docs/migration/DIGITAL_TWIN_LEGACY_AUDIT.md").exists()
    assert (ROOT / "docs/migration/DIGITAL_TWIN_SALVAGE_REPORT.md").exists()
