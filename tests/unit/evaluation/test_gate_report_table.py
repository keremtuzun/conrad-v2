from __future__ import annotations

import json
from pathlib import Path

import scripts.gate_report_table as table


def test_versioned_evidence_supersedes_base_record(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "artifacts/gates/I5"
    folder.mkdir(parents=True)
    (folder / "evidence_formal.json").write_text(json.dumps({"version": "historical"}))
    (folder / "evidence_formal_spatial_v1_1.json").write_text(json.dumps({"version": "spatial"}))
    monkeypatch.setattr(table, "ROOT", tmp_path)

    assert table._evidence("I5", "formal") == {"version": "spatial"}
