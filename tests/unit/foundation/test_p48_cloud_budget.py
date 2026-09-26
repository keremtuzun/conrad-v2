from __future__ import annotations

import json
from pathlib import Path

import yaml

from conrad.foundation.pretraining.cloud_budget import estimate_p48_cloud_budget, write_cloud_budget_report


def _rehearsal(path: Path) -> Path:
    path.write_text(
        json.dumps({"compute": {"throughput_samples_per_s": 0.125}}),
        encoding="utf-8",
    )
    return path


def _config(path: Path, *, steps: int, batch_size: int = 8) -> Path:
    path.write_text(yaml.safe_dump({"optimizer_steps": steps, "batch_size": batch_size}), encoding="utf-8")
    return path


def test_p48_cloud_budget_blocks_full_run_above_user_cap(tmp_path: Path) -> None:
    report = estimate_p48_cloud_budget(
        config_path=_config(tmp_path / "full.yaml", steps=100_000),
        rehearsal_report=_rehearsal(tmp_path / "rehearsal.json"),
        cap_tl=1000.0,
        tl_per_usd=41.0,
        usd_per_hour=0.648,
    )
    assert report["decision"] == "NO-GO"
    assert report["blockers"][0]["blocker_id"] == "P48-CLOUD-BUDGET-01"
    assert report["estimate"]["tl"] > 1000.0


def test_p48_cloud_budget_allows_small_pilot_under_user_cap(tmp_path: Path) -> None:
    report = estimate_p48_cloud_budget(
        config_path=_config(tmp_path / "pilot.yaml", steps=100),
        rehearsal_report=_rehearsal(tmp_path / "rehearsal.json"),
        cap_tl=1000.0,
        tl_per_usd=41.0,
        usd_per_hour=0.648,
    )
    assert report["decision"] == "GO"
    assert report["blockers"] == []
    written = write_cloud_budget_report(report, tmp_path / "budget.json")
    assert json.loads(written.read_text())["gate_id"] == "P4.8-CLOUD-BUDGET"
