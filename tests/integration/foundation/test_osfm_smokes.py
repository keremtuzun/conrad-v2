from __future__ import annotations

import json
from pathlib import Path

from conrad.foundation.pretraining.smoke import public_real_subpipe_probe
from conrad.training.entrypoints import run_training


def test_osfm_contract_tiny_end_to_end_cpu_smoke(tmp_path: Path) -> None:
    result = run_training("configs/train/osfm/contract_tiny.yaml", runs_root=tmp_path)
    assert result["component"] == "foundation.osfm"
    assert result["checkpoint_id"]
    assert result["public_real_ingestion"]["status"] == "BLOCKED_EXTERNAL"
    report = Path(result["run_dir"]) / "reports" / "promotion_record.json"
    assert json.loads(report.read_text(encoding="utf-8"))["promoted"] is True


def test_osfm_s_smoke_architecture_faithful_wiring(tmp_path: Path) -> None:
    result = run_training("configs/train/osfm/s_smoke.yaml", runs_root=tmp_path)
    assert result["architecture_faithful"] is True
    assert result["metrics"]["representation_effective_rank"] == 64.0
    assert Path(result["checkpoint"]).is_file()


def test_public_real_subpipe_reports_blocked_external_without_payload() -> None:
    assert public_real_subpipe_probe({})["status"] == "BLOCKED_EXTERNAL"


def test_osfm_u1_range_smoke_cli_registered(tmp_path: Path) -> None:
    result = run_training("configs/train/osfm/u1_range_smoke.yaml", runs_root=tmp_path)
    assert result["experiment_id"] == "OSFM-U1-RANGE-SMOKE-001"
    assert result["component"] == "foundation.osfm.u1_range"
    metric = result["objectives"]["u1_range_metric_reconstruction"]
    assert metric["u1_range_metric_reconstruction/status"] == "ACTIVE"
