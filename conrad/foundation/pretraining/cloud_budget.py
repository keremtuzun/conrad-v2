"""Cloud GPU budget preflight for governed OS-FM runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from conrad.settings import REPO_ROOT


DEFAULT_REHEARSAL_REPORT = Path("artifacts/gates/P4.8A/rehearsal_l4/reports/p48a_u1_sonar_rehearsal_report.json")
DEFAULT_RESEARCH_CONFIG = Path("configs/train/osfm/research/u1_sonar_research.yaml")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def estimate_p48_cloud_budget(
    *,
    config_path: str | Path = DEFAULT_RESEARCH_CONFIG,
    rehearsal_report: str | Path = DEFAULT_REHEARSAL_REPORT,
    cap_tl: float = 1000.0,
    tl_per_usd: float = 41.0,
    usd_per_hour: float = 0.648,
    safety_factor: float = 1.25,
) -> dict[str, Any]:
    config = yaml.safe_load(_resolve(config_path).read_text(encoding="utf-8"))
    report = json.loads(_resolve(rehearsal_report).read_text(encoding="utf-8"))
    steps = int(config["optimizer_steps"])
    batch_size = int(config["batch_size"])
    throughput = float(report["compute"]["throughput_samples_per_s"])
    estimated_seconds = (steps * batch_size) / throughput
    estimated_hours = estimated_seconds / 3600.0
    estimated_usd = estimated_hours * usd_per_hour * safety_factor
    estimated_tl = estimated_usd * tl_per_usd
    cap_usd = cap_tl / tl_per_usd
    max_hours_under_cap = cap_usd / (usd_per_hour * safety_factor)
    max_steps_under_cap = int((max_hours_under_cap * 3600.0 * throughput) // batch_size)
    decision = "GO" if estimated_tl <= cap_tl else "NO-GO"
    return {
        "schema_version": "1.0.0",
        "gate_id": "P4.8-CLOUD-BUDGET",
        "decision": decision,
        "cap_tl": cap_tl,
        "assumptions": {
            "tl_per_usd": tl_per_usd,
            "usd_per_hour": usd_per_hour,
            "safety_factor": safety_factor,
            "source": "P4.8A measured L4 rehearsal throughput and user cloud cap",
        },
        "config": {
            "path": str(_resolve(config_path)),
            "optimizer_steps": steps,
            "batch_size": batch_size,
        },
        "measured_rehearsal": {
            "path": str(_resolve(rehearsal_report)),
            "throughput_samples_per_s": throughput,
        },
        "estimate": {
            "hours": estimated_hours,
            "usd": estimated_usd,
            "tl": estimated_tl,
            "max_optimizer_steps_under_cap": max_steps_under_cap,
        },
        "blockers": []
        if decision == "GO"
        else [
            {
                "blocker_id": "P48-CLOUD-BUDGET-01",
                "scope": "USER_BUDGET",
                "detail": f"estimated {estimated_tl:.2f} TL exceeds cap {cap_tl:.2f} TL",
            }
        ],
    }


def write_cloud_budget_report(report: dict[str, Any], output: str | Path) -> Path:
    path = _resolve(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
