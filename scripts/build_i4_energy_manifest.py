"""Hash complete local I4 development traces without committing their large run bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("artifacts/experiments/I4-ENERGY-DIAG")
ARMS = ("V4_full", "V4_short_leg")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for seed in range(8100000, 8100040):
        for arm in ARMS:
            path = ROOT / f"I4-ENERGY-DIAG-s{seed}-{arm}" / "reports" / "i4_energy_diagnostic.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc["seed"] != seed or doc["arm"] != arm or doc["partition"] != "i4_energy_v1_development":
                raise ValueError(f"trace provenance mismatch: {path}")
            reports[f"{seed}:{arm}"] = {"path": str(path), "sha256": digest(path)}
    tracked = [
        Path("configs/eval/partitions_i4_energy_v1.yaml"),
        Path("scripts/diagnose_i4_energy.py"),
        Path("scripts/summarize_i4_energy.py"),
        Path("scripts/fit_i4_energy_cost.py"),
        Path("conrad/robotics/trajectory/generator.py"),
        Path("conrad/orchestration/mission_config.py"),
        Path("conrad/orchestration/deliberation.py"),
    ]
    return {
        "evidence_kind": "DEVELOPMENT_TRACE_MANIFEST",
        "execution_path": "Python kernel surrogate",
        "base_git_commit": "b6f7cb203aad0ed9aee4b8ea677ece329f5b008a",
        "report_count": len(reports),
        "reports": reports,
        "current_source_sha256": {str(path): digest(path) for path in tracked},
        "source_note": "Current source hashes are archival references, not proof of identical code at every earlier run start.",
        "held_out_status": "validation, final and OOD not read",
    }


if __name__ == "__main__":
    out = ROOT / "development_trace_manifest.json"
    body = build()
    out.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}: {body['report_count']} reports")
