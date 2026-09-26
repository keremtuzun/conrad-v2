"""Read-only supplemental leakage audit for the completed Spatial V1.1 Unity cycle.

The formal test crashed while labelling scan results because score rows do not
contain ``run_id``.  This utility scans the already-written bundles without
running Unity or changing the immutable formal verdict.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/unity_live"))

from unity_gate_support import leakage_scan  # noqa: E402

RESULTS = ROOT / "artifacts/gates/I5_SPATIAL_V1_1/unity_i5_spatial_v1_1_results.json"
OUTPUT = ROOT / "artifacts/gates/I5_SPATIAL_V1_1/leakage_supplement.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> dict[str, Any]:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite supplemental audit {OUTPUT}")
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    runs: dict[str, dict[str, Any]] = {}
    violations: list[str] = []
    for row in result["per_run"]:
        relative = str(row["run_dir"])
        run_dir = ROOT / relative
        truth = json.loads((run_dir / "truth/truth_record.json").read_text(encoding="utf-8"))
        count, bad = leakage_scan(run_dir, set(truth["meta"]["world_entity_ids"]), set())
        runs[relative] = {"texts_scanned": count, "violations": bad}
        violations.extend(f"{relative}: {item}" for item in bad)
    artifact = {
        "audit": "I5 Spatial V1.1 completed-bundle leakage supplement",
        "evidence_role": "SUPPLEMENTAL_READ_ONLY_AUDIT",
        "formal_test_node_status": "ERROR",
        "formal_test_error": "KeyError: 'run_id' before scan result recording completed",
        "formal_verdict_changed": False,
        "unity_rerun": False,
        "source_results": str(RESULTS.relative_to(ROOT)),
        "source_results_sha256": _sha256(RESULTS),
        "expected_runs": 42,
        "runs_scanned": len(runs),
        "all_runs_scanned": len(runs) == 42,
        "supplemental_status": "PASS" if len(runs) == 42 and not violations else "FAIL",
        "violations": violations,
        "runs": runs,
        "notes": (
            "This scan only reads the exact completed formal bundles. It cannot convert the "
            "formal ERROR node to PASS and cannot change the 7/10 formal FAIL."
        ),
    }
    OUTPUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    report = audit()
    print(
        f"supplemental leakage {report['supplemental_status']}: "
        f"runs={report['runs_scanned']}/{report['expected_runs']} "
        f"violations={len(report['violations'])}"
    )
    raise SystemExit(0 if report["supplemental_status"] == "PASS" else 1)
