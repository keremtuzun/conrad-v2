"""Run the prospectively declared Spatial V1 controlled-view validation once.

This is a software validation matrix through the mission sensor and belief
pipeline. It does not replace autonomous closed-loop or formal Unity evidence.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from tests.integration.test_spatial_mission_belief import _exercise_required_surface

SEEDS = (1401, 1402)
CASES: tuple[tuple[str, Any, str | None, bool], ...] = (
    ("healthy", None, "INTACT", True),
    ("partial_healthy", None, None, False),
    ("outside_support", "outside-support", None, False),
    ("occluded_defect", "occluded-defect", None, False),
    ("covered_corrosion", {"corrosion_depth_m": 0.008}, "SEVERE", True),
    ("covered_crack", {"crack_length_m": 0.02, "crack_depth_m": 0.006}, "SEVERE", True),
    ("subresolution", "subresolution", "INTACT", True),
    ("heterogeneous_healthy", "heterogeneous-healthy", "INTACT", True),
    ("multiple_defects", "multiple-defects", "SEVERE", True),
    ("edge_defect", "edge-defect", "SEVERE", True),
    ("uniform_same_mean", "uniform-same-mean", "INTACT", True),
    ("local_same_mean", "local-same-mean", "SEVERE", True),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new, absent validation directory")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    protocol = {"seeds": SEEDS, "cases": CASES, "total": len(SEEDS) * len(CASES)}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    failed = 0
    with (out / "results.jsonl").open("w", encoding="utf-8") as results:
        for seed in SEEDS:
            for name, state, expected, full_sweep in CASES:
                case_dir = out / f"seed{seed}_{name}"
                case_dir.mkdir()
                row: dict[str, Any] = {"seed": seed, "case": name, "expected": expected}
                try:
                    _exercise_required_surface(case_dir, state, expected, full_sweep, seed)
                    row["status"] = "PASS"
                except Exception:
                    row["status"] = "FAIL"
                    row["traceback"] = traceback.format_exc()
                    failed += 1
                results.write(json.dumps(row) + "\n")
                results.flush()
                print(f"{seed} {name}: {row['status']}", flush=True)
    summary = {"total": len(SEEDS) * len(CASES), "passed": len(SEEDS) * len(CASES) - failed, "failed": failed}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
