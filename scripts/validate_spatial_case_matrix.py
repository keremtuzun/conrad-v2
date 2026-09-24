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
    ("subthreshold_crack_length", {"crack_length_m": 0.009, "crack_depth_m": 0.006}, "INTACT", True),
    ("subthreshold_crack_depth", {"crack_length_m": 0.02, "crack_depth_m": 0.0009}, "INTACT", True),
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
    parser.add_argument("--resume", action="store_true", help="continue only cases without a result")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {out}")
    protocol = {"seeds": SEEDS, "cases": CASES, "total": len(SEEDS) * len(CASES)}
    if args.resume:
        if not out.is_dir() or json.loads((out / "protocol.json").read_text(encoding="utf-8")) != json.loads(
            json.dumps(protocol)
        ):
            raise SystemExit("cannot resume: protocol directory or declaration mismatch")
    else:
        out.mkdir(parents=True)
        (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    result_path = out / "results.jsonl"
    existing = (
        [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
        if args.resume and result_path.exists()
        else []
    )
    by_key = {(row["seed"], row["case"]): row for row in existing}
    declared_keys = {(seed, name) for seed in SEEDS for name, *_ in CASES}
    if len(by_key) != len(existing) or not set(by_key).issubset(declared_keys):
        raise SystemExit("cannot resume duplicate or undeclared case results")
    with result_path.open("a" if args.resume else "w", encoding="utf-8") as results:
        for seed in SEEDS:
            for name, state, expected, full_sweep in CASES:
                if (seed, name) in by_key:
                    continue
                case_dir = out / f"seed{seed}_{name}"
                row: dict[str, Any] = {"seed": seed, "case": name, "expected": expected}
                if case_dir.exists():
                    row["status"] = "INTERRUPTED"
                else:
                    case_dir.mkdir()
                    try:
                        _exercise_required_surface(case_dir, state, expected, full_sweep, seed)
                        row["status"] = "PASS"
                    except Exception:
                        row["status"] = "FAIL"
                        row["traceback"] = traceback.format_exc()
                results.write(json.dumps(row) + "\n")
                results.flush()
                by_key[(seed, name)] = row
                print(f"{seed} {name}: {row['status']}", flush=True)
    passed = sum(row["status"] == "PASS" for row in by_key.values())
    summary = {
        "total": len(SEEDS) * len(CASES),
        "passed": passed,
        "failed_or_interrupted": len(CASES) * len(SEEDS) - passed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if summary["failed_or_interrupted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
