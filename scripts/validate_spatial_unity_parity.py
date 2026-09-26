"""Run the prospectively declared Spatial V1 Unity parity matrix once."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from itertools import product
from pathlib import Path
from typing import Any

from conrad.sim.unity.player import find_player
from tests.unity_live.test_spatial_unity_parity import _exercise_kernel_unity_parity

PLAYER_SHA256 = "36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277"
SEEDS = (1401, 1402)
CASES = tuple(product((False, True), repeat=3))  # occluded, noisy, bounded survey
PROTOCOL_EXTRA: dict[str, Any] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new, absent validation directory")
    parser.add_argument("--resume", action="store_true", help="continue only cases without a result")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {out}")
    player = find_player()
    if player is None:
        raise SystemExit("Unity player unavailable")
    digest = hashlib.sha256(player.read_bytes()).hexdigest()
    if digest != PLAYER_SHA256:
        raise SystemExit(f"Unity player hash mismatch: {digest}")
    protocol = {
        "seeds": SEEDS,
        "cases": CASES,
        "player_sha256": PLAYER_SHA256,
        **PROTOCOL_EXTRA,
    }
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
    by_key = {(row["seed"], row["occluded"], row["noisy"], row["bounded_survey"]): row for row in existing}
    declared_keys = {(seed, *case) for seed in SEEDS for case in CASES}
    if len(by_key) != len(existing) or not set(by_key).issubset(declared_keys):
        raise SystemExit("cannot resume duplicate or undeclared case results")
    with result_path.open("a" if args.resume else "w", encoding="utf-8") as results:
        for seed in SEEDS:
            for occluded, noisy, bounded in CASES:
                key = (seed, occluded, noisy, bounded)
                if key in by_key:
                    continue
                name = f"seed{seed}_occ{int(occluded)}_noise{int(noisy)}_bound{int(bounded)}"
                case_dir = out / name
                row: dict[str, Any] = {
                    "seed": seed,
                    "occluded": occluded,
                    "noisy": noisy,
                    "bounded_survey": bounded,
                }
                if case_dir.exists():
                    row["status"] = "INTERRUPTED"
                else:
                    case_dir.mkdir()
                    try:
                        _exercise_kernel_unity_parity(case_dir, occluded, noisy, bounded, seed)
                        row["status"] = "PASS"
                    except Exception:
                        row["status"] = "FAIL"
                        row["traceback"] = traceback.format_exc()
                results.write(json.dumps(row) + "\n")
                results.flush()
                by_key[key] = row
                print(f"{name}: {row['status']}", flush=True)
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
