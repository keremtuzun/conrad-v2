"""Paired ten-world development screen of the I4 no-zero-value candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = (
    "hidden_state_error_improvement",
    "info_per_kj",
    "energy_j",
    "target_observed",
    "collisions",
    "redundant_observations",
    "travel_m",
)


def _row(root: Path, seed: int, arm: str) -> dict:
    path = root / f"I4-ENERGY-DIAG-s{seed}-{arm}" / "reports" / "i4_energy_diagnostic.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("incumbent_root", type=Path)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for seed in range(8100000, 8100010):
        incumbent = _row(args.incumbent_root, seed, "V4_full")
        candidate = _row(args.candidate_root, seed, "V4_value_filter")
        rows.append(
            {
                "seed": seed,
                "incumbent": {k: incumbent["mission_metrics"][k] for k in METRICS},
                "candidate": {k: candidate["mission_metrics"][k] for k in METRICS},
                "incumbent_views": len(incumbent["views"]),
                "candidate_views": len(candidate["views"]),
            }
        )
    means = {}
    for metric in METRICS:
        old = np.asarray([r["incumbent"][metric] for r in rows], dtype=float)
        new = np.asarray([r["candidate"][metric] for r in rows], dtype=float)
        means[metric] = {
            "incumbent": float(old.mean()),
            "candidate": float(new.mean()),
            "paired_difference": float((new - old).mean()),
        }
    body = {
        "evidence_class": "DEVELOPMENT_SCREEN; Python kernel only",
        "seeds": list(range(8100000, 8100010)),
        "metrics": means,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1), encoding="utf-8")
    print(json.dumps(means, indent=1))


if __name__ == "__main__":
    main()
