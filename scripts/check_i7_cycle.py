"""Evaluate a paired I7 bandwidth/outage cycle with the frozen surrogate rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conrad.evaluation.decision_experiments.com_i7 import assess_cycle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandwidth", type=Path, required=True)
    parser.add_argument("--outage", type=Path, required=True)
    parser.add_argument("--partition", choices=("validation", "final_test"), required=True)
    parser.add_argument("--partition-domain", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = assess_cycle(
        json.loads(args.bandwidth.read_text(encoding="utf-8")),
        json.loads(args.outage.read_text(encoding="utf-8")),
        expected_partition=args.partition,
        expected_domain=args.partition_domain,
        expected_seeds=args.seeds,
    )
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
