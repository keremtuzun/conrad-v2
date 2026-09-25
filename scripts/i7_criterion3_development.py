"""Run a declared I7 criterion-3 screen on mission DEVELOPMENT worlds only.

This is diagnostic evidence, never gate evidence.  Keeping the entry point in a real
module is also required for Windows ``multiprocessing`` spawn; an inline stdin script
cannot be imported by worker processes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from conrad.evaluation.decision_experiments.com_i7 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--scenario", default="I7-OUTAGE-CRITICAL")
    parser.add_argument("--levels", nargs="+", type=float, default=[0.1])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--compact-summary", action="store_true")
    args = parser.parse_args()

    config = {
        "experiment_id": args.experiment_id,
        "hypothesis": (
            "screen development worlds for low-bandwidth criterion-3 failure classes before repair"
        ),
        "primary_metric": "mission_information_retained",
        "direction": "higher",
        "baselines": [
            "C-B0_send_all",
            "C-B1_fifo",
            "C-B2_fixed_priority",
            "C-B4_value_per_bit",
        ],
        "partition": "development",
        "partition_domain": "mission",
        "scenario": args.scenario,
        "bandwidth_levels": args.levels,
        "closed_loop_levels": [],
        "deadline_linkup_s": {"critical": 30.0, "routine": 120.0},
        "queue_trace_every_s": 5,
        "workers": args.workers,
        "keep_bundles": False,
    }
    if args.compact_summary:
        config["baac_override"] = {
            "model_version": "baac-critical-summary-v1",
            "compact_critical_summary": True,
        }
    out = Path("artifacts") / "experiments" / args.experiment_id
    run(config, args.seeds, out)


if __name__ == "__main__":
    main()
