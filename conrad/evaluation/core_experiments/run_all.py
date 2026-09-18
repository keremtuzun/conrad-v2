"""Run the CORE-* experiments with their small CPU configs.

python -m conrad.evaluation.core_experiments.run_all [--only CORE-TBD-E001] [--out artifacts/experiments/core]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from conrad.evaluation.core_experiments import (
    assoc_e001,
    buo_e001,
    full_e001,
    persist_e001,
    rbp_e001,
    tbd_e001,
    unc_e001,
)
from conrad.evaluation.core_experiments.common import DEFAULT_SEEDS, load_config

Runner = Callable[[Mapping[str, Any], int | Sequence[int], str | Path], dict[str, Any]]

EXPERIMENTS: dict[str, tuple[Runner, str]] = {
    "CORE-ASSOC-E001": (assoc_e001.run, "configs/eval/core_assoc_e001.yaml"),
    "CORE-BUO-E001": (buo_e001.run, "configs/eval/core_buo_e001.yaml"),
    "CORE-UNC-E001": (unc_e001.run, "configs/eval/core_unc_e001.yaml"),
    "CORE-RBP-E001": (rbp_e001.run, "configs/eval/core_rbp_e001.yaml"),
    "CORE-TBD-E001": (tbd_e001.run, "configs/eval/core_tbd_e001.yaml"),
    "CORE-PERSIST-E001": (persist_e001.run, "configs/eval/core_persist_e001.yaml"),
    "CORE-FULL-E001": (full_e001.run, "configs/eval/core_full_e001.yaml"),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--out", default="artifacts/experiments/core")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    names = args.only or list(EXPERIMENTS)
    for name in names:
        runner, cfg_path = EXPERIMENTS[name]
        config = load_config(Path(args.root) / cfg_path)
        seeds = [int(s) for s in config.get("seeds", DEFAULT_SEEDS)]
        result = runner(config, seeds, args.out)
        print(json.dumps({"experiment": name, "wall_time_s": round(result["wall_time_s"], 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
