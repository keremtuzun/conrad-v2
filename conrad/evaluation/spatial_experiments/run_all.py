"""Run 2S-E001..E004 from their configs: ``python -m conrad.evaluation.spatial_experiments.run_all``.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from conrad.evaluation.spatial_experiments import (
    common,
    e001_coverage,
    e002_counterfactual,
    e003_pose,
    e004_modality,
)

RunFn = Callable[[Mapping[str, Any], Sequence[int], str | Path], dict[str, Any]]
EXPERIMENTS: dict[str, tuple[str, RunFn]] = {
    "2S-E001": ("configs/eval/2s_e001.yaml", e001_coverage.run),
    "2S-E002": ("configs/eval/2s_e002.yaml", e002_counterfactual.run),
    "2S-E003": ("configs/eval/2s_e003.yaml", e003_pose.run),
    "2S-E004": ("configs/eval/2s_e004.yaml", e004_modality.run),
}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--out", default="artifacts/experiments/spatial")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args(argv)
    for key in args.only:
        path, fn = EXPERIMENTS[key]
        cfg = common.load_config(path)
        seeds = args.seeds or cfg.get("seeds", list(common.DEFAULT_SEEDS))
        fn(cfg, seeds, args.out)
        print(f"{key}: wrote {Path(args.out) / (key + '.json')}")


if __name__ == "__main__":
    main()
