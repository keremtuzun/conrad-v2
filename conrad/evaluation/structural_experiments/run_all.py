"""Run 2T-E001..E004: ``python -m conrad.evaluation.structural_experiments.run_all [out_dir] [ids...]``."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from conrad.evaluation.structural_experiments import e001_direct, e002_persistent, e003_tcdp, e004_temporal
from conrad.evaluation.structural_experiments.common import load_config

DEFAULT_SEEDS = (2026201, 2026202, 2026203)
Runner = Callable[[Mapping[str, Any], Sequence[int], str | Path], dict[str, Any]]
EXPERIMENTS: dict[str, tuple[Runner, str]] = {
    "2T-E001": (e001_direct.run, "configs/eval/2t_e001.yaml"),
    "2T-E002": (e002_persistent.run, "configs/eval/2t_e002.yaml"),
    "2T-E003": (e003_tcdp.run, "configs/eval/2t_e003.yaml"),
    "2T-E004": (e004_temporal.run, "configs/eval/2t_e004.yaml"),
}


def main(argv: Sequence[str]) -> None:
    out = Path(argv[0]) if argv else Path("artifacts/experiments/structural")
    wanted = list(argv[1:]) or list(EXPERIMENTS)
    for eid in wanted:
        runner, cfg_path = EXPERIMENTS[eid]
        cfg = load_config(cfg_path)
        result = runner(cfg, list(cfg.get("seeds", DEFAULT_SEEDS)), out)
        print(f"{eid}: {result['wall_time_s']:.1f}s -> {out / (eid + '.json')}")


if __name__ == "__main__":
    main(sys.argv[1:])
