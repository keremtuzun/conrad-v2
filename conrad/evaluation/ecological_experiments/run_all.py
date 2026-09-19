"""Run 2E-E001..E003 from their configs: python -m conrad.evaluation.ecological_experiments.run_all"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from conrad.evaluation.ecological_experiments import e001_field, e002_turbidity, e003_coupling

REPO = Path(__file__).resolve().parents[3]
EXPERIMENTS = {
    "2e_e001": e001_field.run,
    "2e_e002": e002_turbidity.run,
    "2e_e003": e003_coupling.run,
    # R2: repaired field model, production arm, FINAL partition (docs/audits/MODEL2E_REPAIR.md)
    "2e_e001_r2": e001_field.run,
    "2e_e002_r2": e002_turbidity.run,
    "2e_e003_r2": e003_coupling.run,
    # R3: iteration 2 (observability context + change detection), fresh FINAL-2 partition
    "2e_e001_r3": e001_field.run,
    "2e_e002_r3": e002_turbidity.run,
    "2e_e003_r3": e003_coupling.run,
    # R4: iteration 3 (EB-calibrated temperature depth-trend prior), fresh FINAL-3 partition
    "2e_e001_r4": e001_field.run,
    "2e_e002_r4": e002_turbidity.run,
    "2e_e003_r4": e003_coupling.run,
}


def main(argv: list[str]) -> int:
    out = Path(argv[0]) if argv else REPO / "artifacts" / "experiments" / "ecological"
    only = set(argv[1:])
    for name, fn in EXPERIMENTS.items():
        if only and name not in only:
            continue
        cfg = yaml.safe_load((REPO / "configs" / "eval" / f"{name}.yaml").read_text(encoding="utf-8"))
        res = fn(cfg, cfg["seeds"], out)
        print(f"{res['experiment_id']}: {res['wall_time_s']:.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
