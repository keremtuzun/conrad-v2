"""Run the resumable I5 semantic-repair grid on declared DEVELOPMENT seeds only."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from conrad.evaluation.decision_experiments.m1_action_integrated import run
from conrad.settings import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "m1_action_e006_development.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    requested = Path(args.config)
    config_path = REPO_ROOT / requested if not requested.is_absolute() else requested
    config = dict(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    out = REPO_ROOT / "artifacts" / "experiments" / str(config["experiment_id"])
    result = run(config, [int(seed) for seed in config["seeds"]], out)
    print(yaml.safe_dump(result["verdicts"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
