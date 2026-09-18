"""Runnable CPU experiments for Model 1 / MCBR / BAAC on SYNTHETIC_ONLY belief-level fixtures.

``python -m conrad.evaluation.decision_experiments [out_dir]`` runs M1-UIR-E001, ACTIVE-MCBR-E001 and
COM-BAAC-E001 with the configs in configs/eval and writes one JSON per experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from conrad.evaluation.decision_experiments import active_mcbr, com_baac, m1_uir

EXPERIMENTS = {
    "M1-UIR-E001": (m1_uir.run, "configs/eval/m1_uir_e001.yaml"),
    "ACTIVE-MCBR-E001": (active_mcbr.run, "configs/eval/active_mcbr_e001.yaml"),
    "COM-BAAC-E001": (com_baac.run, "configs/eval/com_baac_e001.yaml"),
}

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch16 completion gates",
        "ch17 Core experiments",
        "ch18 Critical experiments",
        "ch19",
    ],
    "configuration_keys": ["configs/eval/m1_*.yaml", "configs/eval/active_*.yaml", "configs/eval/com_*.yaml"],
    "assumptions": ["all fixtures, worlds and channels are SYNTHETIC_ONLY"],
    "baselines": ["see conrad.decision / conrad.active / conrad.communication metadata"],
    "acceptance_tests": ["artifacts/experiments/decision/*.json"],
    "claim_status": "EVALUATED",
}


def run_all(out_dir: str | Path, root: str | Path = ".") -> dict[str, Any]:
    results = {}
    for name, (fn, cfg_path) in EXPERIMENTS.items():
        cfg = yaml.safe_load((Path(root) / cfg_path).read_text(encoding="utf-8"))
        seeds = [int(s) for s in cfg.pop("seeds")]
        results[name] = fn(cfg, seeds, out_dir)
    return results
