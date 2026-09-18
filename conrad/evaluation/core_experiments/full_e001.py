"""CORE-FULL-E001: full Model 2 Core loop (TBD -> association -> BUO -> RBP -> PMBL) on the sandbox.

Variants: full; no RBP; no uncertainty growth in TBD (process noise 0); baseline latest-only
(truth-associated last measurement, optimistic). Metrics are computed on EVERY live hidden entity at
EVERY step, including entities not observed in that step (predicted / inferred beliefs).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from conrad.evaluation.core_experiments.common import (
    core_config,
    run_experiment,
    scratch_dir,
    seed_everything,
)
from conrad.evaluation.core_experiments.pipeline_runner import RunLog, majority, root, run_pipeline
from conrad.evaluation.core_experiments.sandbox_world import SandboxConfig, SandboxStep, SandboxWorld

EXPERIMENT_ID = "CORE-FULL-E001"


def score(log: RunLog, steps: Sequence[SandboxStep], coupled_targets: set[int]) -> dict[str, float]:
    maj = majority(log)
    errs: list[float] = []
    vars_: list[float] = []
    coupled_errs: list[float] = []
    uc_flag: list[float] = []
    uc_other: list[float] = []
    for k, step in enumerate(steps):
        observed = {s for s in step.truth.source.values() if s is not None}
        flagged = {
            step.truth.source[e]
            for e, f in step.truth.flags.items()
            if "outlier" in f and step.truth.source[e] is not None
        }
        for bid, st in log.per_step_estimates[k].items():
            src = maj.get(root(log, bid), (None, 0.0))[0]
            if src is None or src not in step.truth.states:
                continue
            (uc_flag if src in flagged else uc_other).append(st.uc)
            for p, est in st.estimates.items():
                err = est.mean - step.truth.states[src][p]
                errs.append(err)
                vars_.append(max(est.variance, 1e-9))
                if p == "p1" and src in coupled_targets and src not in observed:
                    coupled_errs.append(err)
    e, v = np.asarray(errs), np.asarray(vars_)
    return {
        "state_rmse": float(np.sqrt(np.mean(e**2))),
        "nll": float(np.mean(0.5 * (np.log(2 * math.pi * v) + e**2 / v))),
        "coverage95": float(np.mean(np.abs(e) <= 1.96 * np.sqrt(v))),
        "confident_wrong_rate": float(np.mean((np.abs(e) > 0.15) & (np.sqrt(v) < 0.05))),
        "coupled_unobserved_p1_rmse": float(np.sqrt(np.mean(np.square(coupled_errs))))
        if coupled_errs
        else float("nan"),
        "uc_mean_steps_with_outlier_evidence": float(np.mean(uc_flag)) if uc_flag else float("nan"),
        "uc_mean_other_steps": float(np.mean(uc_other)) if uc_other else float("nan"),
        "relationships_added": float(log.relationships_added),
    }


def latest_only(steps: Sequence[SandboxStep], coupled_targets: set[int]) -> dict[str, float]:
    last: dict[int, dict[str, float]] = {}
    errs, coupled = [], []
    for step in steps:
        observed = set()
        for ev, _ in step.evidence:
            src = step.truth.source[ev.evidence_id]
            if src is not None:
                last[src] = dict(ev.measurements)
                observed.add(src)
        for src, meas in last.items():
            if src in step.truth.states:
                for p, y in meas.items():
                    errs.append(y - step.truth.states[src][p])
                    if p == "p1" and src in coupled_targets and src not in observed:
                        coupled.append(errs[-1])
    return {
        "state_rmse": float(np.sqrt(np.mean(np.square(errs)))),
        "coupled_unobserved_p1_rmse": float(np.sqrt(np.mean(np.square(coupled))))
        if coupled
        else float("nan"),
    }


def run_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    cfg = core_config(config)
    seed_everything(seed)
    world = SandboxWorld(SandboxConfig(**config.get("sandbox", {})), seed)
    steps = world.episode()
    coupled = world.coupled()
    targets = {t for pair in coupled for t in pair}
    q = float(config.get("process_noise_per_s", 1e-5))
    variants = {
        "full": {"process_noise_per_s": q, "use_rbp": True, "coupled": coupled},
        "no_rbp": {"process_noise_per_s": q, "use_rbp": False, "coupled": coupled},
        "no_tbd_uncertainty_growth": {"process_noise_per_s": 0.0, "use_rbp": True, "coupled": coupled},
    }
    out: dict[str, Any] = {}
    for name, kw in variants.items():
        with scratch_dir() as tmp:
            _, log = run_pipeline(cfg, steps, tmp, seed, **kw)  # type: ignore[arg-type]
            out[name] = score(log, steps, targets)
    out["latest_only_baseline"] = latest_only(steps, targets)
    return out


def run(config: Mapping[str, Any], seed: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    return run_experiment(
        EXPERIMENT_ID,
        run_seed,
        config,
        seed,
        out_dir,
        "full",
        ["no_rbp", "no_tbd_uncertainty_growth", "latest_only_baseline"],
    )
