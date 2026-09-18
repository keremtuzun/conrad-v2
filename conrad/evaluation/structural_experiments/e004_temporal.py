"""2T-E004: temporal benefit under sparse inspections (2T-D3, T4; ch10 TB = Error_latest - Error_persistent).

Full-coverage inspections only on the configured days (default 30, 180, 360, 720, 1080, 1440, 1800);
between them every arm must predict. Arms: MODEL2T_TEMPORAL (analytic core, engineering-prior rates),
NO_RATE_PRIOR (ablation: same filter, zero prior rate), LATEST_OBSERVATION (hold-last).
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from conrad.domains.technical import Model2TConfig, PropagationMode
from conrad.domains.technical.baselines import EngineEstimator, LatestObservation
from conrad.evaluation.structural_experiments.common import (
    DAY,
    QUANTITIES,
    gaussian_scores,
    make_world,
    run_episode,
    write_result,
)
from conrad.evaluation.structural_experiments.scenarios import make_scenario

EXPERIMENT_ID = "2T-E004"


def _schedule(comps: list[UUID], inspect_steps: set[int]) -> Callable[[int], list[UUID]]:
    def visible(k: int) -> list[UUID]:
        return comps if k in inspect_steps else []

    return visible


def run_seed(config: Mapping[str, Any], seed: int, tmp: Path) -> dict[str, Any]:
    tick = float(config.get("tick_days", 30))
    days = [float(d) for d in config.get("inspection_days", [30, 180, 360, 720, 1080, 1440, 1800])]
    steps = int(config.get("horizon_days", 1800) // tick)
    inspect_steps = {round(d / tick) for d in days}
    before_next = {s - 1 for s in inspect_steps if s - 1 not in inspect_steps}
    acc: dict[str, list[float]] = {}
    for spec in config.get("scenarios", [{"kind": "small", "n_segments": 4}, {"kind": "tier", "tier": 3}]):
        for ep in range(int(config.get("episodes_per_scenario", 2))):
            eseed = seed * 100 + ep
            world = make_world(eseed, make_scenario(eseed, spec), tmp)
            base = Model2TConfig()
            ablate = replace(
                base, dynamics=replace(base.dynamics, corrosion_rate_m_per_yr=0.0, crack_rate_m_per_yr=0.0)
            )
            model = EngineEstimator(PropagationMode.NONE, base, seed=eseed)
            model.name = "MODEL2T_TEMPORAL"
            no_rate = EngineEstimator(PropagationMode.NONE, ablate, seed=eseed)
            no_rate.name = "NO_RATE_PRIOR"
            arms = [model, no_rate, LatestObservation()]
            comps = list(world.component_ids)
            recs = run_episode(
                world,
                arms,
                steps=steps,
                dt_s=tick * DAY,
                visible_at=_schedule(comps, inspect_steps),
            )
            first = min(inspect_steps)
            for rec in recs:
                k = rec["step"]
                if k in inspect_steps or k < first:
                    continue
                subsets = ["between"] + (["before_next"] if k in before_next else [])
                for rid, tr in rec["truth"].items():
                    for q in QUANTITIES:
                        if tr[q] is None:
                            continue
                        for a in arms:
                            e = rec["est"][a.name][rid][q]
                            if e is None:
                                continue
                            err, nll, cov = gaussian_scores(e[0], e[1], tr[q])
                            for sub in subsets:
                                acc.setdefault(f"{sub}.{a.name}.{q}.mae_mm", []).append(err)
                                acc.setdefault(f"{sub}.{a.name}.{q}.cov95", []).append(cov)
                                acc.setdefault(f"{sub}.{a.name}.{q}.nll", []).append(nll)
    out: dict[str, Any] = {k: float(np.mean(v)) for k, v in acc.items()}
    out.update(
        {
            k.replace("mae_mm", "median_ae_mm"): float(np.median(v))
            for k, v in acc.items()
            if k.endswith("mae_mm")
        }
    )
    for sub in ("between", "before_next"):
        for q in QUANTITIES:
            lat, mod = (
                out.get(f"{sub}.LATEST_OBSERVATION.{q}.mae_mm"),
                out.get(f"{sub}.MODEL2T_TEMPORAL.{q}.mae_mm"),
            )
            nr = out.get(f"{sub}.NO_RATE_PRIOR.{q}.mae_mm")
            if lat is not None and mod is not None:
                out[f"{sub}.{q}.TB_mm"] = lat - mod
            if lat is not None and nr is not None:
                out[f"{sub}.{q}.TB_no_rate_prior_mm"] = lat - nr
    return out


def run(config: Mapping[str, Any], seeds: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    seeds = [seeds] if isinstance(seeds, int) else list(seeds)
    with tempfile.TemporaryDirectory(prefix="m2t_e004_", ignore_cleanup_errors=True) as tmp:
        per_seed = {s: run_seed(config, s, Path(tmp)) for s in seeds}
    verdicts = {
        f"{sub}.{q}.TB_positive_all_seeds": all(
            (per_seed[s].get(f"{sub}.{q}.TB_mm") or 0.0) > 0 for s in seeds
        )
        for sub in ("between", "before_next")
        for q in QUANTITIES
    }
    return write_result(EXPERIMENT_ID, config, seeds, per_seed, out_dir, started, verdicts)
