"""2T-E001: direct structural inference and calibration vs LATEST_OBSERVATION / SINGLE_FRAME (2T-D1, T2).

TCDP disabled (T2: "Disable TCDP"). Metrics on components inspected at the current step: MAE (mm), Gaussian
NLL (mm units), 95% interval coverage; and U_A response to sensor degradation.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from conrad.domains.technical import PropagationMode
from conrad.domains.technical.baselines import EngineEstimator, LatestObservation, SingleFrame
from conrad.evaluation.structural_experiments.common import (
    DAY,
    QUANTITIES,
    gaussian_scores,
    make_world,
    run_episode,
    summarize,
    write_result,
)
from conrad.evaluation.structural_experiments.scenarios import coverage_schedule, make_scenario

EXPERIMENT_ID = "2T-E001"


def run_seed(config: Mapping[str, Any], seed: int, tmp: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for level in config.get("degradation_levels", [0.0, 0.4]):
        acc: dict[str, list[float]] = {}
        ua_obs: list[float] = []
        for ep in range(int(config.get("episodes_per_seed", 3))):
            eseed = seed * 100 + ep
            world = make_world(eseed, make_scenario(eseed, config.get("scenario", {})), tmp)
            model = EngineEstimator(PropagationMode.NONE, seed=eseed)
            model.name = "MODEL2T_DIRECT"
            ests = [model, LatestObservation(), SingleFrame()]
            vis = coverage_schedule(world.component_ids, float(config.get("coverage", 0.8)), eseed)
            recs = run_episode(
                world,
                ests,
                steps=int(config.get("steps", 16)),
                dt_s=float(config.get("dt_days", 60)) * DAY,
                visible_at=vis,
                degradation_level=float(level),
            )
            for rec in recs:
                for rid, tr in rec["truth"].items():
                    for q in QUANTITIES:
                        if tr[q] is not None:
                            for name, per in rec["est"].items():
                                acc.setdefault(f"{name}.{q}.claim_rate_all", []).append(
                                    float(per[rid][q] is not None)
                                )
                for rid in rec["observed"]:
                    assert model.engine is not None
                    for q in QUANTITIES:
                        truth = rec["truth"][rid][q]
                        if truth is None:
                            continue
                        for name, per in rec["est"].items():
                            e = per[rid][q]
                            if e is None:
                                continue
                            err, nll, cov = gaussian_scores(e[0], e[1], truth)
                            for metric, v in (("mae_mm", err), ("nll", nll), ("cov95", cov)):
                                acc.setdefault(f"{name}.{q}.{metric}", []).append(v)
            assert model.engine is not None
            ua_obs += [
                b.uncertainty().aleatoric for b in model.engine.beliefs.values() if b.direct_support > 0
            ]
        res = summarize(acc)
        res["MODEL2T_DIRECT.mean_ua_observed"] = sum(ua_obs) / len(ua_obs) if ua_obs else None
        out[f"level_{level}"] = res
    return out


def run(config: Mapping[str, Any], seeds: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    seeds = [seeds] if isinstance(seeds, int) else list(seeds)
    with tempfile.TemporaryDirectory(prefix="m2t_e001_", ignore_cleanup_errors=True) as tmp:
        per_seed = {s: run_seed(config, s, Path(tmp)) for s in seeds}
    verdicts: dict[str, Any] = {}
    for level_key in per_seed[seeds[0]]:
        for q in QUANTITIES:
            m = [per_seed[s][level_key].get(f"MODEL2T_DIRECT.{q}.mae_mm") for s in seeds]
            b = [per_seed[s][level_key].get(f"LATEST_OBSERVATION.{q}.mae_mm") for s in seeds]
            verdicts[f"{level_key}.{q}.model_beats_latest_all_seeds"] = all(
                x is not None and y is not None and x < y for x, y in zip(m, b, strict=True)
            )
    lv = list(per_seed[seeds[0]])
    if len(lv) >= 2:
        ua = [
            (
                per_seed[s][lv[0]]["MODEL2T_DIRECT.mean_ua_observed"],
                per_seed[s][lv[-1]]["MODEL2T_DIRECT.mean_ua_observed"],
            )
            for s in seeds
        ]
        verdicts["ua_rises_with_sensor_degradation_all_seeds"] = all(a < b for a, b in ua)
    return write_result(EXPERIMENT_ID, config, seeds, per_seed, out_dir, started, verdicts)
