"""2T-E001-R2: direct structural inference under the REALISTIC Twin2T sensor (docs/audits/MODEL2T_REPAIR.md).

Same episodes as 2T-E001 (TCDP disabled, tier-3 scenarios, 60-day steps, 80 % coverage), scored on the
components inspected at the current step. Arms:
  MODEL2T_DIRECT        repaired Model2T (SENSOR_CHARACTERISED measurement model), analytic core, no TCDP;
  MODEL2T_ABSOLUTE      pre-repair fixed absolute-sigma update (ablation);
  LATEST_OBSERVATION    B1 hold-last reading;  LATEST_OBSERVATION_DEBIASED  B1 / declared sizing factor;
  SINGLE_FRAME          B0;  GRU_TEMPORAL  B2 (fitted on separate training episodes).
Model2T is scored on its posterior mean whether or not the value is claimed: a crack that was never detected
is UNKNOWN as a claim but its censored estimate is still the model's answer ("latent"); claimed-only scores
are reported next to it. The paired unit is the seed (all its episodes and levels); CIs are percentile
bootstrap over seeds.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from conrad.domains.technical import CRACK_LENGTH, PropagationMode, model2t_config_from_dict
from conrad.domains.technical.baselines import (
    EngineEstimator,
    LatestObservation,
    LatestObservationDebiased,
    SingleFrame,
    StructuralEstimator,
)
from conrad.evaluation.partitions import purpose_scope
from conrad.evaluation.structural_experiments.common import (
    DAY,
    QUANTITIES,
    checked_seeds,
    experiment_id,
    gaussian_scores,
    make_world,
    paired_bootstrap,
    run_episode,
    write_result,
)
from conrad.evaluation.structural_experiments.e002_persistent import train_gru
from conrad.evaluation.structural_experiments.scenarios import coverage_schedule, make_scenario

MODEL = "MODEL2T_DIRECT"
BASELINES = ("LATEST_OBSERVATION", "LATEST_OBSERVATION_DEBIASED", "SINGLE_FRAME", "GRU_TEMPORAL")
CRACK_REGIMES = ((0.0, 3e-3), (3e-3, 10e-3), (10e-3, 50e-3), (50e-3, 0.999), (0.999, 10.0))


def _regime(x: float) -> str:
    for lo, hi in CRACK_REGIMES:
        if lo <= x < hi:
            return f"{lo * 1e3:g}-{hi * 1e3:g}mm"
    return "other"


def run_seed(
    config: Mapping[str, Any], seed: int, tmp: Path, gru: StructuralEstimator | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for level in config.get("degradation_levels", [0.0, 0.4]):
        acc: dict[str, list[float]] = {}
        for ep in range(int(config.get("episodes_per_seed", 3))):
            eseed = seed * 100 + ep
            world = make_world(eseed, make_scenario(eseed, config.get("scenario", {})), tmp)
            base = model2t_config_from_dict(dict(config.get("model2t", {})))
            model = EngineEstimator(PropagationMode.NONE, base, seed=eseed)
            model.name = MODEL
            legacy_cfg = replace(base, direct=replace(base.direct, measurement_model="ABSOLUTE_GAUSSIAN"))
            legacy = EngineEstimator(PropagationMode.NONE, legacy_cfg, seed=eseed)
            legacy.name = "MODEL2T_ABSOLUTE"
            ests: list[StructuralEstimator] = [
                model,
                legacy,
                LatestObservation(),
                LatestObservationDebiased(),
                SingleFrame(),
            ]
            if gru is not None:
                ests.append(gru)
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
                for rid in rec["observed"]:
                    for q in QUANTITIES:
                        truth = rec["truth"][rid][q]
                        if truth is None:
                            continue
                        for name, per in rec["est"].items():
                            e = per[rid][q]
                            if name in rec["latent"]:
                                claimed = e is not None
                                e = rec["latent"][name][rid][q]
                                acc.setdefault(f"{name}.{q}.claim_rate", []).append(float(claimed))
                                if claimed and e is not None:
                                    acc.setdefault(f"{name}.{q}.claimed_mae_mm", []).append(
                                        gaussian_scores(e[0], e[1], truth)[0]
                                    )
                            if e is None:
                                continue
                            err, nll, cov = gaussian_scores(e[0], e[1], truth)
                            for metric, v in (("mae_mm", err), ("nll", nll), ("cov95", cov)):
                                acc.setdefault(f"{name}.{q}.{metric}", []).append(v)
                            if q == CRACK_LENGTH:
                                acc.setdefault(f"{name}.{q}.regime.{_regime(truth)}.mae_mm", []).append(err)
        res: dict[str, Any] = {k: float(np.mean(v)) for k, v in acc.items()}
        res.update({k.replace("mae_mm", "n"): float(len(v)) for k, v in acc.items() if "regime" in k})
        out[f"level_{level}"] = res
    return out


def _paired(per_seed: Mapping[int, Any], seeds: Sequence[int], key_a: str, key_b: str, lv: str) -> Any:
    diffs = []
    for s in seeds:
        a, b = per_seed[s][lv].get(key_a), per_seed[s][lv].get(key_b)
        if a is not None and b is not None:
            diffs.append(a - b)
    return paired_bootstrap(diffs, seed=len(diffs))


def run(config: Mapping[str, Any], seeds: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    seeds = checked_seeds(config, [seeds] if isinstance(seeds, int) else list(seeds))
    purpose = str(config.get("purpose", "design"))
    with (
        purpose_scope(purpose),
        tempfile.TemporaryDirectory(prefix="m2t_e001r2_", ignore_cleanup_errors=True) as tmp,
    ):
        gru = None
        if config.get("gru", True):
            # B2 is fitted ONCE, on episodes of a declared development seed (never an evaluated seed).
            gcfg = dict(config.get("gru_training", {}))
            train_seed = int(gcfg.get("train_seed", 5100039))
            if train_seed in seeds:
                raise ValueError("the GRU training seed must not be an evaluated seed")
            gru = train_gru(gcfg, train_seed, Path(tmp))
        per_seed = {s: run_seed(config, s, Path(tmp), gru) for s in seeds}
    levels = [f"level_{lv}" for lv in config.get("degradation_levels", [0.0, 0.4])]
    band = [float(x) for x in config.get("coverage_band", [0.90, 0.99])]
    verdicts: dict[str, Any] = {"partition": config.get("partition"), "coverage_band": band}
    for lv in levels:
        for q in QUANTITIES:
            for base in (*BASELINES, "MODEL2T_ABSOLUTE"):
                ci = _paired(per_seed, seeds, f"{base}.{q}.mae_mm", f"{MODEL}.{q}.mae_mm", lv)
                verdicts[f"{lv}.{q}.mae_gain_vs_{base}"] = ci
            cov = [per_seed[s][lv].get(f"{MODEL}.{q}.cov95") for s in seeds]
            pooled = (
                float(np.mean([c for c in cov if c is not None])) if any(c is not None for c in cov) else None
            )
            verdicts[f"{lv}.{q}.model_cov95"] = pooled
            verdicts[f"{lv}.{q}.model_cov95_in_band"] = pooled is not None and band[0] <= pooled <= band[1]
            gain = verdicts[f"{lv}.{q}.mae_gain_vs_LATEST_OBSERVATION"]
            verdicts[f"{lv}.{q}.beats_latest_ci_above_0"] = gain is not None and gain["ci_low"] > 0.0
    verdicts["direct_inference_works"] = all(
        verdicts[f"{lv}.{q}.beats_latest_ci_above_0"] and verdicts[f"{lv}.{q}.model_cov95_in_band"]
        for lv in levels
        for q in QUANTITIES
    )
    return write_result(
        experiment_id(config, "2T-E001-R2"), config, seeds, per_seed, out_dir, started, verdicts
    )
