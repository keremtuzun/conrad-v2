"""2T-E003-R2: TCDP relational benefit vs contamination under the REALISTIC Twin2T sensor, with paired CIs.

Episodes and arms are exactly 2T-E003's (``e003_tcdp.run_seed``). Added: per-seed pooled statistics and
percentile-bootstrap CIs over seeds (the paired unit), and a gate-ready verdict declared in the config:
  RB_q  = mean over episode kinds of (MAE_INDEPENDENT - MAE_TCDP) on reachable hidden components;
  RC_q  = pooled P(claims degraded | hidden, healthy, degraded observed neighbour) per arm;
  benefit       iff some quantity has CI(RB_q) above 0 and none has CI(RB_q) below 0;
  lower_than_generic iff every quantity with an RC pool has CI(RC_GENERIC - RC_TCDP) above 0.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from conrad.evaluation.partitions import purpose_scope
from conrad.evaluation.structural_experiments.common import (
    QUANTITIES,
    checked_seeds,
    experiment_id,
    paired_bootstrap,
    write_result,
)
from conrad.evaluation.structural_experiments.e003_tcdp import run_seed

KINDS = ("coupled", "misleading", "natural")


def pooled(out: Mapping[str, Any], kinds: Sequence[str]) -> dict[str, float]:
    res: dict[str, float] = {}
    for q in QUANTITIES:
        for sub in ("reachable_hidden_mae_mm", "hidden_mae_mm"):
            rb = [out.get(f"{k}.{q}.RB_tcdp.{sub}") for k in kinds]
            rbg = [out.get(f"{k}.{q}.RB_generic.{sub}") for k in kinds]
            if all(x is not None for x in rb):
                res[f"{q}.RB_tcdp.{sub}"] = float(np.mean([float(x) for x in rb if x is not None]))
            if all(x is not None for x in rbg):
                res[f"{q}.RB_generic.{sub}"] = float(np.mean([float(x) for x in rbg if x is not None]))
        for arm in ("TCDP", "GENERIC_RELATIONAL", "INDEPENDENT_COMPONENT"):
            hits = sum(out.get(f"{k}.{arm}.{q}.rc_hits", 0.0) for k in kinds)
            pool = sum(out.get(f"{k}.{arm}.{q}.rc_pool", 0.0) for k in kinds)
            if pool > 0:
                res[f"{q}.{arm}.rc"] = hits / pool
                res[f"{q}.{arm}.rc_pool"] = pool
    return res


def run(config: Mapping[str, Any], seeds: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    seeds = checked_seeds(config, [seeds] if isinstance(seeds, int) else list(seeds))
    kinds = list(config.get("kinds", KINDS))
    with (
        purpose_scope(str(config.get("purpose", "design"))),
        tempfile.TemporaryDirectory(prefix="m2t_e003r2_", ignore_cleanup_errors=True) as tmp,
    ):
        per_seed: dict[int, Any] = {}
        for s in seeds:
            out = run_seed(config, s, Path(tmp))
            out["pooled"] = pooled(out, kinds)
            per_seed[s] = out
    verdicts: dict[str, Any] = {"partition": config.get("partition")}
    benefit_pos, benefit_neg, rc_ok, rc_any = False, False, True, False
    for q in QUANTITIES:
        rb = paired_bootstrap(
            [per_seed[s]["pooled"].get(f"{q}.RB_tcdp.reachable_hidden_mae_mm") for s in seeds]
        )
        verdicts[f"{q}.RB_tcdp_ci"] = rb
        verdicts[f"{q}.RB_generic_ci"] = paired_bootstrap(
            [per_seed[s]["pooled"].get(f"{q}.RB_generic.reachable_hidden_mae_mm") for s in seeds]
        )
        for kind in kinds:
            verdicts[f"{kind}.{q}.RB_tcdp_ci"] = paired_bootstrap(
                [per_seed[s].get(f"{kind}.{q}.RB_tcdp.reachable_hidden_mae_mm") for s in seeds]
            )
        if rb is not None:
            benefit_pos |= rb["ci_low"] > 0.0
            benefit_neg |= rb["ci_high"] < 0.0
        diffs = []
        for s in seeds:
            p = per_seed[s]["pooled"]
            g, t = p.get(f"{q}.GENERIC_RELATIONAL.rc"), p.get(f"{q}.TCDP.rc")
            if g is not None and t is not None:
                diffs.append(g - t)
        rc = paired_bootstrap(diffs)
        verdicts[f"{q}.rc_generic_minus_tcdp_ci"] = rc
        for arm in ("TCDP", "GENERIC_RELATIONAL", "INDEPENDENT_COMPONENT"):
            hits = sum(
                per_seed[s]["pooled"].get(f"{q}.{arm}.rc", 0.0)
                * per_seed[s]["pooled"].get(f"{q}.{arm}.rc_pool", 0.0)
                for s in seeds
            )
            pool = sum(per_seed[s]["pooled"].get(f"{q}.{arm}.rc_pool", 0.0) for s in seeds)
            verdicts[f"{q}.{arm}.rc_pooled_all_seeds"] = hits / pool if pool else None
        if rc is not None:
            rc_any = True
            rc_ok &= rc["ci_low"] > 0.0
    verdicts["tcdp_benefit"] = benefit_pos and not benefit_neg
    verdicts["tcdp_contamination_lower_than_generic"] = rc_any and rc_ok
    verdicts["excessive_contamination_threshold"] = config.get("excessive_contamination_threshold", "OPEN")
    return write_result(
        experiment_id(config, "2T-E003-R2"), config, seeds, per_seed, out_dir, started, verdicts
    )
