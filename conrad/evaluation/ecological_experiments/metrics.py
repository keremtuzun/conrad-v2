"""2E metrics: field reconstruction + calibration, cover error + calibration, UEI. EVALUATION PLANE.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from conrad.domains.ecological import DAMAGE_CLAIM, Model2E
from conrad.schemas.base import ARCHITECTURE_ID
from conrad.schemas.belief import KnowledgeStatus
from conrad.twins.twin2e import Twin2E

Z95 = 1.959964


def gaussian_scores(err: np.ndarray, var: np.ndarray) -> dict[str, float]:
    var = np.maximum(var, 1e-12)
    z2 = err**2 / var
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "coverage95": float(np.mean(np.abs(err) <= Z95 * np.sqrt(var))),
        "mean_z2": float(np.mean(z2)),
        "nll": float(np.mean(0.5 * (np.log(2 * math.pi * var) + z2))),
        "n": float(err.size),
    }


def field_errors(
    model: Model2E, twin: Twin2E, name: str, cells: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    centers = model.fields.grid.centers if cells is None else model.fields.grid.centers[cells]
    truth = twin.fields.sample(name, centers)
    mean, var = model.fields.moments_at_time(name, twin.now().time_ns)
    m, v = (mean[0], var[0]) if cells is None else (mean[0, cells], var[0, cells])
    return m - truth, v


def cover_errors(model: Model2E, twin: Twin2E) -> tuple[np.ndarray, np.ndarray, list[float]]:
    errs, vars_, conds = [], [], []
    t = twin.now().time_ns
    for eid, e in twin.entities.items():
        if e.kind not in ("BIOFOULING_ON_STRUCTURE", "BENTHIC_PATCH"):
            continue
        b = model.entities.by_registry(eid)
        if b is None:
            continue
        mean, var = model.entities.cover_moments_at(b, t)
        errs.append(mean - e.cover)
        vars_.append(var)
        conds.append(e.condition)
    return np.asarray(errs), np.asarray(vars_), conds


def entity_channels(model: Model2E) -> dict[str, float]:
    ua, uo, n = [], [], 0
    for m in model.export_beliefs():
        if m.ecological is None or m.ecological.kind != "ENTITY" or m.world_entity_id is None:
            continue
        if not any(c.name == "cover_fraction" for c in m.state_summary):
            continue
        ua.append(m.uncertainty.aleatoric)
        uo.append(m.uncertainty.observational)
        n += 1
    return {"mean_UA": float(np.mean(ua)) if ua else 0.0, "mean_UO": float(np.mean(uo)) if uo else 1.0}


def unsupported_damage_claims(model: Model2E) -> dict[str, float]:
    """UEI numerator: damage claims that are not UNKNOWN (none are expected without damage evidence)."""
    total = confident = observed = 0
    for m in model.export_beliefs():
        for c in m.state_summary:
            if c.name != DAMAGE_CLAIM:
                continue
            total += 1
            confident += int(c.status is not KnowledgeStatus.UNKNOWN)
            observed += int(c.status is KnowledgeStatus.OBSERVED)
    return {
        "damage_claims": float(total),
        "non_unknown_damage_claims": float(confident),
        "observed_damage_claims": float(observed),
        "UEI": confident / total if total else 0.0,
    }


def aggregate(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    flat: dict[int, dict[str, float]] = {}

    def walk(prefix: str, v: Any, out: dict[str, float]) -> None:
        if isinstance(v, Mapping):
            for k, x in v.items():
                walk(f"{prefix}.{k}" if prefix else str(k), x, out)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[prefix] = float(v)

    for s, r in per_seed.items():
        flat[s] = {}
        walk("", r, flat[s])
    keys = sorted({k for f in flat.values() for k in f})
    out: dict[str, dict[str, float | None]] = {}
    for k in keys:
        vals = np.asarray([f[k] for f in flat.values() if k in f and np.isfinite(f[k])])
        out[k] = {
            "mean": float(vals.mean()) if vals.size else None,
            "std": float(vals.std(ddof=1)) if vals.size > 1 else None,
            "n": float(vals.size),
        }
    return out


def run_seeds(
    experiment_id: str,
    seed_run: Callable[[Mapping[str, Any], int], dict[str, Any]],
    config: Mapping[str, Any],
    seeds: Sequence[int],
    out_dir: str | Path,
    candidate: str,
    baselines: Sequence[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    per_seed = {int(s): seed_run(config, int(s)) for s in seeds}
    result: dict[str, Any] = {
        "experiment_id": experiment_id,
        "architecture_id": ARCHITECTURE_ID,
        "data": "SYNTHETIC_ONLY Twin2E small world (configs/sim/twin2e_test_small.yaml)",
        "candidate": candidate,
        "baselines": list(baselines),
        "seeds": [int(s) for s in seeds],
        "config": dict(config),
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "summary": aggregate(per_seed),
        "wall_time_s": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "device": "cpu",
            "platform": platform.platform(),
        },
        "claim_note": "Numbers are what this run produced; baseline wins are reported as they are.",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{experiment_id}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
