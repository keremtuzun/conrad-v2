"""Shared experiment plumbing: config loading, seeding, multi-seed aggregation, JSON results."""

from __future__ import annotations

import json
import platform
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from conrad.core.config import CoreConfig, tiny_config
from conrad.persistence.db import make_engine, migrate
from conrad.persistence.repository import Repository
from conrad.schemas.base import ARCHITECTURE_ID

DEFAULT_SEEDS = (2026201, 2026202, 2026203)
SeedRun = Callable[[Mapping[str, Any], int], dict[str, Any]]


def load_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    return data


def core_config(config: Mapping[str, Any]) -> CoreConfig:
    """``model2_core`` block overrides the tiny CPU defaults used by these experiments."""
    return tiny_config(**dict(config.get("model2_core", {})))


def seed_everything(seed: int) -> np.random.Generator:
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def temp_repository(tmp_dir: str) -> Repository:
    path = Path(tmp_dir) / "core_experiment.db"
    migrate(path)
    return Repository(make_engine(path))


def _flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None:
        out[prefix] = float(value)


def aggregate(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    flats: dict[int, dict[str, float]] = {}
    for seed, metrics in per_seed.items():
        flats[seed] = {}
        _flatten("", metrics, flats[seed])
    keys = sorted({k for f in flats.values() for k in f})
    summary: dict[str, dict[str, float | None]] = {}
    for key in keys:
        vals = np.asarray([f[key] for f in flats.values() if key in f and np.isfinite(f[key])])
        summary[key] = {
            "mean": float(vals.mean()) if vals.size else None,
            "std": float(vals.std(ddof=1)) if vals.size > 1 else None,
            "n": float(vals.size),
        }
    return summary


def run_experiment(
    experiment_id: str,
    seed_run: SeedRun,
    config: Mapping[str, Any],
    seed: int | Sequence[int],
    out_dir: str | Path,
    candidate: str,
    baselines: Sequence[str],
) -> dict[str, Any]:
    seeds = [seed] if isinstance(seed, int) else list(seed)
    started = time.perf_counter()
    per_seed = {s: seed_run(config, s) for s in seeds}
    result: dict[str, Any] = {
        "experiment_id": experiment_id,
        "architecture_id": ARCHITECTURE_ID,
        "data": "SYNTHETIC_ONLY sandbox world (conrad.evaluation.core_experiments.sandbox_world)",
        "candidate": candidate,
        "baselines": list(baselines),
        "seeds": seeds,
        "config": dict(config),
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "summary": aggregate(per_seed),
        "wall_time_s": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "platform": platform.platform(),
        },
        "claim_note": "Numbers are what this run produced; no research win is assumed or implied.",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{experiment_id}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def scratch_dir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix="conrad_core_exp_", ignore_cleanup_errors=True)
