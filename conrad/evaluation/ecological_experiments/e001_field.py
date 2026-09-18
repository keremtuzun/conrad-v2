"""2E-E001: field reconstruction error + calibration vs field-only / static-field, by sensor count.

One Twin2E run per seed with the largest mooring set; each (variant, sensor count) model sees only
the first k moorings. Moorings are unsynchronised (different periods and phases). A localised
temperature anomaly and turbidity spike give the fields spatial structure. Scored on HIDDEN belief
cells (cells containing no sensor of that model) against Twin2E truth at the cell centres.

SYNTHETIC_ONLY.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from conrad.evaluation.ecological_experiments.harness import (
    Episode,
    make_models,
    make_scenario,
    make_twin,
    model_config,
    run_steps,
    twin_config,
)
from conrad.evaluation.ecological_experiments.metrics import field_errors, gaussian_scores, run_seeds

EXPERIMENT_ID = "2E-E001"
FIELDS = ("temperature", "turbidity")


def _seed_run(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    tcfg = twin_config(config)
    rng = np.random.default_rng(seed)
    lg = tcfg.local_grid
    lo = np.asarray(lg.origin_m)
    hi = lo + np.asarray(lg.spacing_m) * np.asarray(lg.shape)
    centre_t = (lo + (hi - lo) * rng.uniform(0.2, 0.8, 3)).tolist()
    centre_u = (lo + (hi - lo) * rng.uniform(0.2, 0.8, 3)).tolist()
    events = (
        (
            3600.0,
            "TEMPERATURE_ANOMALY",
            None,
            {
                "delta_c": float(config.get("anomaly_c", 3.0)),
                "duration_s": 86400.0,
                "units": "degC",
                "center_m": centre_t,
                "radius_m": float(config.get("event_radius_m", 25.0)),
            },
        ),
        (
            3600.0,
            "TURBIDITY_SPIKE",
            None,
            {
                "delta_ntu": float(config.get("spike_ntu", 8.0)),
                "units": "NTU",
                "center_m": centre_u,
                "radius_m": float(config.get("event_radius_m", 25.0)),
            },
        ),
    )
    scenario = make_scenario(seed, tcfg, config, events=events)
    twin = make_twin(scenario, tcfg, seed)
    counts = [int(c) for c in config.get("sensor_counts", [1, 2, 4, 8])]
    variants = list(config.get("variants", ["cefd", "field_only", "static_field"]))
    ep = Episode(twin, rng, seed, pose_sigma_m=float(config.get("pose_sigma_m", 0.3)))
    sensors = [
        ep.mooring(
            i,
            lo + (hi - lo) * rng.uniform(0.05, 0.95, 3),
            FIELDS,
            int(rng.integers(1, 4)),
            int(rng.integers(0, 3)),
        )
        for i in range(max(counts))
    ]
    models = {
        (v, k): m
        for k in counts
        for v, m in make_models(variants, seed * 100 + k, scenario, model_config(config)).items()
    }
    grid = next(iter(models.values())).fields.grid
    hidden = {k: np.ones(grid.n_cells, dtype=bool) for k in counts}
    for k in counts:
        for s in sensors[:k]:
            hidden[k][grid.cell_of(s.position)] = False
    eval_every = int(config.get("eval_every_steps", 12))
    acc: dict[tuple[str, int, str], tuple[list[np.ndarray], list[np.ndarray]]] = {}

    def step(i: int) -> None:
        now = twin.now()
        per_sensor = {
            s.index: ep.sense(s.spec, s.position, pose_noise=True)
            for s in sensors
            if (i + s.phase) % s.period_steps == 0
        }
        for (v, k), m in models.items():
            evs = [e for idx, lst in per_sensor.items() if idx < k for e in lst]
            m.ingest(evs)
            m.update_beliefs(now)
            if i >= eval_every and i % eval_every == 0:
                for name in FIELDS:
                    err, var = field_errors(m, twin, name, hidden[k])
                    a = acc.setdefault((v, k, name), ([], []))
                    a[0].append(err)
                    a[1].append(var)

    run_steps(twin, int(config.get("n_steps", 96)), float(config.get("dt_s", 900.0)), step)
    out: dict[str, Any] = {}
    for (v, k, name), (errs, vars_) in sorted(acc.items()):
        out.setdefault(v, {}).setdefault(f"k{k}", {})[name] = gaussian_scores(
            np.concatenate(errs), np.concatenate(vars_)
        )
    for v in variants:
        for k in counts:
            out[v][f"k{k}"]["late_evidence"] = float(models[(v, k)].stats.get("late_evidence", 0))
    return out


def run(config: Mapping[str, Any], seeds: Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    return run_seeds(
        EXPERIMENT_ID,
        _seed_run,
        config,
        seeds,
        out_dir,
        candidate="cefd",
        baselines=list(config.get("variants", ["cefd", "field_only", "static_field"]))[1:],
    )
