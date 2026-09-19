"""2E-E002: entity cover estimation under turbidity (U_A / U_O behaviour).

Background turbidity is swept; a survey robot hovers over each sessile asset in turn (E0 cover
surveys, estimated pose with noise) and three turbidity moorings run unsynchronised. Variants:
``cefd`` (survey noise from the turbidity BELIEF), ``uncoupled`` (turbidity-blind noise, fields
kept) and ``entity_only`` (no field beliefs at all). Scored against Twin2E cover truth.

SYNTHETIC_ONLY.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from conrad.evaluation.ecological_experiments.harness import (
    Episode,
    env,
    make_models,
    make_scenario,
    make_twin,
    model_config,
    run_steps,
    survey_waypoints,
    twin_config,
)
from conrad.evaluation.ecological_experiments.metrics import (
    cover_errors,
    entity_channels,
    experiment_id,
    gaussian_scores,
    run_seeds,
)

EXPERIMENT_ID = "2E-E002"
VARIANTS = ["cefd", "uncoupled", "entity_only"]


def _level_run(config: Mapping[str, Any], seed: int, turbidity: float) -> dict[str, Any]:
    tcfg = twin_config(config)
    rng = np.random.default_rng(seed + int(turbidity * 1000))
    scenario = make_scenario(seed, tcfg, config, environment=env(turbidity=(turbidity, "NTU")))
    twin = make_twin(scenario, tcfg, seed)
    lg = tcfg.local_grid
    lo = np.asarray(lg.origin_m)
    hi = lo + np.asarray(lg.spacing_m) * np.asarray(lg.shape)
    ep = Episode(twin, rng, seed, pose_sigma_m=float(config.get("pose_sigma_m", 0.3)))
    moorings = [
        ep.mooring(i, lo + (hi - lo) * np.array([*rng.uniform(0.1, 0.9, 2), 0.1]), ("turbidity",), 2, i % 2)
        for i in range(int(config.get("n_moorings", 3)))
    ]
    cam = ep.survey_sensor(float(config.get("footprint_radius_m", 6.0)))
    waypoints = survey_waypoints(scenario, float(config.get("survey_altitude_m", 2.5)))
    variants = list(config.get("variants", VARIANTS))
    models = make_models(variants, seed, scenario, model_config(config))
    eval_every = int(config.get("eval_every_steps", 12))
    acc: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {v: ([], []) for v in variants}
    counts = {"surveys": 0, "survey_evidence": 0}

    def step(i: int) -> None:
        evs = [
            e
            for s in moorings
            if (i + s.phase) % s.period_steps == 0
            for e in ep.sense(s.spec, s.position, True)
        ]
        wp = waypoints[i % len(waypoints)]
        hits = ep.sense(cam, wp, pose_noise=True)
        counts["surveys"] += 1
        counts["survey_evidence"] += len(hits)
        for v, m in models.items():
            m.ingest([*evs, *hits])
            m.update_beliefs(twin.now())
            if i >= eval_every and i % eval_every == 0:
                err, var, _ = cover_errors(m, twin)
                acc[v][0].append(err)
                acc[v][1].append(var)

    run_steps(twin, int(config.get("n_steps", 96)), float(config.get("dt_s", 900.0)), step)
    out: dict[str, Any] = {"detections_per_survey": counts["survey_evidence"] / max(1, counts["surveys"])}
    for v, m in models.items():
        out[v] = {
            "cover": gaussian_scores(np.concatenate(acc[v][0]), np.concatenate(acc[v][1])),
            **entity_channels(m),
            "ambiguous_association": float(m.stats.get("ambiguous_association", 0)),
        }
    return out


def _seed_run(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    levels = [float(x) for x in config.get("turbidity_levels_ntu", [1.0, 4.0, 10.0, 25.0])]
    return {f"turbidity_{lvl:g}": _level_run(config, seed, lvl) for lvl in levels}


def run(config: Mapping[str, Any], seeds: Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    variants = list(config.get("variants", VARIANTS))
    return run_seeds(
        experiment_id(config, EXPERIMENT_ID),
        _seed_run,
        config,
        seeds,
        out_dir,
        candidate=variants[0],
        baselines=variants[1:],
    )
