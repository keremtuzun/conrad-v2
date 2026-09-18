"""2E-E003: CEFD coupling benefit vs uncoupled, including counterfactual and confounded worlds.

Worlds per seed (all with a turbidity spike so observability coupling matters):
  base, hot counterfactual (make_counterfactual_pair on environment.temperature),
  HIGH_TEMPERATURE_STABLE_ECOLOGY and NORMAL_TEMPERATURE_DISTURBED_ECOLOGY (make_confounded_variant).
CB_entity = cover RMSE(uncoupled) - cover RMSE(cefd); CB_field = field RMSE(uncoupled) - RMSE(cefd).
Spurious-causality checks: confident stress (p > 0.9) on entities whose TRUE condition stays >= 0.9,
UEI (non-UNKNOWN damage claims), and how far coupling moves the cover MEAN vs uncoupled.

SYNTHETIC_ONLY.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from conrad.domains.ecological import Model2E
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
    field_errors,
    gaussian_scores,
    run_seeds,
    unsupported_damage_claims,
)
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Scenario
from conrad.twins.twin2e import make_confounded_variant, make_counterfactual_pair

EXPERIMENT_ID = "2E-E003"
VARIANTS = ["cefd", "uncoupled"]


def _worlds(config: Mapping[str, Any], seed: int) -> dict[str, Scenario]:
    tcfg = twin_config(config)
    lg = tcfg.local_grid
    centre = (np.asarray(lg.origin_m) + 0.5 * np.asarray(lg.spacing_m) * np.asarray(lg.shape)).tolist()
    spike = (
        float(config.get("spike_time_s", 21600.0)),
        "TURBIDITY_SPIKE",
        None,
        {
            "delta_ntu": float(config.get("spike_ntu", 15.0)),
            "units": "NTU",
            "center_m": centre,
            "radius_m": float(config.get("spike_radius_m", 30.0)),
        },
    )
    t_base = float(config.get("base_temperature_c", 22.0))
    base = make_scenario(seed, tcfg, config, environment=env(temperature=(t_base, "degC")), events=(spike,))
    rng = np.random.default_rng(seed + 7919)
    pair = make_counterfactual_pair(
        base, "environment.temperature", t_base + float(config.get("hot_delta_c", 5.0)), rng
    )
    ids = IdFactory(seed).child("twin2e-confounders")
    mag = float(config.get("confounder_magnitude", 5.0))
    return {
        "base": pair.factual,
        "hot_counterfactual": pair.counterfactual,
        "confounded_stable": make_confounded_variant(
            base, "HIGH_TEMPERATURE_STABLE_ECOLOGY", ids, mag, np.random.default_rng(seed + 7919)
        ),
        "confounded_disturbed": make_confounded_variant(
            base, "NORMAL_TEMPERATURE_DISTURBED_ECOLOGY", ids, mag, np.random.default_rng(seed + 7919)
        ),
    }


def _world_run(config: Mapping[str, Any], seed: int, scenario: Scenario) -> dict[str, Any]:
    tcfg = twin_config(config)
    twin = make_twin(scenario, tcfg, seed)
    rng = np.random.default_rng(seed + 17)
    lg = tcfg.local_grid
    lo = np.asarray(lg.origin_m)
    hi = lo + np.asarray(lg.spacing_m) * np.asarray(lg.shape)
    ep = Episode(twin, rng, seed, pose_sigma_m=float(config.get("pose_sigma_m", 0.3)))
    moorings = [
        ep.mooring(
            i, lo + (hi - lo) * rng.uniform(0.1, 0.9, 3), ("temperature", "turbidity"), 1 + i % 3, i % 2
        )
        for i in range(int(config.get("n_moorings", 4)))
    ]
    cam = ep.survey_sensor(float(config.get("footprint_radius_m", 6.0)))
    waypoints = survey_waypoints(scenario)
    variants = list(config.get("variants", VARIANTS))
    models = make_models(variants, seed, scenario, model_config(config))
    eval_every = int(config.get("eval_every_steps", 12))
    acc: dict[tuple[str, str], tuple[list[np.ndarray], list[np.ndarray]]] = {}

    def step(i: int) -> None:
        evs = [
            e
            for s in moorings
            if (i + s.phase) % s.period_steps == 0
            for e in ep.sense(s.spec, s.position, True)
        ]
        evs += ep.sense(cam, waypoints[i % len(waypoints)], pose_noise=True)
        for v, m in models.items():
            m.ingest(evs)
            m.update_beliefs(twin.now())
            if i >= eval_every and i % eval_every == 0:
                for key, (err, var) in (
                    ("cover", cover_errors(m, twin)[:2]),
                    ("temperature", field_errors(m, twin, "temperature")),
                    ("turbidity", field_errors(m, twin, "turbidity")),
                ):
                    a = acc.setdefault((v, key), ([], []))
                    a[0].append(err)
                    a[1].append(var)

    run_steps(twin, int(config.get("n_steps", 96)), float(config.get("dt_s", 900.0)), step)
    out: dict[str, Any] = {}
    for (v, key), (errs, vars_) in acc.items():
        out.setdefault(v, {})[key] = gaussian_scores(np.concatenate(errs), np.concatenate(vars_))
    conds = cover_errors(models[variants[0]], twin)[2]
    out["true_mean_condition"] = float(np.mean(conds))
    for v, m in models.items():
        stress = [b.stress_p for b in m.entities.beliefs.values() if b.sessile and b.asset is not None]
        truth = {eid: e.condition for eid, e in twin.entities.items()}
        spurious = [
            1.0 if (b.stress_p or 0.0) > 0.9 and truth[b.asset.registry_id] >= 0.9 else 0.0
            for b in m.entities.beliefs.values()
            if b.sessile and b.asset is not None
        ]
        out[v]["mean_stress_likelihood"] = (
            float(np.mean([s for s in stress if s is not None]))
            if any(s is not None for s in stress)
            else 0.0
        )
        out[v]["confident_stress_on_healthy_fraction"] = float(np.mean(spurious)) if spurious else 0.0
        out[v].update(unsupported_damage_claims(m))
    if "cefd" in models and "uncoupled" in models:
        c, u = out["cefd"], out["uncoupled"]
        out["CB_entity_cover_rmse"] = u["cover"]["rmse"] - c["cover"]["rmse"]
        out["CB_field_temperature_rmse"] = u["temperature"]["rmse"] - c["temperature"]["rmse"]
        out["CB_field_turbidity_rmse"] = u["turbidity"]["rmse"] - c["turbidity"]["rmse"]
        diffs = _cover_shift(models, twin.now().time_ns)
        out["mean_abs_cover_shift_cefd_vs_uncoupled"] = float(np.mean(diffs)) if diffs else 0.0
    return out


def _cover_shift(models: dict[str, Model2E], t_ns: int) -> list[float]:
    """|cover mean(cefd) - cover mean(uncoupled)| per registered sessile asset."""
    out = []
    for b in models["cefd"].entities.beliefs.values():
        if not b.sessile or b.asset is None:
            continue
        other = models["uncoupled"].entities.by_registry(b.asset.registry_id)
        if other is None:
            continue
        c = models["cefd"].entities.cover_moments_at(b, t_ns)[0]
        u = models["uncoupled"].entities.cover_moments_at(other, t_ns)[0]
        out.append(abs(c - u))
    return out


def _seed_run(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {name: _world_run(config, seed, scen) for name, scen in _worlds(config, seed).items()}


def run(config: Mapping[str, Any], seeds: Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    variants = list(config.get("variants", VARIANTS))
    return run_seeds(
        EXPERIMENT_ID, _seed_run, config, seeds, out_dir, candidate=variants[0], baselines=variants[1:]
    )
