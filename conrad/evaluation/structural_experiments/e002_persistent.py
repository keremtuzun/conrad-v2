"""2T-E002: persistent updating under partial coverage (2T-D2, T3; coverage levels from ch10).

Arms: MODEL2T (full Model2Child path with its production defaults: TCDP on until ADR-0009, OFF since),
INDEPENDENT_COMPONENT, LATEST_OBSERVATION, GRU_TEMPORAL.
Metrics: error on components hidden now but seen before; UNKNOWN / INFERRED rate on never-observed
components; U_O by visibility class; one persisted episode per seed through the Repository.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from conrad.domains.technical import PropagationMode
from conrad.domains.technical.baselines import (
    EngineEstimator,
    GRUTemporal,
    LatestObservation,
    Model2TEstimator,
    StructuralEstimator,
)
from conrad.evaluation.partitions import purpose_scope
from conrad.evaluation.structural_experiments.common import (
    CLOCK,
    DAY,
    QUANTITIES,
    World,
    checked_seeds,
    experiment_id,
    gaussian_scores,
    make_world,
    paired_bootstrap,
    run_episode,
    summarize,
    write_result,
)
from conrad.evaluation.structural_experiments.scenarios import coverage_schedule, make_scenario
from conrad.persistence.db import make_engine, migrate
from conrad.persistence.repository import Repository
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp

EXPERIMENT_ID = "2T-E002"


def _world(config: Mapping[str, Any], eseed: int, tmp: Path) -> World:
    return make_world(eseed, make_scenario(eseed, config.get("scenario", {})), tmp)


def train_gru(config: Mapping[str, Any], seed: int, tmp: Path) -> GRUTemporal:
    """Fit B2 on separate training episodes (truth used only as regression targets)."""
    gcfg = dict(config.get("gru", {}))
    gru = GRUTemporal(
        int(gcfg.get("hidden", 16)), int(gcfg.get("epochs", 150)), float(gcfg.get("lr", 1e-2)), seed
    )
    seqs = []
    rng = np.random.default_rng([seed, 99])
    steps, dt = int(config.get("steps", 20)), float(config.get("dt_days", 45)) * DAY
    for j in range(int(gcfg.get("train_episodes", 6))):
        eseed = seed * 1000 + 500 + j
        world = _world(config, eseed, tmp)
        cov = float(rng.choice(config.get("coverages", [0.5, 0.2, 0.05])))
        vis = coverage_schedule(world.component_ids, cov, eseed)
        gru.reset(world.registry, stamp(0.0, CLOCK))
        truth_seq = []
        for k in range(1, steps + 1):
            world.twin.advance(dt)
            gru.step(stamp(world.twin.time_s, CLOCK), world.observe(vis(k)))
            truth_seq.append(world.truth())
        for rid in world.component_ids:
            hist = np.asarray(gru.history[rid], dtype=np.float64)
            seen = np.cumsum(hist[:, 3]) > 0
            y = np.zeros((steps, 2))
            mask = np.zeros((steps, 2), dtype=bool)
            for k, tr in enumerate(truth_seq):
                for i, q in enumerate(QUANTITIES):
                    if tr[rid][q] is not None and seen[k]:
                        y[k, i], mask[k, i] = 1e3 * float(tr[rid][q] or 0.0), True
            if mask.any():
                seqs.append((hist, y, mask))
    gru.fit(seqs)
    return gru


def _score(recs: list[dict[str, Any]], paired: bool = False) -> dict[str, float | None]:
    """Per visibility class. ``paired`` (R3 configs, declared before the run): Model2T arms are scored on their
    internal estimate (``latent``, claimed or not, as in 2T-E001-R2) and an item counts only if EVERY arm has an
    estimate, so all arms are compared on the same components. Unpaired (R2 and earlier): each arm on the
    items it claims, which compared Model2T's detected cracks against the baselines' every crack."""
    acc: dict[str, list[float]] = {}
    seen: set[Any] = set()
    for rec in recs:
        obs = rec["observed"]
        for rid, tr in rec["truth"].items():
            cls = "observed_now" if rid in obs else "hidden_seen_before" if rid in seen else "never_observed"
            for q in QUANTITIES:
                if tr[q] is None:
                    continue
                ests: dict[str, Any] = {}
                for name, per in rec["est"].items():
                    e = per[rid][q]
                    acc.setdefault(f"{cls}.{name}.{q}.unknown_rate", []).append(float(e is None))
                    if paired and name in rec["latent"] and cls != "never_observed":
                        e = rec["latent"][name][rid][q]
                    ests[name] = e
                if paired and any(e is None for e in ests.values()):
                    continue
                for name, e in ests.items():
                    if e is not None:
                        err, _nll, cov = gaussian_scores(e[0], e[1], tr[q])
                        acc.setdefault(f"{cls}.{name}.{q}.mae_mm", []).append(err)
                        acc.setdefault(f"{cls}.{name}.{q}.cov95", []).append(cov)
            acc.setdefault(f"{cls}.MODEL2T.mean_uo", []).append(rec["probe"][rid])
        seen |= obs
    return summarize(acc)


def _uo_probe(model: Model2TEstimator) -> Callable[[], dict[Any, float]]:
    def probe() -> dict[Any, float]:
        assert model.model is not None
        return {rid: b.uncertainty().observational for rid, b in model.model.beliefs.items()}

    return probe


def run_seed(config: Mapping[str, Any], seed: int, tmp: Path) -> dict[str, Any]:
    gru = train_gru(config, seed, tmp)
    out: dict[str, Any] = {}
    for cov in config.get("coverages", [0.5, 0.2, 0.05]):
        scores: list[dict[str, float | None]] = []
        for ep in range(int(config.get("episodes_per_seed", 2))):
            eseed = seed * 100 + ep
            world = _world(config, eseed, tmp)
            model = Model2TEstimator(seed=eseed)
            ests: list[StructuralEstimator] = [
                model,
                EngineEstimator(PropagationMode.NONE, seed=eseed),
                LatestObservation(),
                gru,
            ]
            recs = run_episode(
                world,
                ests,
                steps=int(config.get("steps", 20)),
                dt_s=float(config.get("dt_days", 45)) * DAY,
                visible_at=coverage_schedule(world.component_ids, float(cov), eseed),
                probe=_uo_probe(model),
            )
            scores.append(_score(recs, str(config.get("scoring", "unpaired")) == "paired_latent"))
        keys = sorted({k for s in scores for k in s})
        out[f"coverage_{cov}"] = {
            k: float(np.mean(v)) if (v := [float(x) for s in scores if (x := s.get(k)) is not None]) else None
            for k in keys
        }
    out["persistence"] = _persisted_episode(config, seed, tmp)
    return out


def _persisted_episode(config: Mapping[str, Any], seed: int, tmp: Path) -> dict[str, float]:
    path = tmp / f"e002_{seed}.sqlite"
    migrate(path)
    repo = Repository(make_engine(path))
    run_id = IdFactory(seed).child("run").new()
    world = _world(config, seed * 100, tmp)
    model = Model2TEstimator(seed=seed, repository=repo, run_id=run_id)
    t0 = time.perf_counter()
    run_episode(
        world,
        [model],
        steps=int(config.get("steps", 20)),
        dt_s=float(config.get("dt_days", 45)) * DAY,
        visible_at=coverage_schedule(world.component_ids, 0.2, seed * 100),
    )
    assert model.model is not None
    heads = repo.heads(run_id)
    return {
        "revisions_committed": float(len(repo.all_revisions(run_id))),
        "heads": float(len(heads)),
        "beliefs": float(len(model.model.beliefs)),
        "heads_match_memory": float(
            all(
                model.model.beliefs[h.registry_entity_id].revision == h.revision
                for h in heads
                if h.registry_entity_id
            )
        ),
        "wall_time_s": time.perf_counter() - t0,
    }


def run(config: Mapping[str, Any], seeds: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    seeds = checked_seeds(config, [seeds] if isinstance(seeds, int) else list(seeds))
    with (
        purpose_scope(str(config.get("purpose", "design"))),
        tempfile.TemporaryDirectory(prefix="m2t_e002_", ignore_cleanup_errors=True) as tmp,
    ):
        per_seed = {s: run_seed(config, s, Path(tmp)) for s in seeds}
    verdicts: dict[str, Any] = {}
    for cov in config.get("coverages", [0.5, 0.2, 0.05]):
        for q in QUANTITIES:
            for base in ("LATEST_OBSERVATION", "GRU_TEMPORAL"):
                pairs = [
                    (
                        per_seed[s][f"coverage_{cov}"].get(f"hidden_seen_before.MODEL2T.{q}.mae_mm"),
                        per_seed[s][f"coverage_{cov}"].get(f"hidden_seen_before.{base}.{q}.mae_mm"),
                    )
                    for s in seeds
                ]
                verdicts[f"coverage_{cov}.{q}.model2t_beats_{base}_hidden_all_seeds"] = all(
                    a is not None and b is not None and a < b for a, b in pairs
                )
            ur = [
                per_seed[s][f"coverage_{cov}"].get(f"never_observed.INDEPENDENT_COMPONENT.{q}.unknown_rate")
                for s in seeds
            ]
            verdicts[f"coverage_{cov}.{q}.independent_unknown_rate_on_never_observed"] = ur
            for base in ("LATEST_OBSERVATION", "GRU_TEMPORAL"):
                diffs = []
                for s in seeds:
                    c = per_seed[s][f"coverage_{cov}"]
                    a, b = (
                        c.get(f"hidden_seen_before.{base}.{q}.mae_mm"),
                        c.get(f"hidden_seen_before.MODEL2T.{q}.mae_mm"),
                    )
                    if a is not None and b is not None:
                        diffs.append(a - b)
                verdicts[f"coverage_{cov}.{q}.hidden_mae_gain_vs_{base}_ci"] = paired_bootstrap(diffs)
    return write_result(
        experiment_id(config, EXPERIMENT_ID), config, seeds, per_seed, out_dir, started, verdicts
    )
