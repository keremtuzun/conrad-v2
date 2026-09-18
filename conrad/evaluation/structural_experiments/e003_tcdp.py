"""2T-E003: TCDP relational benefit vs relational contamination (2T-D4, T5/T6; ch10 RB and RC).

Arms: INDEPENDENT_COMPONENT (no propagation), GENERIC_RELATIONAL (mechanism-agnostic), TCDP.
Episode kinds (all Twin2T truth):
  coupled    - correlated initial states injected along mechanism-valid edges of one hot cluster plus a
               shared ENVIRONMENT_CHANGE event (the coupling TCDP is meant to exploit; scenario-injected);
  misleading - localized severe degradation beside healthy neighbours (surface region via PART_OF, weld via
               ATTACHED_TO, crack on a segment next to its support and weld);
  natural    - unmodified tier-3 procedural scenarios.
RB = Error_independent - Error_TCDP on hidden components (UNKNOWN -> the model's own latent prior mean).
RC = P(claims degradation | hidden, truly healthy, with a truly degraded observed neighbour).
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from conrad.domains.technical import CORROSION_DEPTH, PropagationMode, TCDPConfig, relational_contamination
from conrad.domains.technical.baselines import EngineEstimator
from conrad.domains.technical.config import YEAR_S, Model2TConfig
from conrad.domains.technical.registry import QUANTITY_MECHANISM
from conrad.evaluation.structural_experiments.common import (
    DAY,
    QUANTITIES,
    make_world,
    run_episode,
    write_result,
)
from conrad.evaluation.structural_experiments.scenarios import (
    entity_types,
    make_scenario,
    populate,
    relations,
    unpopulated_small,
    with_events,
)
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Scenario, ScenarioEvent

EXPERIMENT_ID = "2T-E003"
ARMS = (PropagationMode.NONE, PropagationMode.GENERIC, PropagationMode.TCDP)
SHARED_ENV = ("CONNECTED_TO", "ATTACHED_TO", "CONTACTS")


def _nbrs(
    rels: list[tuple[UUID, UUID, str]], eid: UUID, types: Sequence[str] | None = None
) -> list[tuple[UUID, str]]:
    out = []
    for a, b, t in rels:
        if types is not None and t not in types:
            continue
        if a == eid:
            out.append((b, t))
        elif b == eid:
            out.append((a, t))
    return out


def coupled_scenario(seed: int, n_segments: int) -> Scenario:
    base = unpopulated_small(seed, n_segments)
    rng = np.random.default_rng([seed, 21])
    types, rels = entity_types(base), relations(base)
    segs = sorted((e for e, t in types.items() if t == "SEGMENT"), key=str)
    hot = segs[int(rng.integers(len(segs)))]
    cluster = {hot} | {n for n, _ in _nbrs(rels, hot, SHARED_ENV)}
    load = {n for n, _ in _nbrs(rels, hot, ("SUPPORTED_BY",))}
    vals: dict[UUID, dict[str, float]] = {}
    for e, t in types.items():
        if t in ("ASSET", "PIPELINE"):
            continue
        corr = rng.uniform(2.0e-3, 3.5e-3) if e in cluster else rng.uniform(0.0, 4.0e-4)
        crack = rng.uniform(8e-3, 1.2e-2) if e in load | {hot} else rng.uniform(0.0, 1.0e-3)
        vals[e] = {"initial_corrosion_depth_m": float(corr), "initial_crack_length_m": float(crack)}
    ev = ScenarioEvent(
        event_id=IdFactory(seed).child("events").new(),
        time_s=0.0,
        event_type="ENVIRONMENT_CHANGE",
        target_entity_id=hot,
        parameters={"temperature_c": 28.0, "dissolved_oxygen_mg_l": 10.0},
    )
    return with_events(populate(base, seed, vals), [ev])


def misleading_scenario(seed: int, n_segments: int) -> tuple[Scenario, set[UUID]]:
    base = unpopulated_small(seed, n_segments)
    rng = np.random.default_rng([seed, 22])
    types, rels = entity_types(base), relations(base)
    segs = sorted((e for e, t in types.items() if t == "SEGMENT"), key=str)
    parts = {s: [n for n, t in _nbrs(rels, s) if types[n] in ("SURFACE_REGION", "WELD")] for s in segs}
    region = next(n for n in parts[segs[0]] if types[n] == "SURFACE_REGION")
    weld = next(n for n in parts[segs[1 % len(segs)]] if types[n] == "WELD")
    cracked = segs[-1]
    vals: dict[UUID, dict[str, float]] = {
        e: {
            "initial_corrosion_depth_m": float(rng.uniform(0, 3e-4)),
            "initial_crack_length_m": float(rng.uniform(0, 1e-3)),
        }
        for e, t in types.items()
        if t not in ("ASSET", "PIPELINE")
    }
    vals[region]["initial_corrosion_depth_m"] = 3.5e-3
    vals[weld]["initial_corrosion_depth_m"] = 3.5e-3
    vals[cracked]["initial_crack_length_m"] = 1.5e-2
    return populate(base, seed, vals), {region, weld, cracked}


def _prior_level(cfg: Model2TConfig, q: str, t_s: float) -> float:
    rate = cfg.dynamics.corrosion_rate_m_per_yr if q == CORROSION_DEPTH else cfg.dynamics.crack_rate_m_per_yr
    return cfg.prior.mean[q] + rate * t_s / YEAR_S


def run_episode_kind(kind: str, eseed: int, config: Mapping[str, Any], tmp: Path) -> dict[str, list[float]]:
    n_seg = int(config.get("n_segments", 6))
    rng = np.random.default_rng([eseed, 23])
    degraded_src: set[UUID] = set()
    if kind == "coupled":
        scenario = coupled_scenario(eseed, n_seg)
    elif kind == "misleading":
        scenario, degraded_src = misleading_scenario(eseed, n_seg)
    else:
        scenario = make_scenario(eseed, {"kind": "tier", "tier": 3})
    world = make_world(eseed, scenario, tmp)
    rels = relations(scenario)
    comps = list(world.component_ids)
    must_hide = {n for d in degraded_src for n, _ in _nbrs(rels, d) if n in comps and n not in degraded_src}
    rest = [c for c in comps if c not in must_hide and c not in degraded_src]
    frac = float(config.get("observed_fraction", 0.5))
    observed = set(degraded_src) | {c for c in rest if rng.random() < frac}
    hidden = [c for c in comps if c not in observed]
    cfg = Model2TConfig()
    arms = [EngineEstimator(m, cfg, seed=eseed) for m in ARMS]
    recs = run_episode(
        world,
        arms,
        steps=int(config.get("steps", 12)),
        dt_s=float(config.get("dt_days", 60)) * DAY,
        visible_at=lambda _k: sorted(observed, key=str),
    )
    tau = {q: float(v) for q, v in dict(config.get("degraded_threshold_m", {})).items()} or {
        "corrosion_depth_m": 1.0e-3,
        "crack_length_m": 5.0e-3,
    }
    valid_types = TCDPConfig().mechanism_relations
    acc: dict[str, list[float]] = {}
    for rec in recs[int(config.get("burn_in_steps", 2)) :]:
        tr = rec["truth"]
        for q in QUANTITIES:
            vtypes = valid_types[QUANTITY_MECHANISM[q].value]
            rc_in: dict[str, list[list[bool]]] = {a.name: [[], [], []] for a in arms}
            for rid in hidden:
                if tr[rid][q] is None:
                    continue
                reachable = any(n in observed for n, _ in _nbrs(rels, rid, vtypes))
                healthy = tr[rid][q] < tau[q]
                bad_nbr = any(
                    n in observed and n in tr and tr[n][q] is not None and tr[n][q] >= tau[q]
                    for n, _ in _nbrs(rels, rid)
                )
                for a in arms:
                    e = rec["est"][a.name][rid][q]
                    mean = e[0] if e is not None else _prior_level(cfg, q, rec["time_s"])
                    err = abs(mean - tr[rid][q]) * 1e3
                    acc.setdefault(f"{kind}.{a.name}.{q}.hidden_mae_mm", []).append(err)
                    if reachable:
                        acc.setdefault(f"{kind}.{a.name}.{q}.reachable_hidden_mae_mm", []).append(err)
                    acc.setdefault(f"{kind}.{a.name}.{q}.claim_rate_hidden", []).append(float(e is not None))
                    rc_in[a.name][0].append(e is not None and e[0] >= tau[q])
                    rc_in[a.name][1].append(healthy)
                    rc_in[a.name][2].append(bad_nbr)
            for name, (c, h, n) in rc_in.items():
                pool = sum(1 for x, y in zip(h, n, strict=True) if x and y)
                if pool:
                    rc = relational_contamination(c, h, n) or 0.0
                    acc.setdefault(f"{kind}.{name}.{q}.rc_hits", []).append(rc * pool)
                    acc.setdefault(f"{kind}.{name}.{q}.rc_pool", []).append(float(pool))
    return acc


def run_seed(config: Mapping[str, Any], seed: int, tmp: Path) -> dict[str, Any]:
    acc: dict[str, list[float]] = {}
    for kind in config.get("kinds", ["coupled", "misleading", "natural"]):
        for ep in range(int(config.get("episodes_per_kind", 3))):
            for k, v in run_episode_kind(kind, seed * 100 + ep, config, tmp).items():
                acc.setdefault(k, []).extend(v)
    pooled = ("rc_pool", "rc_hits")
    out: dict[str, Any] = {k: float(np.mean(v)) for k, v in acc.items() if not k.endswith(pooled)}
    out.update({k: float(np.sum(v)) for k, v in acc.items() if k.endswith(pooled)})
    for k in [k for k in out if k.endswith("rc_pool")]:
        out[k.replace("rc_pool", "rc")] = out[k.replace("rc_pool", "rc_hits")] / out[k]
    for kind in config.get("kinds", ["coupled", "misleading", "natural"]):
        for q in QUANTITIES:
            for sub in ("hidden_mae_mm", "reachable_hidden_mae_mm"):
                ind, tc, gen = (
                    out.get(f"{kind}.{n}.{q}.{sub}")
                    for n in ("INDEPENDENT_COMPONENT", "TCDP", "GENERIC_RELATIONAL")
                )
                if ind is not None and tc is not None and gen is not None:
                    out[f"{kind}.{q}.RB_tcdp.{sub}"] = ind - tc
                    out[f"{kind}.{q}.RB_generic.{sub}"] = ind - gen
    return out


def run(config: Mapping[str, Any], seeds: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    seeds = [seeds] if isinstance(seeds, int) else list(seeds)
    with tempfile.TemporaryDirectory(prefix="m2t_e003_", ignore_cleanup_errors=True) as tmp:
        per_seed = {s: run_seed(config, s, Path(tmp)) for s in seeds}
    verdicts: dict[str, Any] = {}
    for kind in config.get("kinds", ["coupled", "misleading", "natural"]):
        for q in QUANTITIES:
            rb = [per_seed[s].get(f"{kind}.{q}.RB_tcdp.reachable_hidden_mae_mm") for s in seeds]
            verdicts[f"{kind}.{q}.tcdp_RB_positive_all_seeds"] = all(x is not None and x > 0 for x in rb)
            rc_t = [per_seed[s].get(f"{kind}.TCDP.{q}.rc") for s in seeds]
            rc_g = [per_seed[s].get(f"{kind}.GENERIC_RELATIONAL.{q}.rc") for s in seeds]
            pairs = [(a, b) for a, b in zip(rc_t, rc_g, strict=True) if a is not None and b is not None]
            if len(pairs) == len(seeds):
                verdicts[f"{kind}.{q}.tcdp_rc_below_generic_all_seeds"] = all(a < b for a, b in pairs)
    return write_result(EXPERIMENT_ID, config, seeds, per_seed, out_dir, started, verdicts)
