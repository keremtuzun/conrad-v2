"""CORE-PERSIST-E001: persistent identity, phantom control, re-identification, provenance integrity,
duplicate re-delivery (CC-01) and working-memory reset equivalence (CC-09) over long sandbox episodes.

Variants: full PMBL; ablation without the candidate stage (confirm after 1 observation); full PMBL
plus an evaluation-side merge policy; full PMBL with periodic working-memory resets. Baseline for
state: latest-observation-only per hidden entity (uses truth association, so it is optimistic).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from conrad.core.config import CoreConfig
from conrad.core.pipeline import Model2Core
from conrad.evaluation.core_experiments.common import (
    core_config,
    run_experiment,
    scratch_dir,
    seed_everything,
)
from conrad.evaluation.core_experiments.pipeline_runner import RunLog, majority, root, run_pipeline
from conrad.evaluation.core_experiments.sandbox_world import SandboxConfig, SandboxStep, SandboxWorld
from conrad.schemas.belief import Lifecycle, UpdateKind

EXPERIMENT_ID = "CORE-PERSIST-E001"
CONFIRMED_LIKE = {Lifecycle.CONFIRMED, Lifecycle.ACTIVE, Lifecycle.REACTIVATED, Lifecycle.DORMANT}


def identity_metrics(
    core: Model2Core, log: RunLog, steps: Sequence[SandboxStep], gap: int
) -> dict[str, float]:
    maj = majority(log)
    weights = {b: sum(1 for e, a in log.assignment.items() if root(log, a) == b) for b in maj}
    purity = sum(maj[b][1] * w for b, w in weights.items()) / max(sum(weights.values()), 1)
    roots_per_entity: dict[int, set[Any]] = {}
    for eid, bid in log.assignment.items():
        src = log.source[eid]
        if src is not None:
            roots_per_entity.setdefault(src, set()).add(root(log, bid))
    heads = {c.belief_id: c for c in core.pmbl.store.heads()}
    ever_confirmed = {
        b for b in heads if any(r.cell.lifecycle in CONFIRMED_LIKE for r in core.pmbl.store.repo.revisions(b))
    }
    clutter_beliefs = [b for b, (src, _) in maj.items() if src is None]
    reid_hits = reid_total = 0
    last_seen: dict[int, tuple[int, Any]] = {}
    for eid in sorted(log.assignment, key=lambda e: (log.evidence_step[e], e.int)):
        src, k = log.source[eid], log.evidence_step[eid]
        if src is None:
            continue
        if src in last_seen and k - last_seen[src][0] >= gap:
            reid_total += 1
            reid_hits += int(root(log, log.assignment[eid]) == root(log, last_seen[src][1]))
        last_seen[src] = (k, log.assignment[eid])
    prov_ok = 0
    for cell in heads.values():
        try:
            core.pmbl.store.repo.provenance_closure(cell.provenance_root)
            prov_ok += 1
        except Exception:  # an unresolvable closure is exactly what this metric counts
            continue
    direct = [
        r for b in heads for r in core.pmbl.store.repo.revisions(b) if r.update_kind is UpdateKind.DIRECT
    ]
    return {
        "identity_purity": float(purity),
        "fragmentation_beliefs_per_entity": float(np.mean([len(v) for v in roots_per_entity.values()])),
        "false_confirmed_rate_clutter": float(
            np.mean([b in ever_confirmed for b in clutter_beliefs]) if clutter_beliefs else 0.0
        ),
        "n_clutter_beliefs": float(len(clutter_beliefs)),
        "reidentification_after_gap": reid_hits / reid_total if reid_total else float("nan"),
        "n_reidentification_events": float(reid_total),
        "provenance_integrity": prov_ok / max(len(heads), 1),
        "direct_revisions_with_evidence": float(
            np.mean([bool(r.consumed_evidence_ids) for r in direct]) if direct else 1.0
        ),
        "duplicate_rejection_rate": log.duplicates_rejected / log.duplicates_sent
        if log.duplicates_sent
        else float("nan"),
        "n_beliefs": float(len(heads)),
        "n_merges": float(len(log.merges)),
    }


def state_rmse(log: RunLog, steps: Sequence[SandboxStep]) -> float:
    maj = majority(log)
    errs = []
    for k, step in enumerate(steps):
        for bid, st in log.per_step_estimates[k].items():
            src = maj.get(root(log, bid), (None, 0.0))[0]
            if src is None or src not in step.truth.states:
                continue
            for p, est in st.estimates.items():
                errs.append(est.mean - step.truth.states[src][p])
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def latest_only_rmse(steps: Sequence[SandboxStep]) -> float:
    last: dict[int, dict[str, float]] = {}
    errs = []
    for step in steps:
        for ev, _ in step.evidence:
            src = step.truth.source[ev.evidence_id]
            if src is not None:
                last[src] = dict(ev.measurements)
        for src, meas in last.items():
            if src in step.truth.states:
                errs += [meas[p] - step.truth.states[src][p] for p in meas]
    return float(np.sqrt(np.mean(np.square(errs))))


def run_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    cfg = core_config(config)
    seed_everything(seed)
    wcfg = SandboxConfig(**config.get("sandbox", {}))
    steps = SandboxWorld(wcfg, seed).episode()
    q, gap = float(config.get("process_noise_per_s", 1e-5)), int(config.get("reid_gap_steps", 3))
    variants: dict[str, tuple[CoreConfig, dict[str, Any]]] = {
        "pmbl_full": (cfg, {}),
        "ablation_no_candidate_stage": (
            cfg.model_copy(
                update={"pmbl": cfg.pmbl.model_copy(update={"confirm_independent_observations": 1})}
            ),
            {},
        ),
        "pmbl_full_with_merge_policy": (
            cfg,
            {"merge_distance_m": float(config.get("merge_distance_m", 0.5))},
        ),
        "pmbl_full_with_resets": (cfg, {"reset_every": int(config.get("reset_every", 4))}),
    }
    out: dict[str, Any] = {}
    finals: dict[str, dict[Any, Any]] = {}
    for name, (vcfg, kw) in variants.items():
        with scratch_dir() as tmp:
            core, log = run_pipeline(
                vcfg,
                steps,
                tmp,
                seed,
                process_noise_per_s=q,
                redeliver_prob=float(config.get("redeliver_prob", 0.2)),
                **kw,
            )
            out[name] = {**identity_metrics(core, log, steps, gap), "state_rmse": state_rmse(log, steps)}
            finals[name] = {
                c.belief_id.int: tuple((cl.name, cl.value) for cl in c.claims)
                for c in core.pmbl.store.heads()
            }
    out["latest_only_baseline"] = {"state_rmse": latest_only_rmse(steps)}
    out["cc09_reset_equivalence"] = float(finals["pmbl_full"] == finals["pmbl_full_with_resets"])
    return out


def run(config: Mapping[str, Any], seed: int | Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    return run_experiment(
        EXPERIMENT_ID,
        run_seed,
        config,
        seed,
        out_dir,
        "pmbl_full",
        ["ablation_no_candidate_stage", "pmbl_full_with_merge_policy", "latest_only_baseline"],
    )
