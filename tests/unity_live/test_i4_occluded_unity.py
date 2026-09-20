"""Gate I4 FORMAL path on the world family ACTIVE_INSPECTION_OCCLUDED_V1.

Closed active inspection through the built Unity player on the held-out final split of
``configs/eval/partitions_i4_occluded.yaml`` (8000200-8000219). The family, the splits, the metrics and the
decision rule were declared in ``docs/audits/I4_WORLD_FAMILY.md`` before any world of the family was built;
what this run executes is declared in ``configs/eval/i4_occluded_unity.yaml`` before any flight.

This module replaces ``test_i4_unity.py`` as the formal I4 path. That module stays as the record of the
2026-09-19/20 run on ``straight_pipeline``, whose 12 worlds are SPENT and whose family did not exercise the
ch25 premise: the nominal transit route already saw the defect in a share of those worlds.

REDUCED ARM SET (declared, not chosen after the fact): four arms, exactly the ones the ch25 criteria name -
PRODUCTION (the frozen planner of ``configs/active/mcbr_frozen_v3.yaml`` with the belief-side predictive
model), A-B1 fixed nominal route, A-B0 random views, A-B2 coverage-only. At about 83 s per sequential flight
the full ten-arm comparison is 200 flights; the six arms not flown are selection evidence from the validation
split, never gate comparators. ``test_declared_arm_set_is_recorded`` refuses evidence that does not carry the
reduction and the dropped arms, exactly as the I7 harness does for its reduced bandwidth sweep.

Budgets are matched for every arm: mission duration 100 s, ``max_plans_per_need`` 4, one structural sensor,
the same candidate generator, feasibility filter and PlanningRequest. Energy and travel are measured. Runs are
strictly SEQUENTIAL (one player at a time). Scores are the target's actual hidden-state error against Twin
truth, read on the truth side after the run.
"""

from __future__ import annotations

import faulthandler
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from unity_gate_support import GATE_RUNS, GATES, MISSION_CONFIG, leakage_scan, measured, reset_measured

from conrad.active.production import PRODUCTION, load_frozen
from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.active_mcbr_i4_occluded import FAMILY_ID
from conrad.evaluation.decision_experiments.active_mcbr_reeval import score_mission_run
from conrad.evaluation.metrics import paired_seed_comparison
from conrad.orchestration.mission_config import runtime_config
from conrad.settings import REPO_ROOT
from conrad.sim.mission.run import settings_for
from conrad.sim.mission.unity_run import player_identity, prepare_unity, replay_unity_run, resolve_unity

GATE = "I4"
SCENARIO = "I4-OCCLUDED-UNITY"
CONFIG = REPO_ROOT / "configs" / "eval" / "i4_occluded_unity.yaml"
FROZEN = REPO_ROOT / "configs" / "active" / "mcbr_frozen_v3.yaml"
FIXED, RANDOM, COVERAGE = "A-B1_fixed_inspection", "A-B0_random", "A-B2_coverage"
RESULTS = GATES / GATE / "unity_i4_occluded_results.json"
# 80 sequential 100 s Unity missions take far longer than the 600 s per-test guard of conftest.py.
HARD_TIMEOUT_S = 6 * 3600.0


@pytest.fixture(autouse=True)
def _hard_timeout() -> Iterator[None]:  # overrides the conftest guard for this module only
    faulthandler.dump_traceback_later(HARD_TIMEOUT_S, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def config() -> dict[str, Any]:
    return dict(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


def arm_set(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg["arm_set"])


def planners(cfg: dict[str, Any]) -> tuple[str, ...]:
    arms = tuple(arm_set(cfg)["arms"])
    if arms[0] != PRODUCTION or set(arms[1:]) != {FIXED, RANDOM, COVERAGE}:
        raise AssertionError(f"the declared arm set must be PRODUCTION plus the three comparators: {arms}")
    return arms


def final_worlds(cfg: dict[str, Any]) -> list[int]:
    """The declared final split, read once. Every seed must belong to it; nothing else is reachable."""
    lo, hi = cfg["final_worlds"]["range"]
    declared = list(range(int(lo), int(hi)))
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        split = P.split(P.I4_OCCLUDED_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION)
    stray = sorted(set(declared) - set(split.world_seeds))
    if stray or not declared:
        raise P.PartitionAccessError(f"declared I4 worlds are not the final_test split: {stray[:10]}")
    return declared


def fly(scenario: str, seed: int, planner: str, root: Path) -> Path:
    settings = settings_for(MISSION_CONFIG)
    _, runtime_raw = resolve_unity(scenario, dict(settings.sim.get("mission", {})))
    rcfg = runtime_config({**runtime_raw, "planner": planner})
    run_id = f"{scenario}-s{seed}-{planner}"
    done = root / run_id / "reports" / "metrics.json"
    if done.exists() and (root / run_id / "bundle_manifest.json").exists():
        return root / run_id  # resume an interrupted session: finished bundles are kept, never re-scored
    s = prepare_unity(scenario, MISSION_CONFIG, run_id=run_id, runs_root=root, seed=seed, stored_runtime=rcfg)
    try:
        s.run()
        out = s.finish()
    except BaseException:
        s.abort()
        raise
    return Path(out["run_dir"])


@pytest.fixture(scope="module")
def flights() -> dict[str, Any]:
    cfg = config()
    arms = planners(cfg)
    seeds = final_worlds(cfg)
    root = GATE_RUNS / "I4-OCCLUDED"
    per_world: dict[str, dict[str, Any]] = {}
    dirs: dict[str, dict[str, str]] = {}
    for seed in seeds:  # strictly sequential: one Unity player at a time
        for planner in arms:
            d = fly(SCENARIO, seed, planner, root)
            per_world.setdefault(str(seed), {})[planner] = score_mission_run(
                d, seed, FAMILY_ID, planner, float(cfg["runtime"]["duration_s"])
            )
            dirs.setdefault(str(seed), {})[planner] = str(d.relative_to(REPO_ROOT))
    frozen = load_frozen(FROZEN)
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        digest = P.load_i4_occluded()["digest"]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "scenario": SCENARIO,
                "evidence_kind": "FORMAL",
                "world_family": FAMILY_ID,
                "partition_file": cfg["partition_file"],
                "partition_digest": digest,
                "family_digest": cfg["family_digest"],
                "world_seeds": seeds,
                "arm_set": arm_set(cfg),
                "runtime": cfg["runtime"],
                "decision_rule": cfg["decision_rule"],
                "frozen_planner": {
                    "file": str(FROZEN.relative_to(REPO_ROOT)),
                    "selected": frozen["planner"]["selected"],
                    "config_digest": frozen["config_digest"],
                    "mission_predictive_digest": frozen.get("mission_predictive_digest"),
                },
                "player": player_identity(),
                "run_dirs": dirs,
                "per_world": per_world,
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    reset_measured(
        GATE,
        {
            "scenario": SCENARIO,
            "world_family": FAMILY_ID,
            "worlds": seeds,
            "arm_set": arm_set(cfg),
            "partition": f"{cfg['partition_file']} final_test (declared in configs/eval/i4_occluded_unity.yaml)",
            "partition_digest": digest,
            "supersedes": "the straight_pipeline formal I4 run of 2026-09-19/20 (12 SPENT worlds; that family "
            "did not exercise the ch25 premise)",
            "results": str(RESULTS.relative_to(REPO_ROOT)),
        },
    )
    return {"per_world": per_world, "dirs": dirs, "seeds": seeds, "cfg": cfg, "arms": arms}


def _paired(per_world: dict[str, dict[str, Any]], base: str, metric: str, direction: str) -> dict[str, Any]:
    b = {int(w): float(per_world[w][base][metric]) for w in per_world}
    c = {int(w): float(per_world[w][PRODUCTION][metric]) for w in per_world}
    pc = paired_seed_comparison(
        b, c, np.random.default_rng(20260919), metric=metric, direction=direction, n_resamples=4000
    )
    return {
        "benefit_mean": pc.benefit.point,
        "ci95": [pc.benefit.low, pc.benefit.high],
        "n_worlds": pc.benefit.n,
        "production_mean": pc.candidate_mean,
        "baseline_mean": pc.baseline_mean,
        "fraction_improved": pc.fraction_of_seeds_improved,
    }


def _beats(row: dict[str, Any]) -> bool:
    low = row["ci95"][0]
    return low is not None and low > 0.0


def _criterion(flights: dict[str, Any], name: str, base: str, metrics: dict[str, str]) -> None:
    pw = flights["per_world"]
    rows = {m: _paired(pw, base, m, d) for m, d in metrics.items()}
    means = {p: {m: float(np.mean([pw[w][p][m] for w in pw])) for m in metrics} for p in (PRODUCTION, base)}
    measured(GATE, name, comparator=base, paired=rows, means=means)
    failures = [f"{m}: {r['benefit_mean']:+.4f} CI95 {r['ci95']}" for m, r in rows.items() if not _beats(r)]
    assert not failures, f"PRODUCTION does not beat {base}: " + "; ".join(failures)


def test_declared_arm_set_is_recorded(flights):
    """A reduced arm set is evidence only when it is recorded as reduced, with the arms it drops named."""
    a = arm_set(flights["cfg"])
    measured(
        GATE,
        "arm set",
        name=a["name"],
        reduced=a["reduced"],
        arms=a["arms"],
        dropped_arms=a["dropped_arms"],
        reason=a["reason"],
        dropped_arms_evidence=a["dropped_arms_evidence"],
        flights=len(flights["seeds"]) * len(flights["arms"]),
        world_family=FAMILY_ID,
    )
    assert a["reduced"] and a["dropped_arms"], "a reduced arm set must name the arms it drops"
    assert set(a["arms"]).isdisjoint(a["dropped_arms"])
    # no criterion may depend on a dropped arm
    assert {FIXED, RANDOM, COVERAGE} <= set(a["arms"])
    assert len(flights["seeds"]) * len(flights["arms"]) == a["flights"]["final"]


def test_results_file_is_written(flights):
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    frozen = load_frozen(FROZEN)
    measured(
        GATE,
        "results",
        path=str(RESULTS.relative_to(REPO_ROOT)),
        n_worlds=len(data["per_world"]),
        partition_digest=data["partition_digest"],
        family_digest=data["family_digest"],
        config_digest=data["frozen_planner"]["config_digest"],
    )
    assert data["partition_digest"] == P.I4_OCCLUDED_PARTITIONS_SHA256
    assert data["frozen_planner"]["config_digest"] == frozen["config_digest"]
    assert len(data["per_world"]) == len(flights["seeds"])


def test_closed_loop_mcbr_view_improves_hidden_target(flights):
    """critical structure partly hidden -> uncertain -> MCBR view -> navigation -> new evidence -> belief improves"""
    pw = flights["per_world"]
    prod = [pw[w][PRODUCTION] for w in pw]
    planned = [r for r in prod if "PLAN" in r["plans"]]
    informative = [r for r in planned if r["observations"] > r["redundant_observations"]]
    hse = float(np.mean([r["hidden_state_error_improvement"] for r in prod]))
    fixed_seen = float(np.mean([pw[w][FIXED]["patch_max_visible_fraction"] for w in pw]))
    measured(
        GATE,
        "closed loop",
        worlds=len(prod),
        worlds_where_mcbr_planned=len(planned),
        worlds_with_informative_mcbr_view=len(informative),
        mean_hidden_state_error_improvement=hse,
        target_observed_fraction=float(np.mean([r["target_observed"] for r in prod])),
        patch_max_visible_fraction=[r["patch_max_visible_fraction"] for r in prod],
        # evidence that the premise holds in this family: the nominal route barely sees the critical surface
        fixed_route_patch_max_visible_fraction_mean=fixed_seen,
        fixed_route_target_observed_fraction=float(np.mean([pw[w][FIXED]["target_observed"] for w in pw])),
    )
    assert planned, "MCBR never planned a view on any I4 world"
    assert len(informative) * 2 >= len(planned) and hse > 0.0


def test_beats_fixed_views_on_hidden_state_reconstruction(flights):
    _criterion(flights, "beats fixed views", FIXED, {"hidden_state_error_improvement": "higher"})


def test_beats_random_views(flights):
    _criterion(flights, "beats random views", RANDOM, {"hidden_state_error_improvement": "higher"})


def test_beats_coverage_only(flights):
    _criterion(flights, "beats coverage-only", COVERAGE, {"hidden_state_error_improvement": "higher"})


@pytest.mark.parametrize("base", [FIXED, RANDOM, COVERAGE])
def test_beats_simple_views_on_information_per_time_and_energy(flights, base):
    _criterion(
        flights,
        f"information/time/energy vs {base}",
        base,
        {"info_per_time": "higher", "info_per_kj": "higher"},
    )


def test_no_twin_truth_leakage_on_runtime_side(flights):
    scanned, violations = 0, []
    for w, by_planner in flights["dirs"].items():
        d = REPO_ROOT / by_planner[PRODUCTION]
        meta = json.loads((d / "truth" / "truth_record.json").read_text(encoding="utf-8"))["meta"]
        occluders = set((meta.get("view_occlusion") or {}).get("entity_ids", []))
        n, v = leakage_scan(d, set(meta["world_entity_ids"]) | occluders, set())
        scanned += n
        violations += [f"{w}: {x}" for x in v]
    measured(GATE, "leakage", texts_scanned=scanned, violations=violations[:10])
    assert scanned > 100 and not violations


def test_bundle_replays_deterministically(flights):
    first = str(flights["seeds"][0])
    rep = replay_unity_run(
        REPO_ROOT / flights["dirs"][first][PRODUCTION], GATE_RUNS / "replay" / "I4-OCCLUDED"
    )
    measured(
        GATE,
        "replay",
        world=int(first),
        **{
            k: rep[k]
            for k in (
                "events",
                "decisions",
                "revisions",
                "trajectory",
                "equal",
                "verified_files",
                "verified_objects",
                "same_player_binary",
            )
        },
    )
    assert rep["equal"], rep
