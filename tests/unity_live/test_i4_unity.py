"""Gate I4 FORMAL path: closed active inspection through the built Unity player on held-out worlds.

Worlds: the 12 I4 worlds declared in ``configs/eval/active_mcbr_e004.yaml`` BEFORE any run (unity_gate final_test
7800002-7800013; 7800000/7800001 are the I1/I3 roles). Same worlds as the surrogate ACTIVE-MCBR-E004.

Every world is flown once per planner (PRODUCTION = the frozen planner of ``configs/active/mcbr_frozen_v2.yaml``
with the belief-side predictive model; A-B1 fixed views; A-B0 random views; A-B2 coverage-only) with the SAME
budgets: mission duration 100 s (time), ``max_plans_per_need`` 4 (observations), one structural sensor, the same
candidate generator, feasibility filter and PlanningRequest. Energy and travel are measured. Runs are strictly
SEQUENTIAL (one player at a time). Scores are the target's actual hidden-state error against Twin truth, read on
the truth side after the run.

Decision rule (pre-declared in the E004 config): PRODUCTION beats a baseline iff the lower bound of the paired
95 % percentile-bootstrap CI over worlds (4000 resamples) of the benefit is > 0.
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
from conrad.evaluation.decision_experiments.active_mcbr_e004 import declared_i4_worlds
from conrad.evaluation.decision_experiments.active_mcbr_reeval import score_mission_run
from conrad.evaluation.metrics import paired_seed_comparison
from conrad.orchestration.mission_config import runtime_config
from conrad.settings import REPO_ROOT
from conrad.sim.mission.run import settings_for
from conrad.sim.mission.unity_run import player_identity, prepare_unity, replay_unity_run, resolve_unity

GATE = "I4"
SCENARIO = "I4-UNITY"
E004_CONFIG = REPO_ROOT / "configs" / "eval" / "active_mcbr_e004.yaml"
FIXED, RANDOM, COVERAGE = "A-B1_fixed_inspection", "A-B0_random", "A-B2_coverage"
PLANNERS = (PRODUCTION, FIXED, RANDOM, COVERAGE)
RESULTS = GATES / GATE / "unity_i4_results.json"
FAMILY = "straight_pipeline"
# 48 sequential 100 s Unity missions take far longer than the 600 s per-test guard of conftest.py.
I4_HARD_TIMEOUT_S = 4 * 3600.0


@pytest.fixture(autouse=True)
def _hard_timeout() -> Iterator[None]:  # overrides the conftest guard for this module only
    faulthandler.dump_traceback_later(I4_HARD_TIMEOUT_S, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _worlds() -> list[int]:
    cfg = yaml.safe_load(E004_CONFIG.read_text(encoding="utf-8"))
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        return declared_i4_worlds(cfg)


def _fly(seed: int, planner: str, root: Path) -> Path:
    settings = settings_for(MISSION_CONFIG)
    _, runtime_raw = resolve_unity(SCENARIO, dict(settings.sim.get("mission", {})))
    rcfg = runtime_config({**runtime_raw, "planner": planner})
    run_id = f"{SCENARIO}-s{seed}-{planner}"
    done = root / run_id / "reports" / "metrics.json"
    if done.exists() and (root / run_id / "bundle_manifest.json").exists():
        return root / run_id  # resume an interrupted session: finished bundles are kept, never re-scored
    s = prepare_unity(SCENARIO, MISSION_CONFIG, run_id=run_id, runs_root=root, seed=seed, stored_runtime=rcfg)
    try:
        s.run()
        out = s.finish()
    except BaseException:
        s.abort()
        raise
    return Path(out["run_dir"])


@pytest.fixture(scope="module")
def flights() -> dict[str, Any]:
    seeds = _worlds()
    root = GATE_RUNS / GATE
    per_world: dict[str, dict[str, Any]] = {}
    dirs: dict[str, dict[str, str]] = {}
    for seed in seeds:  # strictly sequential: one Unity player at a time
        for planner in PLANNERS:
            d = _fly(seed, planner, root)
            row = score_mission_run(d, seed, FAMILY, planner, 100.0)
            per_world.setdefault(str(seed), {})[planner] = row
            dirs.setdefault(str(seed), {})[planner] = str(d.relative_to(REPO_ROOT))
    frozen = load_frozen()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "scenario": SCENARIO,
                "evidence_kind": "FORMAL",
                "partition_file": "configs/eval/partitions_unity_gates.yaml",
                "partition_digest": P.load_unity_gates()["digest"],
                "world_seeds": seeds,
                "planners": list(PLANNERS),
                "frozen_planner": {
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
            "worlds": seeds,
            "planners": list(PLANNERS),
            "partition": "configs/eval/partitions_unity_gates.yaml final_test (declared in active_mcbr_e004.yaml)",
            "results": str(RESULTS.relative_to(REPO_ROOT)),
        },
    )
    return {"per_world": per_world, "dirs": dirs, "seeds": seeds}


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


def test_closed_loop_mcbr_view_improves_hidden_target(flights):
    """critical structure partly hidden -> uncertain -> MCBR view -> navigation -> new evidence -> belief improves"""
    pw = flights["per_world"]
    prod = [pw[w][PRODUCTION] for w in pw]
    planned = [r for r in prod if "PLAN" in r["plans"]]
    informative = [r for r in planned if r["observations"] > r["redundant_observations"]]
    hse = float(np.mean([r["hidden_state_error_improvement"] for r in prod]))
    measured(
        GATE,
        "closed loop",
        worlds=len(prod),
        worlds_where_mcbr_planned=len(planned),
        worlds_with_informative_mcbr_view=len(informative),
        mean_hidden_state_error_improvement=hse,
        target_observed_fraction=float(np.mean([r["target_observed"] for r in prod])),
        patch_max_visible_fraction=[r["patch_max_visible_fraction"] for r in prod],
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
        n, v = leakage_scan(d, set(meta["world_entity_ids"]), set())
        scanned += n
        violations += [f"{w}: {x}" for x in v]
    measured(GATE, "leakage", texts_scanned=scanned, violations=violations[:10])
    assert scanned > 100 and not violations


def test_bundle_replays_deterministically(flights):
    first = str(flights["seeds"][0])
    rep = replay_unity_run(REPO_ROOT / flights["dirs"][first][PRODUCTION], GATE_RUNS / "replay" / GATE)
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
