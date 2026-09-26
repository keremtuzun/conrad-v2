"""Gate I4, FORMAL RUN 4: fresh-sample resolution of the frozen V4 information/energy criterion.

Declared in ``configs/eval/i4_v4_unity_final_r4.yaml`` BEFORE any flight on these worlds, and designed and
selected entirely on the development and validation splits of ``configs/eval/partitions_i4_mcbr_v4.yaml``
(``docs/audits/MCBR_V4.md``). The mechanism is frozen in ``configs/active/mcbr_frozen_v4.yaml``.

Runs 1, 2 and 3 stand unchanged and remain FAIL. This module neither re-flies nor re-scores them. Run 4 uses
60 fresh seeds outside those spent blocks and flies only the four gate arms named by the I4 criteria.

Budgets are matched for every arm exactly as in runs 1 and 2: mission duration 100 s, ``max_plans_per_need``
4, one structural sensor, the same candidate generator, feasibility filter and PlanningRequest, Twin2E off,
and measured energy and navigation accounting. Scores are the target's actual hidden-state error against Twin
truth, read on the truth side after the run.
"""

from __future__ import annotations

import faulthandler
import hashlib
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
from conrad.evaluation.decision_experiments.i4_view_execution import (
    _first_direct_revision_s,
    execution_row,
)
from conrad.evaluation.metrics import paired_seed_comparison
from conrad.orchestration.mission_config import runtime_config
from conrad.settings import REPO_ROOT
from conrad.sim.mission.run import settings_for
from conrad.sim.mission.unity_run import player_identity, prepare_unity, replay_unity_run, resolve_unity

GATE = "I4"
SCENARIO = "I4-OCCLUDED-UNITY"
CONFIG = REPO_ROOT / "configs" / "eval" / "i4_v4_unity_final_r4.yaml"
FROZEN = REPO_ROOT / "configs" / "active" / "mcbr_frozen_v4.yaml"
FIXED, RANDOM, COVERAGE = "A-B1_fixed_inspection", "A-B0_random", "A-B2_coverage"
RESULTS = GATES / GATE / "unity_i4_v4_final_r4_results.json"
RUN1 = GATES / GATE / "unity_i4_occluded_results.json"
RUN2 = GATES / GATE / "unity_i4_occluded_rep2_results.json"
RUN3 = GATES / GATE / "unity_i4_v4_final_results.json"
# 240 sequential 100 s Unity flights at roughly 90 s per flight: about 6 h, with guard slack.
HARD_TIMEOUT_S = 12 * 3600.0


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
    if arms[0] != PRODUCTION or {FIXED, RANDOM, COVERAGE} - set(arms):
        raise AssertionError(f"the declared arm set must be PRODUCTION plus the three comparators: {arms}")
    return arms


def final_worlds(cfg: dict[str, Any]) -> list[int]:
    """The declared 60 fresh worlds, read once and checked against every named forbidden block."""
    lo, hi = cfg["final_worlds"]["range"]
    declared = list(range(int(lo), int(hi)))
    forbidden: set[int] = set()
    for block in cfg.get("forbidden_sources", []):
        blo, bhi = block["range"]
        forbidden.update(range(int(blo), int(bhi)))
    overlap = sorted(set(declared) & forbidden)
    if overlap or not declared or len(declared) != int(cfg["final_worlds"]["n"]):
        raise P.PartitionAccessError(f"declared I4 run-4 worlds are not fresh: {overlap[:10]}")
    return declared


def config_digest() -> str:
    return hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def fly(scenario: str, seed: int, arm: str, override: dict[str, Any], root: Path) -> Path:
    cfg = config()
    settings = settings_for(MISSION_CONFIG)
    settings.sim.setdefault("unity", {}).update(cfg.get("unity_runtime", {}))
    _, runtime_raw = resolve_unity(scenario, dict(settings.sim.get("mission", {})))
    rcfg = runtime_config({**runtime_raw, **override})
    run_id = f"{scenario}-s{seed}-{arm}"
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


def score(d: Path, seed: int, arm: str, duration: float) -> dict[str, Any]:
    base = score_mission_run(d, seed, FAMILY_ID, arm, duration)
    base["first_direct_revision_s"] = _first_direct_revision_s(d)
    row = execution_row(d, base)
    row["planner"] = arm
    return row


@pytest.fixture(scope="module")
def flights() -> dict[str, Any]:
    cfg = config()
    arms = planners(cfg)
    overrides = dict(cfg["arm_runtime"])
    seeds = final_worlds(cfg)
    duration = float(cfg["runtime"]["duration_s"])
    root = GATE_RUNS / "I4-V4-FINAL-R4"
    per_world: dict[str, dict[str, Any]] = {}
    dirs: dict[str, dict[str, str]] = {}
    for seed in seeds:  # strictly sequential: one Unity player at a time
        for arm in arms:
            d = fly(SCENARIO, seed, arm, overrides[arm], root)
            per_world.setdefault(str(seed), {})[arm] = score(d, seed, arm, duration)
            dirs.setdefault(str(seed), {})[arm] = str(d.relative_to(REPO_ROOT))
    frozen = load_frozen(FROZEN)
    digest = config_digest()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "scenario": SCENARIO,
                "evidence_kind": "FORMAL",
                "run": 4,
                "mechanism": "MCBR V4 view execution protocol (docs/audits/MCBR_V4.md)",
                "run1_results": cfg["run1_results"],
                "run2_results": cfg["run2_results"],
                "run3_results": cfg["run3_results"],
                "run1_verdict": cfg["run1_verdict"],
                "run2_verdict": cfg["run2_verdict"],
                "run3_verdict": cfg["run3_verdict"],
                "world_family": FAMILY_ID,
                "partition_file": cfg["partition_file"],
                "partition_digest": digest,
                "family_digest": cfg["family_digest"],
                "world_seeds": seeds,
                "forbidden_sources": cfg["forbidden_sources"],
                "arm_set": arm_set(cfg),
                "arm_runtime": overrides,
                "runtime": cfg["runtime"],
                "run_once": cfg["run_once"],
                "decision_rule": cfg["decision_rule"],
                "frozen_planner": {
                    "file": str(FROZEN.relative_to(REPO_ROOT)),
                    "selected": frozen["planner"]["selected"],
                    "config_digest": frozen["config_digest"],
                    "mission_predictive_digest": frozen.get("mission_predictive_digest"),
                    "view_execution_digest": frozen.get("view_execution_digest"),
                    "view_execution": frozen.get("view_execution"),
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
            "run": 4,
            "world_family": FAMILY_ID,
            "worlds": seeds,
            "arm_set": arm_set(cfg),
            "partition": f"{cfg['partition_file']} final_test 60 fresh worlds (declared in {CONFIG.name})",
            "partition_digest": digest,
            "mechanism": "MCBR V4 view execution protocol, frozen in configs/active/mcbr_frozen_v4.yaml",
            "earlier_runs": {
                "run1": {"results": cfg["run1_results"], "verdict": cfg["run1_verdict"]},
                "run2": {"results": cfg["run2_results"], "verdict": cfg["run2_verdict"]},
                "run3": {"results": cfg["run3_results"], "verdict": cfg["run3_verdict"]},
            },
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


REPORT = (
    "hidden_state_error_improvement",
    "info_per_time",
    "info_per_kj",
    "observations",
    "fraction_views_flown",
    "fraction_views_reached",
    "redundant_observations",
    "travel_m",
    "energy_j",
    "time_s",
    "target_observed",
    "corrosion_abs_error_m",
    "crack_abs_error_m",
    "collisions",
    "defect_read",
    "views_before_detection",
    "time_to_detection_s",
    "empty_plan_rate",
    "planning_latency_ms",
)


def test_every_arm_is_reported_on_every_metric(flights):
    """Every arm, every declared metric, plus the post-detection severity error over detected worlds only."""
    pw = flights["per_world"]
    table: dict[str, dict[str, float]] = {}
    for arm in flights["arms"]:
        rows = [pw[w][arm] for w in pw]
        read = [r for r in rows if r["defect_read"] > 0.5]
        table[arm] = {m: float(np.mean([r[m] for r in rows])) for m in REPORT}
        table[arm]["worlds_read"] = float(len(read))
        for q in ("corrosion_abs_error_m", "crack_abs_error_m"):
            table[arm][f"post_detection_{q}"] = float(np.mean([r[q] for r in read])) if read else float("nan")
    measured(GATE, "per arm", n_worlds=len(pw), metrics=table)
    assert set(table) == set(flights["arms"])
    assert all(np.isfinite(table[a]["hidden_state_error_improvement"]) for a in table)


def test_declared_arm_set_is_recorded(flights):
    """A reduced arm set is evidence only when it is recorded as reduced, with the arms it drops named."""
    a = arm_set(flights["cfg"])
    measured(
        GATE,
        "arm set",
        name=a["name"],
        reduced=a["reduced"],
        arms=a["arms"],
        gate_arms=a["gate_arms"],
        control_arms=a["control_arms"],
        dropped_arms=a["dropped_arms"],
        reason=a["reason"],
        dropped_arms_evidence=a["dropped_arms_evidence"],
        flights=len(flights["seeds"]) * len(flights["arms"]),
        world_family=FAMILY_ID,
        run_once=flights["cfg"]["run_once"],
    )
    assert a["reduced"] and a["dropped_arms"], "a reduced arm set must name the arms it drops"
    assert set(a["arms"]).isdisjoint(a["dropped_arms"])
    assert {FIXED, RANDOM, COVERAGE} <= set(a["arms"])  # no criterion may depend on a dropped arm
    assert set(a["control_arms"]).isdisjoint(a["gate_arms"])
    assert len(flights["seeds"]) * len(flights["arms"]) == a["flights"]["final"]


def test_run4_is_declared_fresh_and_does_not_touch_the_earlier_runs(flights):
    """Run 4 is evidence only if its worlds are fresh, its size was fixed in advance and the rest stay unread."""
    cfg = flights["cfg"]
    run1 = json.loads(RUN1.read_text(encoding="utf-8"))
    run2 = json.loads(RUN2.read_text(encoding="utf-8"))
    run3 = json.loads(RUN3.read_text(encoding="utf-8"))
    seeds = set(flights["seeds"])
    forbidden = {
        s
        for block in cfg["forbidden_sources"]
        for s in range(int(block["range"][0]), int(block["range"][1]))
    }
    measured(
        GATE,
        "declaration",
        run=4,
        n_worlds=len(seeds),
        declared_before_flight=str(CONFIG.relative_to(REPO_ROOT)),
        forbidden_sources=cfg["forbidden_sources"],
        run1_verdict=cfg["run1_verdict"],
        run2_verdict=cfg["run2_verdict"],
        run3_verdict=cfg["run3_verdict"],
        run_once=cfg["run_once"],
    )
    assert cfg["run1_verdict"] == "FAIL" and cfg["run2_verdict"] == "FAIL" and cfg["run3_verdict"] == "FAIL"
    assert seeds.isdisjoint(run1["world_seeds"]) and seeds.isdisjoint(run2["world_seeds"])
    assert seeds.isdisjoint(run3["world_seeds"]) and seeds.isdisjoint(forbidden)
    assert len(seeds) == int(cfg["final_worlds"]["n"])
    assert cfg["partition_digest"] == "DECLARED_BY_THIS_FILE"
    assert cfg["family_digest"] == run1["family_digest"]
    assert cfg["runtime"] == run1["runtime"] == run2["runtime"], "budgets must match runs 1 and 2"


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
        view_execution_digest=data["frozen_planner"]["view_execution_digest"],
        player_exe_sha256=data["player"].get("player_exe_sha256"),
    )
    assert data["partition_digest"] == config_digest()
    assert data["frozen_planner"]["config_digest"] == frozen["config_digest"]
    assert data["frozen_planner"]["view_execution_digest"] == frozen["view_execution_digest"]
    assert data["frozen_planner"]["view_execution"]["enabled"] is True
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
        fraction_of_accepted_views_flown=float(np.mean([r["fraction_views_flown"] for r in prod])),
        patch_max_visible_fraction=[r["patch_max_visible_fraction"] for r in prod],
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
    for w, by_arm in flights["dirs"].items():
        d = REPO_ROOT / by_arm[PRODUCTION]
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
        REPO_ROOT / flights["dirs"][first][PRODUCTION], GATE_RUNS / "replay" / "I4-V4-FINAL-R4"
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
