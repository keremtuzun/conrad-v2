"""Gate I4, PRE-REGISTERED REPLICATION (run 2) of the two criteria formal run 1 left unresolved.

Run 1 (``test_i4_occluded_unity.py``, final_test 8000200-8000219, four arms) recorded gate I4 = FAIL: it beat
fixed views and random views with CIs above 0, and did not beat coverage-only
(+0.163 [-0.045, +0.369]) or beat it on information per time and per energy. The power warning in
``docs/audits/I4_WORLD_FAMILY.md`` section 8.2 was written before run 1, so the honest response is more worlds
under an identical design, declared in advance.

Everything is identical to run 1 except the two things declared in ``configs/eval/i4_occluded_unity_rep2.yaml``
before the first flight: fresh digest-pinned worlds (8001000-8001059, 60 worlds) and a two-arm set. The frozen
planner (``configs/active/mcbr_frozen_v3.yaml``), the family digest, the scenario, the candidate generator,
the feasibility filter, the budgets, the metric, the bootstrap and its seed are unchanged. Run 1's worlds are
SPENT and are neither re-run nor re-scored; both runs are reported side by side.

The fixed-views and random-views criteria are NOT re-decided here. They passed in run 1 on their own held-out
worlds, and re-running a criterion that already passed until it passes again is the shopping this
pre-registration exists to prevent.
"""

from __future__ import annotations

import faulthandler
import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import yaml
from test_i4_occluded_unity import SCENARIO, fly
from unity_gate_support import GATE_RUNS, GATES

from conrad.active.production import PRODUCTION, load_frozen
from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.active_mcbr_i4_occluded import FAMILY_ID
from conrad.evaluation.decision_experiments.active_mcbr_reeval import score_mission_run
from conrad.evaluation.metrics import paired_seed_comparison
from conrad.settings import REPO_ROOT

GATE = "I4"
CONFIG = REPO_ROOT / "configs" / "eval" / "i4_occluded_unity_rep2.yaml"
RUN1 = GATES / GATE / "unity_i4_occluded_results.json"
RESULTS = GATES / GATE / "unity_i4_occluded_rep2_results.json"
COVERAGE = "A-B2_coverage"
HARD_TIMEOUT_S = 8 * 3600.0


@pytest.fixture(autouse=True)
def _hard_timeout() -> Iterator[None]:  # overrides the conftest guard for this module only
    faulthandler.dump_traceback_later(HARD_TIMEOUT_S, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def config() -> dict[str, Any]:
    return dict(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


def replication_worlds(cfg: dict[str, Any]) -> list[int]:
    """The declared replication split, read once, and disjoint from run 1's spent worlds."""
    lo, hi = cfg["final_worlds"]["range"]
    declared = list(range(int(lo), int(hi)))
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        split = P.split(P.I4_OCCLUDED_V2_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION)
        spent = set(
            P.split(P.I4_OCCLUDED_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION).world_seeds
        )
    stray = sorted(set(declared) - set(split.world_seeds))
    if stray or not declared or set(declared) & spent:
        raise P.PartitionAccessError(
            f"declared replication worlds are not the fresh final split: {stray[:10]}"
        )
    return declared


@pytest.fixture(scope="module")
def flights() -> dict[str, Any]:
    cfg = config()
    arms = tuple(cfg["arm_set"]["arms"])
    assert arms == (PRODUCTION, COVERAGE), arms
    seeds = replication_worlds(cfg)
    root = GATE_RUNS / "I4-OCCLUDED-REP2"
    per_world: dict[str, dict[str, Any]] = {}
    dirs: dict[str, dict[str, str]] = {}
    for seed in seeds:  # strictly sequential: one Unity player at a time
        for planner in arms:
            d = fly(SCENARIO, seed, planner, root)
            per_world.setdefault(str(seed), {})[planner] = score_mission_run(
                d, seed, FAMILY_ID, planner, float(cfg["runtime"]["duration_s"])
            )
            dirs.setdefault(str(seed), {})[planner] = str(d.relative_to(REPO_ROOT))
    frozen = load_frozen(REPO_ROOT / cfg["frozen_planner_file"])
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        digest = P.load_i4_occluded_v2()["digest"]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "scenario": SCENARIO,
                "evidence_kind": "FORMAL",
                "run": 2,
                "replicates": cfg["replicates"],
                "run1_results": cfg["run1_results"],
                "run1_verdict": cfg["run1_verdict"],
                "world_family": FAMILY_ID,
                "partition_file": cfg["partition_file"],
                "partition_digest": digest,
                "family_digest": cfg["family_digest"],
                "world_seeds": seeds,
                "arm_set": cfg["arm_set"],
                "runtime": cfg["runtime"],
                "decision_rule": cfg["decision_rule"],
                "frozen_planner": {
                    "file": cfg["frozen_planner_file"],
                    "selected": frozen["planner"]["selected"],
                    "config_digest": frozen["config_digest"],
                    "mission_predictive_digest": frozen.get("mission_predictive_digest"),
                },
                "run_dirs": dirs,
                "per_world": per_world,
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    return {"per_world": per_world, "seeds": seeds, "cfg": cfg}


def paired(per_world: dict[str, dict[str, Any]], metric: str) -> dict[str, Any]:
    b = {int(w): float(per_world[w][COVERAGE][metric]) for w in per_world}
    c = {int(w): float(per_world[w][PRODUCTION][metric]) for w in per_world}
    pc = paired_seed_comparison(
        b, c, np.random.default_rng(20260919), metric=metric, direction="higher", n_resamples=4000
    )
    return {
        "benefit_mean": pc.benefit.point,
        "ci95": [pc.benefit.low, pc.benefit.high],
        "n_worlds": pc.benefit.n,
        "production_mean": pc.candidate_mean,
        "coverage_mean": pc.baseline_mean,
        "fraction_improved": pc.fraction_of_seeds_improved,
    }


def test_replication_is_declared_and_disjoint_from_run_1(flights):
    """The replication is evidence only when it is pinned, fresh and identical to run 1 in every other way."""
    cfg = flights["cfg"]
    run1 = json.loads(RUN1.read_text(encoding="utf-8"))
    assert cfg["run1_verdict"] == "FAIL"
    assert set(flights["seeds"]).isdisjoint(run1["world_seeds"]), "run 1 worlds must not be re-flown"
    assert cfg["partition_digest"] == P.I4_OCCLUDED_V2_PARTITIONS_SHA256
    assert cfg["family_digest"] == run1["family_digest"]
    assert cfg["runtime"] == run1["runtime"]
    assert cfg["arm_set"]["reduced"] and cfg["arm_set"]["dropped_arms"]
    frozen = load_frozen(REPO_ROOT / cfg["frozen_planner_file"])
    assert frozen["config_digest"] == run1["frozen_planner"]["config_digest"]


def test_beats_coverage_only(flights):
    row = paired(flights["per_world"], "hidden_state_error_improvement")
    low = row["ci95"][0]
    assert low is not None and low > 0.0, (
        f"does not beat coverage-only: {row['benefit_mean']:+.4f} {row['ci95']}"
    )


@pytest.mark.parametrize("metric", ["info_per_time", "info_per_kj"])
def test_beats_coverage_on_information_per_time_and_energy(flights, metric):
    row = paired(flights["per_world"], metric)
    low = row["ci95"][0]
    assert low is not None and low > 0.0, f"{metric}: {row['benefit_mean']:+.6f} {row['ci95']}"
