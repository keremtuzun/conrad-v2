"""Gate I4 behavioral criterion: the frozen production planner beats fixed, random and coverage views.

Criterion (fixed 2026-09-19, BEFORE any final-partition result existed):

* ACTIVE-MCBR-E002 (abstract occlusion world), FINAL_TEST partition, >= 20 independent worlds: for each of
  A-B1_fixed_inspection, A-B0_random and A-B2_coverage, the paired per-world benefit of PRODUCTION on
  ``mission_error_reduction`` (actual mission-relevant hidden-state error) AND on
  ``hidden_state_error_reduction`` (whole belief) has a 95 % bootstrap CI whose lower bound is above 0.
* ACTIVE-MCBR-E003 (integrated SURROGATE mission, FLAGSHIP-I4 family), FINAL_TEST partition: the same on the
  target's ``hidden_state_error_improvement``.
* Budgets are matched (same observation cap, energy/time caps, sensors, mission duration) and the artifacts
  must have been produced by the currently frozen production planner on the pinned partition file.

The test reads the stored final-evaluation artifacts; it fails when an artifact is missing or the criterion
is not met. It never re-runs anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conrad.active.production import load_frozen
from conrad.evaluation.partitions import PARTITIONS_SHA256
from conrad.settings import REPO_ROOT

ART = REPO_ROOT / "artifacts" / "experiments"
COMPARATORS = ("A-B1_fixed_inspection", "A-B0_random", "A-B2_coverage")
MIN_WORLDS = 20


def _artifact(experiment: str, name: str) -> dict:
    path = ART / experiment / name
    if not path.exists():
        pytest.fail(f"I4 evidence missing: {path} (run `conrad eval run --experiment {experiment}`)")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_provenance(art: dict) -> dict:
    frozen = load_frozen()
    assert art["frozen_planner"]["config_digest"] == frozen["config_digest"], (
        "artifact is from another planner"
    )
    part = art["partitions"]["final_test"]
    assert part["partition_digest"] == PARTITIONS_SHA256, "artifact used a different partition file"
    return part


def _assert_beats(part: dict, metrics: tuple[str, ...], min_worlds: int) -> None:
    failures = []
    for base in COMPARATORS:
        for m in metrics:
            row = part["paired_production_vs"][base][m]
            assert row["n_worlds"] >= min_worlds, f"only {row['n_worlds']} worlds"
            low = row["ci95"][0]
            if low is None or low <= 0:
                failures.append(f"{m} vs {base}: benefit {row['benefit_mean']:.4f}, CI95 {row['ci95']}")
    assert not failures, "I4 criterion not met:\n" + "\n".join(failures)


def test_i4_abstract_world_final_partition():
    part = _check_provenance(_artifact("ACTIVE-MCBR-E002", "active_mcbr_e002.json"))
    _assert_beats(part, ("mission_error_reduction", "hidden_state_error_reduction"), MIN_WORLDS)


@pytest.mark.xfail(
    strict=True,
    reason="GATE I4 = FAIL (docs/audits/MCBR_REEVALUATION.md): production planner loses to fixed/coverage in the "
    "integrated surrogate mission on the final partition. Strict: if this starts passing, update the gate record.",
)
def test_i4_integrated_surrogate_mission_final_partition():
    art = _artifact("ACTIVE-MCBR-E003", "active_mcbr_e003.json")
    assert art["evidence_kind"] == "SURROGATE"
    part = _check_provenance(art)
    _assert_beats(part, ("hidden_state_error_improvement",), MIN_WORLDS)
