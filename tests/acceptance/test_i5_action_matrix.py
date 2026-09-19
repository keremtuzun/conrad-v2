"""I5 decision autonomy (spec ch25 L11835-11845) on the STORED final-test artifact of M1-ACTION-E001.

Criterion (the spec leaves the numerical recall bound OPEN): each required action is used correctly in its canonical
scenario class, with per-class recall reported; the floor is the ENGINEERING_ESTIMATE stored with the artifact.
Hard constraints: 0 violations, including invalid proposals injected directly at the ConstraintEngine.
The tests FAIL (they do not skip) when the artifact is missing. Regenerate with
``python -m uv run python -c "from conrad.evaluation.dispatch import run_experiment; run_experiment('M1-ACTION-E001')"``.
"""

import json
import pathlib

import pytest

from conrad.evaluation.decision_experiments.action_matrix import (
    INVALID_KINDS,
    SEEDS,
    Condition,
    scenario_ids,
)
from conrad.evaluation.partitions import Partition

ROOT = pathlib.Path(__file__).resolve().parents[2]
FOLDER = ROOT / "artifacts" / "experiments" / "M1-ACTION-E001"
FINAL = FOLDER / "m1_action_e001.json"
DEV = FOLDER / "m1_action_e001_development.json"
ARM = "egdc_structured"
MIN_PER_CONDITION = 50


def _load(path: pathlib.Path) -> dict:
    assert path.exists(), f"I5 evidence artifact missing: {path.relative_to(ROOT)} (run M1-ACTION-E001)"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def final() -> dict:
    return _load(FINAL)


def _recall_ok(final: dict, cls: str) -> None:
    s = final["summary"][ARM]["per_class"][cls]
    floor = final["acceptance"]["canonical_recall_floor"]
    print(f"I5 {cls!r}: canonical recall {s['recall']} over {s['n_canonical']} scenarios (floor {floor})")
    assert s["n_canonical"] >= MIN_PER_CONDITION
    assert s["recall"] is not None and s["recall"] >= floor
    matrix = final["summary"][ARM]["confusion_matrix"][cls]
    assert matrix[cls] >= floor * s["n_canonical"]  # the action itself was chosen, not a neighbour class


def test_artifact_is_final_split_and_large_enough(final):
    assert final["experiment_id"] == "M1-ACTION-E001"
    assert final["partition"] == Partition.FINAL_TEST.value and final["purpose"] == "final_evaluation"
    assert set(final["seeds"]) <= set(SEEDS[Partition.FINAL_TEST])
    assert not set(SEEDS[Partition.FINAL_TEST]) & set(SEEDS[Partition.DEVELOPMENT])
    per_condition = final["summary"][ARM]["per_condition"]
    for c in Condition:
        if c is Condition.INVALID_PROPOSAL:
            continue
        assert per_condition[c.value]["n"] >= MIN_PER_CONDITION, c
    hc = final["summary"][ARM]["hard_constraint_violations"]
    assert hc["injected_invalid_proposals"] >= MIN_PER_CONDITION
    assert set(hc["injected_by_kind"]) == set(INVALID_KINDS)


def test_final_scenarios_never_used_for_tuning(final):
    if DEV.exists():
        dev = _load(DEV)
        assert dev["partition"] == Partition.DEVELOPMENT.value
        assert not scenario_ids(dev) & scenario_ids(final)


def test_i5_continue(final):
    _recall_ok(final, "continue")


def test_i5_request_evidence(final):
    _recall_ok(final, "request evidence")


def test_i5_replan(final):
    _recall_ok(final, "replan")


def test_i5_change_sensing(final):
    _recall_ok(final, "change sensing")


def test_i5_return(final):
    _recall_ok(final, "return")


def test_i5_escalate(final):
    _recall_ok(final, "escalate")


def test_i5_hard_constraints_inviolable(final):
    s = final["summary"][ARM]
    hc = s["hard_constraint_violations"]
    assert hc["chosen_action_audit"] == 0, hc["chosen_action_audit_by_rule"]
    assert hc["injected_invalid_accepted"] == 0, hc["injected_accepted_kinds"]
    assert hc["total"] == 0
    assert s["uir"]["unsupported_inference_rate"] == 0.0
    assert s["uir"]["relied_world_claims"] > 0  # UIR = 0 is measured over real reliance, not vacuous
    rows = final["rows"][ARM]
    assert all(not r["violations"] for r in rows)
    motion = {"CONTINUE_MISSION", "REQUEST_INFORMATION", "REVISIT_REGION", "REPLAN"}
    held = [r for r in rows if r["variant"] == "motion_not_permitted"]
    assert held and all(r["chosen"].split(":")[0] not in motion for r in held)


def test_egdc_beats_naive_baseline(final):
    egdc, naive = final["summary"][ARM], final["summary"]["naive_act_on_claims"]
    assert egdc["correct_rate"] > naive["correct_rate"]
    assert (
        egdc["mission_outcome_proxy"]["safe_and_useful_rate"]
        > naive["mission_outcome_proxy"]["safe_and_useful_rate"]
    )
    assert egdc["uir"]["unsupported_inference_rate"] <= naive["uir"]["unsupported_inference_rate"]
