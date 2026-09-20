"""Gate I5, integrated-mission half (python-kernel SURROGATE, FINAL seeds).

Iteration 2: reads M1-ACTION-E003 (``conrad eval run --experiment M1-ACTION-E003``, fresh final seeds of
``configs/eval/partitions_i5_v2.yaml``); a missing artifact fails. The spent iteration-1 record M1-ACTION-E002 is
kept on disk and pinned by ``test_e002_iteration1_record_is_unchanged``. The claim tests
at the bottom state what I5 needs. Where the stored final result does not support a claim, the test is a STRICT
xfail whose reason quotes the measured failure, so a silent improvement or regression both show up.
"""

import json
from collections import Counter

import pytest

from conrad.evaluation.decision_experiments.m1_action_integrated import BASELINES, PRIMARY, SPECS
from conrad.evaluation.partitions import I5_V2_DOMAIN, Partition, Purpose, split
from conrad.settings import REPO_ROOT

ARTIFACT = REPO_ROOT / "artifacts" / "experiments" / "M1-ACTION-E003" / "m1_action_e003.json"
E002 = REPO_ROOT / "artifacts" / "experiments" / "M1-ACTION-E002" / "m1_action_e002.json"


@pytest.fixture(scope="module")
def result():
    assert ARTIFACT.exists(), f"missing {ARTIFACT}; run `conrad eval run M1-ACTION-E002`"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_is_final_split_surrogate(result):
    assert result["partition"] == "final_test"
    assert "SURROGATE" in result["evidence_class"]
    assert result["data_status"] == "SYNTHETIC_ONLY"
    final = split(I5_V2_DOMAIN, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION).world_seeds
    assert result["partition_domain"] == I5_V2_DOMAIN
    assert result["seeds"] and set(result["seeds"]) <= set(final)


def test_every_scenario_and_arm_ran_on_every_final_seed(result):
    runs = Counter((r["scenario"], r["arm"]) for r in result["per_run"])
    assert set(result["scenarios"]) == set(SPECS)
    for sc in SPECS:
        for arm in (PRIMARY, *BASELINES):
            assert runs[(sc, arm)] == len(result["seeds"]), (sc, arm)
    assert {(r["scenario"], r["seed"]) for r in result["per_run"] if r["arm"] == PRIMARY} == {
        (sc, s) for sc in SPECS for s in result["seeds"]
    }


def test_uir_is_measured_on_the_e001_basis(result):
    rows = [r["driving"] for r in result["per_run"] if r["arm"] == PRIMARY]
    relied = sum(r["uir"]["relied_world_claims"] for r in rows)
    bad = sum(r["uir"]["relied_unsupported_claims"] for r in rows)
    pooled = result["closed_loop"][PRIMARY]["ALL"]["uir"]
    assert pooled["relied_world_claims"] == relied and pooled["relied_unsupported_claims"] == bad
    assert all(r["n_decisions"] > 0 for r in rows)


def test_baselines_ran_closed_loop_on_the_same_seeds(result):
    by_arm = {
        arm: {(r["scenario"], r["seed"]) for r in result["per_run"] if r["arm"] == arm}
        for arm in result["arms"]
    }
    assert set(result["arms"]) == {PRIMARY, *BASELINES}
    assert all(v == by_arm[PRIMARY] for v in by_arm.values())
    for arm in (PRIMARY, *BASELINES):
        assert result["mission_outcomes"][arm]["ALL"]["n"] == len(SPECS) * len(result["seeds"])


def test_replan_fires_inside_integrated_missions(result):
    """The planned-route producer makes REPLAN reachable in a real mission (it never was before)."""
    rows = [r for r in result["per_run"] if r["arm"] == PRIMARY and r["scenario"] == "I5-ROUTE-BLOCKED"]
    assert any(r["driving"]["actions"].get("REPLAN", 0) > 0 for r in rows)


def test_no_hard_constraint_violation_in_any_egdc_mission(result):
    assert result["closed_loop"][PRIMARY]["ALL"]["violations_total"] == 0


def test_nominal_over_escalation_is_measured(result):
    """The verdict counts over-escalations in every nominal scenario (I5-NOMINAL and I5-NOMINAL-READABLE)."""
    v = result["verdicts"]
    nominal = [sc for sc in result["scenarios"] if SPECS[sc].check_over_escalation]
    assert set(nominal) == {"I5-NOMINAL", "I5-NOMINAL-READABLE"}
    assert v["nominal_over_escalations"] == sum(
        result["closed_loop"][PRIMARY][sc]["over_escalations"] for sc in nominal
    )


def test_e002_iteration1_record_is_unchanged():
    """The spent iteration-1 final result stays as measured (409 nominal ESCALATE decisions)."""
    old = json.loads(E002.read_text(encoding="utf-8"))
    assert old["partition"] == "final_test" and old["seeds"][0] == 7600000
    assert old["verdicts"]["nominal_over_escalations"] == 409
    assert not old["verdicts"]["actions_exercised_correctly"]


# ------------------------------------------------------------------------------------------ gate claims
@pytest.mark.xfail(
    strict=True,
    reason=(
        "M1-ACTION-E003 FINAL (2026-09-19, I5 iteration 2): the nominal continue warrant (critical component "
        "OBSERVED INTACT) still never arose, 0/10 in I5-NOMINAL and 0/10 in I5-NOMINAL-READABLE, because "
        "Model2T needs 80 % surface coverage before a component-level condition is OBSERVED and the missions "
        "reach 0.6-0.7. Over-escalation is much smaller but not gone: 74 ESCALATE decisions in 20 nominal "
        "missions (E002: 409 in 10), all of them after the information attempts on an open critical item were "
        "used up, and 13 of the 20 missions have none. The other six scenarios meet the 0.9 floor except "
        "critical finding 8/10 and store-and-forward 8/10 (the finding was never observed on two seeds); "
        "route blocked 9/10, uncertain belief, battery and time 10/10; 0 hard-constraint violations."
    ),
)
def test_actions_exercised_correctly_inside_integrated_missions(result):
    assert result["verdicts"]["actions_exercised_correctly"], result["verdicts"]["per_scenario_correct_rate"]


def test_traceable_decisions_with_low_measured_uir(result):
    assert result["verdicts"]["traceable_low_uir"]


def test_competitive_mission_outcomes_vs_decision_baselines(result):
    assert result["verdicts"]["competitive_outcomes"], result["verdicts"]["competitive"]
