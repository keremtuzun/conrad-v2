"""Gate I5, integrated-mission half (python-kernel SURROGATE, FINAL seeds).

Repair iteration 6: reads M1-ACTION-E008 (``conrad eval run --experiment M1-ACTION-E008``, fresh final seeds of
``configs/eval/partitions_i5_v6.yaml``); a missing artifact fails. The spent iteration-1 and iteration-2 records
M1-ACTION-E002 and M1-ACTION-E003 are kept on disk and pinned by ``test_spent_iteration_records_are_unchanged``.
The iteration-3 NOT APPLICABLE exemption for ``I5-NOMINAL`` is WITHDRAWN at HEAD, so every scenario is scored.
The claim tests at the bottom state what I5 needs. Where the stored final result does not support a claim, the
test is a STRICT xfail whose reason quotes the measured failure, so a silent improvement or regression both
show up.
"""

import json
from collections import Counter

import pytest

from conrad.evaluation.decision_experiments.m1_action_integrated import BASELINES, PRIMARY, SPECS
from conrad.evaluation.partitions import I5_V6_DOMAIN, Partition, Purpose, split
from conrad.settings import REPO_ROOT

ARTIFACT = REPO_ROOT / "artifacts" / "experiments" / "M1-ACTION-E008" / "m1_action_e008.json"
E002 = REPO_ROOT / "artifacts" / "experiments" / "M1-ACTION-E002" / "m1_action_e002.json"
E003 = REPO_ROOT / "artifacts" / "experiments" / "M1-ACTION-E003" / "m1_action_e003.json"


@pytest.fixture(scope="module")
def result():
    assert ARTIFACT.exists(), f"missing {ARTIFACT}; run `conrad eval run M1-ACTION-E008`"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_is_final_split_surrogate(result):
    assert result["partition"] == "final_test"
    assert "SURROGATE" in result["evidence_class"]
    assert result["data_status"] == "SYNTHETIC_ONLY"
    final = split(I5_V6_DOMAIN, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION).world_seeds
    assert result["partition_domain"] == I5_V6_DOMAIN
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


def test_spent_iteration_records_are_unchanged():
    """The spent final results stay as measured: 409 nominal ESCALATE decisions in E002, 74 in E003."""
    old = json.loads(E002.read_text(encoding="utf-8"))
    assert old["partition"] == "final_test" and old["seeds"][0] == 7600000
    assert old["verdicts"]["nominal_over_escalations"] == 409
    assert not old["verdicts"]["actions_exercised_correctly"]
    prev = json.loads(E003.read_text(encoding="utf-8"))
    assert prev["partition"] == "final_test" and prev["seeds"][0] == 7900000
    assert prev["verdicts"]["nominal_over_escalations"] == 74
    assert not prev["verdicts"]["actions_exercised_correctly"]
    # the nominal continue warrant never arose under Model2T iteration 3, in either nominal scenario
    e = prev["closed_loop"][PRIMARY]
    assert e["I5-NOMINAL"]["warrant_reached"] == 0 and e["I5-NOMINAL-READABLE"]["warrant_reached"] == 0


def test_no_scenario_is_exempt_from_the_action_criterion(result):
    """Iteration 4 withdrew the I5-NOMINAL exemption: every declared scenario is scored, none is skipped.

    Iteration 3 exempted I5-NOMINAL on the measured claim that its warrant "cannot arise by construction".
    That claim is false at HEAD (the warrant arose on development seed 7500001), so the exemption is gone and
    the criterion is strictly harder to pass. The iteration-3 measurement is kept as ``not_applicable_reason``
    so the record of why the exemption existed is not lost.
    """
    assert result["verdicts"]["scenarios_not_applicable"] == {}
    assert set(result["verdicts"]["scenarios_scored_for_actions"]) == set(SPECS)
    for sc in SPECS:
        assert result["scenarios"][sc]["warrant_by_construction"] is True, sc
    assert len(result["scenarios"]["I5-NOMINAL"]["not_applicable_reason"]) > 200  # kept as the record
    e = result["closed_loop"][PRIMARY]["I5-NOMINAL"]
    assert e["n"] == len(result["seeds"]) and e["decisions"] > 0


# ------------------------------------------------------------------------------------------ gate claims
def test_actions_exercised_correctly_inside_integrated_missions(result):
    assert result["verdicts"]["actions_exercised_correctly"], result["verdicts"][
        "per_scenario_correct_given_warrant_rate"
    ]


def test_traceable_decisions_with_low_measured_uir(result):
    assert result["verdicts"]["traceable_low_uir"]


def test_competitive_mission_outcomes_vs_decision_baselines(result):
    assert result["verdicts"]["competitive_outcomes"], result["verdicts"]["competitive"]
