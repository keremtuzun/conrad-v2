"""M1-ACTION-E002 harness pieces that do not need a mission: seed partition, labels, rule/FSM baseline."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from conrad.decision import EGDC, DecisionConfig
from conrad.evaluation.decision_experiments.fixtures import make_belief, make_context, make_requirement, unc
from conrad.evaluation.decision_experiments.m1_action_integrated import (
    BASELINES,
    PRIMARY,
    SPECS,
    RuleFSMPolicy,
    _atomic_write_json,
    _load_checkpoint,
    _write_checkpoint,
    make_arm,
    matches,
    seeds_for,
    verdicts,
)
from conrad.evaluation.partitions import (
    I5_DOMAIN,
    I5_V2_DOMAIN,
    I5_V3_DOMAIN,
    I5_V4_DOMAIN,
    I5_V5_DOMAIN,
    Partition,
    PartitionAccessError,
    PartitionIntegrityError,
    Purpose,
    load,
    load_i5,
    load_i5_unity,
    load_i5_v2,
    load_i5_v3,
    load_i5_v4,
    load_i5_v5,
    load_nav,
    load_unity_gates,
    partition_of,
    purpose_scope,
    split,
    validate_i5,
    validate_i5_v2,
    validate_i5_v3,
    validate_i5_v4,
    validate_i5_v5,
)
from conrad.schemas.decision import ActionType
from conrad.schemas.ids import IdFactory
from conrad.sim.mission.scenarios import SCENARIOS


def test_completed_mission_rows_checkpoint_atomically_and_refuse_incompatible_resume(tmp_path):
    path = tmp_path / "checkpoint.json"
    declaration = {"sha256": "declaration-a", "experiment_id": "DEV"}
    rows = {
        "DEV-I5-NOMINAL-s7500000-egdc_structured": {
            "run_id": "DEV-I5-NOMINAL-s7500000-egdc_structured",
            "seed": 7500000,
            "scenario": "I5-NOMINAL",
            "arm": PRIMARY,
        }
    }
    _write_checkpoint(path, declaration, rows)
    assert not list(tmp_path.glob("*.tmp"))
    assert _load_checkpoint(path, declaration) == rows
    with pytest.raises(RuntimeError, match="incompatible checkpoint"):
        _load_checkpoint(path, {"sha256": "declaration-b"})


def test_atomic_json_write_retries_transient_windows_permission_error(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    original = Path.replace
    attempts = 0

    def flaky_replace(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("transient scanner lock")
        return original(source, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(
        "conrad.evaluation.decision_experiments.m1_action_integrated.time.sleep", lambda _: None
    )
    _atomic_write_json(path, {"complete": True})

    assert attempts == 2
    assert path.read_text(encoding="utf-8").strip() == '{\n "complete": true\n}'
    assert not list(tmp_path.glob("*.tmp"))


def test_i5_seeds_are_pinned_disjoint_and_guarded():
    raw = load_i5()["raw"]
    dev = seeds_for(Partition.DEVELOPMENT, Purpose.DESIGN)
    assert dev and partition_of(I5_DOMAIN, dev[0]) is Partition.DEVELOPMENT
    with pytest.raises(PartitionAccessError):
        split(I5_DOMAIN, Partition.FINAL_TEST, Purpose.DESIGN)
    with purpose_scope("design"), pytest.raises(PartitionAccessError):
        split(I5_DOMAIN, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION)
    bad = {**raw, "world_seeds": {**raw["world_seeds"], "final_test": {"explicit": [5300000]}}}
    with pytest.raises(Exception, match="collide"):
        validate_i5(bad, load()["raw"], load_nav()["raw"])


def test_every_spec_is_a_registered_scenario():
    assert set(SPECS) <= set(SCENARIOS)
    # v1 (M1-ACTION-E002) lists the original seven; v2 (M1-ACTION-E003) adds I5-NOMINAL-READABLE
    assert set(load_i5()["raw"]["scenarios"]) <= set(SPECS)
    assert set(load_i5_v2()["raw"]["scenarios"]) == set(SPECS)


def test_i5_v2_final_is_fresh_and_guarded():
    raw = load_i5_v2()["raw"]
    final = set(split(I5_V2_DOMAIN, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION).world_seeds)
    spent = set(split(I5_DOMAIN, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION).world_seeds)
    assert final and not final & spent
    assert partition_of(I5_V2_DOMAIN, 7600000) is None
    with pytest.raises(PartitionAccessError):
        split(I5_V2_DOMAIN, Partition.FINAL_TEST, Purpose.DESIGN)
    args = (load()["raw"], load_nav()["raw"], load_i5()["raw"], load_unity_gates()["raw"])
    for bad_seed in (7600003, 7800005, 7810001, 7300002, 7500001):  # spent I5, Unity gates, E001, v1 dev
        bad = {**raw, "world_seeds": {**raw["world_seeds"], "final_test": {"explicit": [bad_seed]}}}
        with pytest.raises(PartitionIntegrityError, match=r"collide|overlap"):
            validate_i5_v2(bad, *args)


def test_i5_v3_final_is_fresh_and_guarded():
    """M1-ACTION-E004 (I5 iteration 3): a third final split, disjoint from both spent ones."""
    raw = load_i5_v3()["raw"]
    final = set(split(I5_V3_DOMAIN, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION).world_seeds)
    spent = set(split(I5_DOMAIN, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION).world_seeds) | set(
        split(I5_V2_DOMAIN, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION).world_seeds
    )
    assert final and not final & spent
    assert partition_of(I5_V3_DOMAIN, 7900000) is None and partition_of(I5_V3_DOMAIN, 7600000) is None
    assert set(load_i5_v3()["raw"]["scenarios"]) == set(SPECS)
    with pytest.raises(PartitionAccessError):
        split(I5_V3_DOMAIN, Partition.FINAL_TEST, Purpose.DESIGN)
    args = (
        load()["raw"],
        load_nav()["raw"],
        load_i5()["raw"],
        load_i5_v2()["raw"],
        load_unity_gates()["raw"],
    )
    # spent v1 final, spent v2 final, Unity gates, E001 final, I4 occluded family, shared development
    for bad_seed in (7600003, 7900004, 7800005, 7300002, 8000201, 7500001):
        bad = {**raw, "world_seeds": {**raw["world_seeds"], "final_test": {"explicit": [bad_seed]}}}
        with pytest.raises(PartitionIntegrityError, match=r"collide|overlap"):
            validate_i5_v3(bad, *args)


def test_i5_v4_final_is_fresh_and_guarded():
    """M1-ACTION-E005 (I5 iteration 4): a fourth final split, disjoint from all three spent ones."""
    raw = load_i5_v4()["raw"]
    final = set(split(I5_V4_DOMAIN, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION).world_seeds)
    spent = {
        s
        for domain in (I5_DOMAIN, I5_V2_DOMAIN, I5_V3_DOMAIN)
        for s in split(domain, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION).world_seeds
    }
    assert final and not final & spent
    for spent_seed in (7600000, 7900000, 7700000):
        assert partition_of(I5_V4_DOMAIN, spent_seed) is None
    assert set(raw["scenarios"]) == set(SPECS)
    with pytest.raises(PartitionAccessError):
        split(I5_V4_DOMAIN, Partition.FINAL_TEST, Purpose.DESIGN)
    args = (
        load()["raw"],
        load_nav()["raw"],
        [load_i5()["raw"], load_i5_v2()["raw"], load_i5_v3()["raw"]],
        load_i5_unity()["raw"],
        load_unity_gates()["raw"],
    )
    # spent v1/v2/v3 finals, Unity gates, formal Unity I5 worlds, E001 final, I4 family, shared development
    for bad_seed in (7600003, 7900004, 7700005, 7800005, 7710002, 7300002, 8000201, 7500001):
        bad = {**raw, "world_seeds": {**raw["world_seeds"], "final_test": {"explicit": [bad_seed]}}}
        with pytest.raises(PartitionIntegrityError, match=r"collide|overlap"):
            validate_i5_v4(bad, *args)


def test_i5_v5_final_is_fresh_digest_pinned_and_guarded():
    """M1-ACTION-E007: post-repair final seeds are disjoint from every spent I5 split."""
    loaded = load_i5_v5()
    raw = loaded["raw"]
    final = set(split(I5_V5_DOMAIN, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION).world_seeds)
    older_domains = (I5_DOMAIN, I5_V2_DOMAIN, I5_V3_DOMAIN, I5_V4_DOMAIN)
    spent = {
        seed
        for domain in older_domains
        for seed in split(domain, Partition.FINAL_TEST, Purpose.FINAL_EVALUATION).world_seeds
    }
    assert final == set(range(7730000, 7730010))
    assert not final & spent
    assert loaded["digest"]
    for spent_seed in (7600000, 7900000, 7700000, 7720000):
        assert partition_of(I5_V5_DOMAIN, spent_seed) is None
    assert set(raw["scenarios"]) == set(SPECS)
    with pytest.raises(PartitionAccessError):
        split(I5_V5_DOMAIN, Partition.FINAL_TEST, Purpose.DESIGN)
    args = (
        load()["raw"],
        load_nav()["raw"],
        [load_i5()["raw"], load_i5_v2()["raw"], load_i5_v3()["raw"], load_i5_v4()["raw"]],
        load_i5_unity()["raw"],
        load_unity_gates()["raw"],
    )
    for bad_seed in (7600003, 7900004, 7700005, 7720002, 7800005, 7710002, 7300002, 8000201, 7500001):
        bad = {**raw, "world_seeds": {**raw["world_seeds"], "final_test": {"explicit": [bad_seed]}}}
        with pytest.raises(PartitionIntegrityError, match=r"collide|overlap"):
            validate_i5_v5(bad, *args)


def test_the_i5_nominal_exemption_is_withdrawn_at_head():
    """I5 iteration 4: the "warrant cannot arise" exemption is withdrawn, so the scenario is scored."""
    assert SPECS["I5-NOMINAL"].warrant_by_construction is True
    assert SPECS["I5-NOMINAL"].check_over_escalation is True
    assert "WITHDRAWN" in SPECS["I5-NOMINAL"].not_applicable_reason
    assert all(s.warrant_by_construction for s in SPECS.values())


def test_a_scenario_whose_warrant_cannot_arise_is_not_applicable_not_zero():
    """I5 iteration 3: ``warrant_by_construction=False`` reports NOT APPLICABLE, it does not score 0."""
    summary: dict[str, Any] = {
        "closed_loop": {
            PRIMARY: {
                "SCORED": {
                    "warrant_reached": 1,
                    "correct_rate": 1.0,
                    "correct_given_warrant": 1.0,
                    "over_escalations": 0,
                },
                "UNREACHABLE": {
                    "warrant_reached": 0,
                    "correct_rate": 0.0,
                    "correct_given_warrant": None,
                    "over_escalations": 0,
                },
                "ALL": {
                    "violations_total": 0,
                    "traceable_decisions": 10,
                    "decisions": 10,
                    "uir": {"unsupported_inference_rate": 0.0},
                },
            },
            **{b: {"ALL": {"violations_total": 0}} for b in BASELINES},
        },
        "mission_outcomes": {
            arm: {
                "ALL": {"task_success": 1, "safety_events_total": 0},
                "SCORED": {"task_success": 1},
                "UNREACHABLE": {"task_success": 0},
            }
            for arm in (PRIMARY, *BASELINES)
        },
    }
    specs = {
        "SCORED": SPECS["I5-UNCERTAIN-BELIEF"],
        "UNREACHABLE": SPECS["I5-UNCERTAIN-BELIEF"].__class__(
            "UNREACHABLE",
            "continue",
            "critical_intact",
            ("CONTINUE_MISSION:*",),
            frozenset(),
            "inspected_without_escalation",
            warrant_by_construction=False,
            not_applicable_reason="measured: the onset never occurs",
        ),
    }
    config = {"success_floor": 0.9, "uir_max": 0.0}
    with mock.patch.dict(SPECS, specs, clear=False):
        v = verdicts(summary, config, list(specs))
    assert v["scenarios_not_applicable"] == {"UNREACHABLE": "measured: the onset never occurs"}
    assert v["scenarios_scored_for_actions"] == ["SCORED"]
    assert v["per_scenario_correct_rate"]["UNREACHABLE"] == 0.0  # still reported
    assert v["per_scenario_correct_given_warrant_rate"]["UNREACHABLE"] is None
    assert v["actions_exercised_correctly"] is True
    # and a scored scenario below the floor still fails
    summary["closed_loop"][PRIMARY]["SCORED"]["correct_given_warrant"] = 0.5
    with mock.patch.dict(SPECS, specs, clear=False):
        assert verdicts(summary, config, list(specs))["actions_exercised_correctly"] is False

    # A scored scenario with no observed warrant is not silently exempted.
    summary["closed_loop"][PRIMARY]["SCORED"]["correct_given_warrant"] = 1.0
    summary["closed_loop"][PRIMARY]["UNREACHABLE"]["correct_given_warrant"] = None
    specs["UNREACHABLE"] = replace(specs["UNREACHABLE"], warrant_by_construction=True)
    with mock.patch.dict(SPECS, specs, clear=False):
        assert verdicts(summary, config, list(specs))["actions_exercised_correctly"] is False


def test_expected_label_matching():
    assert matches("REPLAN:ROUTE_BLOCKED", ("REPLAN:ROUTE_BLOCKED",))
    assert not matches("REPLAN:INFORMATION_ATTEMPTS_EXHAUSTED", ("REPLAN:ROUTE_BLOCKED",))
    assert matches("RETURN_TO_SAFE_STATE:", ("RETURN_TO_SAFE_STATE:*",))


def test_rule_fsm_retreats_below_battery_reserve_and_never_escalates():
    ids = IdFactory(51)
    b = make_belief(ids, uncertainty=unc(uo=0.9))
    req = make_requirement(ids, belief_ids=[b.belief_id])
    low = make_context(ids, [b], [req], battery=0.05)
    out = make_arm("rule_fsm", ids, DecisionConfig()).decide(low)
    assert out.record.chosen is not None and out.record.chosen.action_type is ActionType.RETURN_TO_SAFE_STATE
    ok = make_context(ids, [b], [req], battery=0.9)
    cfg = DecisionConfig()
    chosen = EGDC(ids, cfg, policy=RuleFSMPolicy(cfg)).decide(ok).record.chosen
    assert chosen is not None and chosen.action_type is not ActionType.ESCALATE_TO_OPERATOR
