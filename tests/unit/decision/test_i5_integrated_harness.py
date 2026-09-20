"""M1-ACTION-E002 harness pieces that do not need a mission: seed partition, labels, rule/FSM baseline."""

import pytest

from conrad.decision import EGDC, DecisionConfig
from conrad.evaluation.decision_experiments.fixtures import make_belief, make_context, make_requirement, unc
from conrad.evaluation.decision_experiments.m1_action_integrated import (
    SPECS,
    RuleFSMPolicy,
    make_arm,
    matches,
    seeds_for,
)
from conrad.evaluation.partitions import (
    I5_DOMAIN,
    I5_V2_DOMAIN,
    Partition,
    PartitionAccessError,
    PartitionIntegrityError,
    Purpose,
    load,
    load_i5,
    load_i5_v2,
    load_nav,
    load_unity_gates,
    partition_of,
    purpose_scope,
    split,
    validate_i5,
    validate_i5_v2,
)
from conrad.schemas.decision import ActionType
from conrad.schemas.ids import IdFactory
from conrad.sim.mission.scenarios import SCENARIOS


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
