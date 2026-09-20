"""I5 action semantics added for M1-ACTION-E001: replan, change sensing, time reserve, report, calibration gap."""

from typing import Any

import pytest

from conrad.decision import EGDC, ClaimGraphBuilder, ConstraintEngine, DecisionConfig, DecisionSummary
from conrad.evaluation.decision_experiments.action_matrix import (
    INVALID_KINDS,
    SEEDS,
    Condition,
    ScenarioGenerator,
    action_label,
    invalid_case,
    partition_seeds,
    run_seed,
)
from conrad.evaluation.decision_experiments.fixtures import (
    make_belief,
    make_context,
    make_requirement,
    region,
    unc,
)
from conrad.evaluation.partitions import Partition, PartitionAccessError, Purpose
from conrad.schemas.comms import LinkStatus
from conrad.schemas.decision import ActionProposal, ActionType, ClaimEdgeType, GroundingStatus
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Domain


def _history(ids, target, kind, n):
    return [
        DecisionSummary(decision_id=ids.new(), time_ns=0, action_type=kind, target_belief_ids=(target,))
        for _ in range(n)
    ]


def _route_ctx(ids, obstacle_kw):
    t = make_belief(ids)
    obstacle = make_belief(
        ids,
        domain=Domain.SPATIAL,
        properties={"occupied": True},
        support=region((4.0, -3.0, 0.0), half=0.4),
        **obstacle_kw,
    )
    req = make_requirement(ids, belief_ids=[t.belief_id])
    leg = region((4.0, -3.0, 0.0), half=1.0).model_dump(mode="json")
    return make_context(ids, [t, obstacle], [req], notes={"planned_route": [leg]})


def test_grounded_obstacle_on_route_replans_and_blocks_continue():
    ids = IdFactory(31)
    out = EGDC(ids).decide(_route_ctx(ids, {}))
    chosen = out.record.chosen
    assert chosen is not None and chosen.action_type is ActionType.REPLAN
    assert chosen.parameters["reason"] == "ROUTE_BLOCKED"
    claims = {c.claim_id: c for c in out.record.claims}
    assert chosen.supporting_claims
    assert all(claims[c].grounding is GroundingStatus.GROUNDED for c in chosen.supporting_claims)
    cont = next(a for a in out.record.candidates if a.action_type is ActionType.CONTINUE_MISSION)
    assert any(
        e.edge_type is ClaimEdgeType.BLOCKS and e.target_claim_id == cont.action_id for e in out.record.edges
    )


@pytest.mark.parametrize(
    "obstacle_kw",
    [{"uncertainty": unc(uo=0.9)}, {"n_evidence": 0}, {"time_s": 10.0}],
    ids=["unobserved", "no_evidence", "stale"],
)
def test_ungrounded_or_uncertain_obstacle_does_not_replan(obstacle_kw):
    ids = IdFactory(32)
    out = EGDC(ids).decide(_route_ctx(ids, obstacle_kw))
    assert out.record.chosen is not None and out.record.chosen.action_type is not ActionType.REPLAN


def test_high_ua_repeats_first_then_changes_modality():
    ids = IdFactory(33)
    b = make_belief(ids, uncertainty=unc(ua=0.8))
    req = make_requirement(ids, belief_ids=[b.belief_id])
    first = EGDC(ids).decide(make_context(ids, [b], [req]))
    assert action_label(first.record.chosen) == "REQUEST_INFORMATION:IMPROVE_MEASUREMENT"
    hist = _history(ids, b.belief_id, ActionType.REQUEST_INFORMATION, 1)
    again = EGDC(ids).decide(make_context(ids, [b], [req], previous=hist))
    assert again.record.chosen is not None
    assert again.record.chosen.action_type is ActionType.CHANGE_SENSOR_MODE


def test_time_below_reserve_is_a_hard_constraint_and_returns():
    ids = IdFactory(34)
    cfg = DecisionConfig()
    b = make_belief(ids)
    req = make_requirement(ids, belief_ids=[b.belief_id])
    ctx = make_context(ids, [b], [req])
    assert ctx.resource_state is not None
    low = ctx.resource_state.model_copy(update={"time_remaining_s": cfg.constraints.time_reserve_s / 2})
    ctx = ctx.model_copy(update={"resource_state": low})
    graph = ClaimGraphBuilder(ids, cfg).build(ctx)
    engine = ConstraintEngine(cfg)
    cont = ActionProposal(action_id=ids.new(), action_type=ActionType.CONTINUE_MISSION)
    assert "TIME_BELOW_RESERVE" in engine.check(cont, graph, ctx).reason_codes
    ret = ActionProposal(action_id=ids.new(), action_type=ActionType.RETURN_TO_SAFE_STATE)
    assert engine.check(ret, graph, ctx).accepted
    chosen = EGDC(ids).decide(ctx).record.chosen
    assert chosen is not None and chosen.action_type is ActionType.RETURN_TO_SAFE_STATE


def test_unknown_time_remaining_never_triggers_the_time_reserve():
    ids = IdFactory(35)
    b = make_belief(ids)
    ctx = make_context(ids, [b], [make_requirement(ids, belief_ids=[b.belief_id])])
    assert ctx.resource_state is not None and ctx.resource_state.time_remaining_s is None
    chosen = EGDC(ids).decide(ctx).record.chosen
    assert chosen is not None and chosen.action_type is ActionType.CONTINUE_MISSION


def test_critical_finding_on_down_link_is_stored_once_then_mission_continues():
    ids = IdFactory(36)
    b = make_belief(ids, properties={"condition": "DAMAGED"})
    req = make_requirement(ids, belief_ids=[b.belief_id])
    kw: dict[str, Any] = {
        "link_status": LinkStatus.DOWN,
        "operator_reachable": False,
        "notes": {"pending_report_belief_ids": [str(b.belief_id)]},
    }
    first = EGDC(ids).decide(make_context(ids, [b], [req], **kw)).record.chosen
    assert first is not None and first.action_type is ActionType.STORE_AND_FORWARD
    hist = _history(ids, b.belief_id, ActionType.STORE_AND_FORWARD, 1)
    kw["notes"] = {"pending_report_belief_ids": [str(b.belief_id)]}
    later = EGDC(ids).decide(make_context(ids, [b], [req], previous=hist, **kw)).record.chosen
    assert later is not None and later.action_type is ActionType.CONTINUE_MISSION


@pytest.mark.parametrize("modalities", [("SONAR",), ("SONAR", "RGB")])
def test_uncalibrated_source_is_confirmed_not_escalated(modalities):
    ids = IdFactory(37)
    b = make_belief(ids, uncertainty=unc(calibrated=False))
    req = make_requirement(ids, belief_ids=[b.belief_id], consequence=0.9)
    ctx = make_context(ids, [b], [req], modalities=modalities, notes={"modalities_used": ["SONAR"]})
    out = EGDC(ids).decide(ctx)
    assert action_label(out.record.chosen) == "REQUEST_INFORMATION:CONFIRM_CONDITION"
    assert out.record.chosen is not None and out.record.chosen.parameters["calibration_check"] is True


def test_uncalibrated_source_escalates_once_attempts_are_exhausted():
    ids = IdFactory(38)
    b = make_belief(ids, uncertainty=unc(calibrated=False))
    req = make_requirement(ids, belief_ids=[b.belief_id], consequence=0.9)
    hist = _history(
        ids, b.belief_id, ActionType.REQUEST_INFORMATION, DecisionConfig().max_information_attempts
    )
    ctx = make_context(
        ids, [b], [req], modalities=("SONAR",), notes={"modalities_used": ["SONAR"]}, previous=hist
    )
    chosen = EGDC(ids).decide(ctx).record.chosen
    assert chosen is not None and chosen.action_type is ActionType.ESCALATE_TO_OPERATOR


def test_true_ood_without_alternate_still_escalates():
    ids = IdFactory(39)
    b = make_belief(ids, uncertainty=unc(ue=0.8, calibrated=False))
    req = make_requirement(ids, belief_ids=[b.belief_id], consequence=0.9)
    ctx = make_context(ids, [b], [req], modalities=("SONAR",), notes={"modalities_used": ["SONAR"]})
    chosen = EGDC(ids).decide(ctx).record.chosen
    assert chosen is not None and chosen.action_type is ActionType.ESCALATE_TO_OPERATOR


def test_scenario_generator_is_deterministic():
    def once():
        gen = ScenarioGenerator(SEEDS[Partition.DEVELOPMENT][0])
        return [
            gen.build(c, i).ctx.canonical_json()
            for c in (Condition.HIGH_UA, Condition.ROUTE_BLOCKED)
            for i in (0, 1)
        ]

    assert once() == once()
    a = run_seed(SEEDS[Partition.DEVELOPMENT][1], 2)
    b = run_seed(SEEDS[Partition.DEVELOPMENT][1], 2)
    assert a["rows"] == b["rows"] and a["injected"] == b["injected"]


def test_tuning_cannot_read_final_seeds():
    assert partition_seeds(Partition.DEVELOPMENT, Purpose.TUNING)
    with pytest.raises(PartitionAccessError):
        partition_seeds(Partition.FINAL_TEST, Purpose.TUNING)
    with pytest.raises(PartitionAccessError):
        partition_seeds(Partition.DEVELOPMENT, Purpose.FINAL_EVALUATION)


def test_every_injected_invalid_kind_is_rejected():
    gen = ScenarioGenerator(SEEDS[Partition.DEVELOPMENT][2])
    engine = ConstraintEngine(gen.cfg)
    for i, kind in enumerate(INVALID_KINDS):
        case = invalid_case(gen, kind, i)
        verdict = engine.check(case.action, case.graph, case.ctx, case.consequence)
        assert not verdict.accepted and verdict.reason_codes, kind
