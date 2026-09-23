"""I5 iteration 2 decision semantics (docs/audits/I5_ACTION_MATRIX.md, "Iteration 2")."""

from conrad.decision import EGDC, DecisionConfig, DecisionSummary
from conrad.decision.consequence import ConsequenceConfig
from conrad.evaluation.decision_experiments.fixtures import (
    make_belief,
    make_context,
    make_requirement,
    region,
    unc,
)
from conrad.schemas.decision import ActionType, InformationNeed, QuestionType
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Domain


def _requests(ids, target, n, executed, revision=0):
    return [
        DecisionSummary(
            decision_id=ids.new(),
            time_ns=0,
            action_type=ActionType.REQUEST_INFORMATION,
            target_belief_ids=(target,),
            executed=executed,
            target_revisions=(revision,),
        )
        for _ in range(n)
    ]


def _need(ids, target):
    return InformationNeed(
        need_id=ids.new(),
        trace_id=ids.new(),
        target_belief_ids=(target,),
        question_type=QuestionType.EXTEND_COVERAGE,
        target_properties=("condition",),
        priority=0.8,
    )


def test_deferred_requests_are_not_attempts():
    ids = IdFactory(71)
    b = make_belief(ids, uncertainty=unc(uo=0.9))
    req = make_requirement(ids, belief_ids=[b.belief_id])
    n = DecisionConfig().max_information_attempts
    deferred = make_context(ids, [b], [req], previous=_requests(ids, b.belief_id, n, executed=False))
    assert deferred.attempts_on((b.belief_id,), ActionType.REQUEST_INFORMATION) == 0
    chosen = EGDC(ids).decide(deferred).record.chosen
    assert chosen is not None and chosen.action_type is ActionType.REQUEST_INFORMATION
    done = make_context(ids, [b], [req], previous=_requests(ids, b.belief_id, n, executed=None))
    assert done.attempts_on((b.belief_id,), ActionType.REQUEST_INFORMATION) == n


def test_escalate_never_outranks_an_untried_autonomous_path():
    """Repeat-decayed information value still beats the operator while attempts remain (factor 0)."""
    assert ConsequenceConfig().operator_value_autonomous_factor == 0.0
    ids = IdFactory(72)
    b = make_belief(ids, uncertainty=unc(uo=0.9))
    req = make_requirement(ids, belief_ids=[b.belief_id])
    n = DecisionConfig().max_information_attempts - 1
    ctx = make_context(ids, [b], [req], previous=_requests(ids, b.belief_id, n, executed=True))
    out = EGDC(ids).decide(ctx).record
    ranks = [a.action_type for a in out.candidates]
    assert ranks.index(ActionType.REQUEST_INFORMATION) < ranks.index(ActionType.ESCALATE_TO_OPERATOR)


def test_active_matching_information_need_postpones_attempts_exhausted_escalation():
    ids = IdFactory(721)
    b = make_belief(ids, uncertainty=unc(uo=0.9))
    req = make_requirement(ids, belief_ids=[b.belief_id])
    n = DecisionConfig().max_information_attempts
    base = make_context(ids, [b], [req], previous=_requests(ids, b.belief_id, n, executed=True))
    active = base.model_copy(update={"active_information_needs": (_need(ids, b.belief_id),)})

    assert active.information_in_flight((b.belief_id,))
    chosen = EGDC(ids).decide(active).record.chosen
    assert chosen is not None and chosen.action_type is not ActionType.ESCALATE_TO_OPERATOR

    chosen_after_goal_closes = EGDC(ids).decide(base).record.chosen
    assert chosen_after_goal_closes is not None
    assert chosen_after_goal_closes.action_type is ActionType.ESCALATE_TO_OPERATOR


def test_in_flight_need_for_another_belief_does_not_suppress_escalation():
    ids = IdFactory(722)
    b = make_belief(ids, uncertainty=unc(uo=0.9))
    other = make_belief(ids)
    req = make_requirement(ids, belief_ids=[b.belief_id])
    n = DecisionConfig().max_information_attempts
    base = make_context(ids, [b, other], [req], previous=_requests(ids, b.belief_id, n, executed=True))
    ctx = base.model_copy(update={"active_information_needs": (_need(ids, other.belief_id),)})

    assert not ctx.information_in_flight((b.belief_id,))
    chosen = EGDC(ids).decide(ctx).record.chosen
    assert chosen is not None and chosen.action_type is ActionType.ESCALATE_TO_OPERATOR


def test_explicit_acquisition_unavailable_waits_instead_of_livelocking_or_escalating():
    ids = IdFactory(723)
    b = make_belief(ids, uncertainty=unc(uo=0.9))
    req = make_requirement(ids, belief_ids=[b.belief_id])
    n = DecisionConfig().max_information_attempts
    base = make_context(ids, [b], [req], previous=_requests(ids, b.belief_id, n, executed=True))
    ctx = base.model_copy(
        update={"unavailable_information_targets": {b.belief_id: "NO_FEASIBLE_OBSERVATION"}}
    )

    chosen = EGDC(ids).decide(ctx).record.chosen
    assert chosen is not None and chosen.action_type is ActionType.WAIT
    assert chosen.parameters["reason"] == "INFORMATION_ACQUISITION_UNAVAILABLE"


def test_uncalibrated_gap_is_closed_by_a_newer_direct_revision_after_an_executed_request():
    ids = IdFactory(73)
    b = make_belief(ids, uncertainty=unc(calibrated=False), revision=3)
    req = make_requirement(ids, belief_ids=[b.belief_id])
    fresh = make_context(ids, [b], [req], previous=_requests(ids, b.belief_id, 1, executed=True, revision=1))
    chosen = EGDC(ids).decide(fresh).record.chosen
    assert chosen is not None and chosen.action_type is ActionType.CONTINUE_MISSION
    for hist in (
        _requests(ids, b.belief_id, 1, executed=False, revision=1),  # deferred: never carried out
        _requests(ids, b.belief_id, 1, executed=True, revision=3),  # no newer revision since the request
    ):
        chosen = EGDC(ids).decide(make_context(ids, [b], [req], previous=hist)).record.chosen
        assert chosen is not None and chosen.action_type is not ActionType.CONTINUE_MISSION


def test_cross_domain_context_is_limited_to_the_requirement_region():
    ids = IdFactory(74)
    b = make_belief(ids)
    here = region((5.0, 5.0, 0.0))
    req = make_requirement(
        ids, belief_ids=[b.belief_id], context_domains=[Domain.SPATIAL], target_region=here
    )
    far = make_belief(
        ids,
        domain=Domain.SPATIAL,
        properties={"occupied": True},
        coverage=0.1,
        support=region((40.0, 40.0, 0.0)),
        uncertainty=unc(uo=0.9),
    )
    chosen = EGDC(ids).decide(make_context(ids, [b, far], [req])).record.chosen
    assert chosen is not None and chosen.action_type is ActionType.CONTINUE_MISSION
    near = make_belief(
        ids,
        domain=Domain.SPATIAL,
        properties={"occupied": True},
        coverage=0.1,
        support=here,
        uncertainty=unc(uo=0.9),
    )
    chosen = EGDC(ids).decide(make_context(ids, [b, near], [req])).record.chosen
    assert chosen is not None and chosen.action_type is not ActionType.CONTINUE_MISSION
