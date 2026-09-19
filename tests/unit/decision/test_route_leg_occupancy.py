"""Planned-route blockers in the Model2S map-block form (integrated missions, M1-ACTION-E002).

A Model2S block belief has no boolean ``occupied`` claim; it blocks a route leg only when the runtime's
cell-level check lists it for that leg in ``notes["route_leg_occupancy"]``.
"""

from conrad.decision import EGDC
from conrad.decision.route import route_blockers
from conrad.evaluation.decision_experiments.fixtures import (
    make_belief,
    make_context,
    make_requirement,
    region,
)
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.decision import ActionType
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Domain


def _ctx(ids, confirm: bool, leg_index: int = 0, **block_kw):
    t = make_belief(ids)
    block = make_belief(
        ids,
        domain=Domain.SPATIAL,
        properties={"occupancy.observed": 0.2, "coverage": 0.6},
        support=region((4.0, -3.0, 0.0), half=1.4),
        **block_kw,
    )
    req = make_requirement(ids, belief_ids=[t.belief_id])
    legs = [region((4.0, -3.0, 0.0), half=0.45).model_dump(mode="json")]
    notes = {
        "planned_route": legs,
        "route_leg_occupancy": {str(leg_index): [str(block.belief_id)]} if confirm else {},
    }
    return make_context(ids, [t, block], [req], notes=notes)


def test_confirmed_block_on_a_leg_replans():
    ids = IdFactory(41)
    ctx = _ctx(ids, confirm=True)
    assert len(route_blockers(ctx, EGDC(ids).config)) == 1
    out = EGDC(ids).decide(ctx)
    assert out.record.chosen is not None and out.record.chosen.action_type is ActionType.REPLAN


def test_block_overlap_alone_is_not_an_obstacle():
    """A 2.8 m map block overlapping the corridor usually holds seabed or pipe cells: no confirmation, no replan."""
    ids = IdFactory(42)
    ctx = _ctx(ids, confirm=False)
    assert route_blockers(ctx, EGDC(ids).config) == []


def test_confirmation_for_another_leg_does_not_count():
    ids = IdFactory(43)
    assert route_blockers(_ctx(ids, confirm=True, leg_index=3), EGDC(ids).config) == []


def test_unevidenced_or_stale_block_does_not_block():
    ids = IdFactory(44)
    assert route_blockers(_ctx(ids, confirm=True, n_evidence=0), EGDC(ids).config) == []
    ids = IdFactory(45)
    assert route_blockers(_ctx(ids, confirm=True, time_s=10.0), EGDC(ids).config) == []


def test_unknown_status_block_does_not_block():
    ids = IdFactory(46)
    assert route_blockers(_ctx(ids, confirm=True, status=KnowledgeStatus.UNKNOWN), EGDC(ids).config) == []
