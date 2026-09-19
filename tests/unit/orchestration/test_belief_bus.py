"""Belief Bus: revision monotonicity, staleness, availability, failure isolation, cross-domain context."""

from conrad.evaluation.decision_experiments.fixtures import make_belief
from conrad.orchestration.belief_bus import (
    REASON_FOREIGN_WRITE,
    REASON_NON_MONOTONIC,
    BeliefBus,
    BeliefBusConfig,
)
from conrad.schemas.belief import Availability, BeliefMessage, BeliefQuery
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import SourceType
from conrad.schemas.timebase import stamp
from conrad.schemas.world import Domain


def _ns(t):
    return stamp(t, "SIM").time_ns


def test_revision_monotonic_per_belief():
    ids = IdFactory(1)
    bus = BeliefBus(ids)
    b0 = make_belief(ids, revision=0, time_s=1.0)
    assert bus.publish(b0).accepted
    b1 = make_belief(ids, belief_id=b0.belief_id, revision=1, time_s=2.0)
    assert bus.publish(b1).accepted
    again = make_belief(ids, belief_id=b0.belief_id, revision=1, time_s=3.0)
    older = make_belief(ids, belief_id=b0.belief_id, revision=0, time_s=3.0)
    assert bus.publish(again).reason_code == REASON_NON_MONOTONIC
    assert bus.publish(older).reason_code == REASON_NON_MONOTONIC
    head = bus.head(b0.belief_id)
    assert head is not None
    assert head.revision == 1


def test_children_never_write_each_others_beliefs():
    ids = IdFactory(2)
    bus = BeliefBus(ids)
    t = make_belief(ids, domain=Domain.TECHNICAL, revision=0)
    bus.publish(t)
    forged = make_belief(ids, domain=Domain.SPATIAL, belief_id=t.belief_id, revision=1)
    r = bus.publish(forged)
    assert not r.accepted and r.reason_code == REASON_FOREIGN_WRITE
    head = bus.head(t.belief_id)
    assert head is not None
    assert head.domain is Domain.TECHNICAL


def test_staleness_and_availability_from_config():
    ids = IdFactory(3)
    bus = BeliefBus(ids, BeliefBusConfig(freshness_s={"TECHNICAL": 5.0, "SPATIAL": 100.0}))
    bus.publish(make_belief(ids, domain=Domain.TECHNICAL, time_s=10.0))
    bus.publish(make_belief(ids, domain=Domain.SPATIAL, time_s=10.0))
    av = bus.availability(_ns(20.0))
    assert av["TECHNICAL"] == Availability.STALE
    assert av["SPATIAL"] == Availability.AVAILABLE
    assert av["ECOLOGICAL"] == Availability.UNAVAILABLE  # never published
    bus.report_availability(Domain.SPATIAL, Availability.DEGRADED)
    assert bus.availability(_ns(20.0))["SPATIAL"] == Availability.DEGRADED
    bus.report_availability(Domain.SPATIAL, Availability.UNAVAILABLE)
    assert bus.availability(_ns(20.0))["SPATIAL"] == Availability.UNAVAILABLE


def test_one_failing_consumer_does_not_block_others():
    ids = IdFactory(4)
    bus = BeliefBus(ids)
    got: list[BeliefMessage] = []
    ctx: list[BeliefMessage] = []

    def broken(_m):
        raise RuntimeError("consumer crashed")

    bus.subscribe("broken", broken)
    bus.subscribe("ok", got.append)
    bus.attach_child(Domain.ECOLOGICAL, lambda ms: (_ for _ in ()).throw(ValueError("child down")))
    bus.attach_child(Domain.SPATIAL, lambda ms: ctx.extend(ms))
    m = make_belief(ids, domain=Domain.TECHNICAL)
    bus.publish(m)
    assert bus.dispatch() == 1
    assert got == [m] and ctx == [m]
    consumers = {f.consumer for f in bus.failures}
    assert consumers == {"subscriber:broken", "child:ECOLOGICAL"}
    # the failure is recorded (not swallowed) and degrades only the failing child's domain
    bus.publish(make_belief(ids, domain=Domain.ECOLOGICAL, time_s=100.0))
    assert bus.availability(_ns(100.5))["ECOLOGICAL"] == Availability.DEGRADED


def test_cross_domain_delivery_is_context_only_and_never_to_self():
    ids = IdFactory(5)
    bus = BeliefBus(ids)
    seen: dict[Domain, list[BeliefMessage]] = {d: [] for d in Domain}
    for d in Domain:
        bus.attach_child(d, seen[d].extend)
    m = make_belief(ids, domain=Domain.TECHNICAL)
    bus.publish(m)
    bus.dispatch()
    assert seen[Domain.TECHNICAL] == []
    assert seen[Domain.SPATIAL] == [m] and seen[Domain.ECOLOGICAL] == [m]
    assert all(c.source_type is SourceType.CROSS_DOMAIN_CONTEXT for c in bus.context_deliveries)


def test_query_is_coherent_snapshot_with_revisions():
    ids = IdFactory(6)
    bus = BeliefBus(ids)
    msgs = [make_belief(ids, time_s=10.0 + i) for i in range(5)]
    bus.publish_many(msgs)
    snap = bus.query(BeliefQuery(domain=Domain.TECHNICAL, max_results=3, include_provenance=True), _ns(15.0))
    assert len(snap.messages) == 3 and snap.provenance["truncated"]
    assert set(snap.provenance["revisions"]) == {str(m.belief_id) for m in snap.messages}
    assert "provenance_refs" in snap.provenance
    fresh = bus.query(BeliefQuery(), _ns(15.0), max_age_s=2.5)
    assert {m.timestamp.time_ns for m in fresh.messages} == {_ns(13.0), _ns(14.0)}
