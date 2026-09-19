"""Property tests: delta round trip, scheduler budgets, queue never drops critical units."""

from hypothesis import given, settings
from hypothesis import strategies as st

from conrad.communication import (
    BAAC_POLICY,
    BASELINE_POLICIES,
    BAACConfig,
    ChannelSim,
    LinkProfile,
    PersistentQueue,
    ReceiverKnowledge,
    UnitBuilder,
    apply_deltas,
    belief_view,
    compute_deltas,
    schedule,
)
from conrad.communication.queue import QueueEntry
from conrad.evaluation.decision_experiments.fixtures import make_belief, unc
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp

value = st.one_of(st.floats(-1e3, 1e3, allow_nan=False), st.text(max_size=8), st.booleans())
channel = st.floats(0.0, 1.0, allow_nan=False)


@settings(max_examples=60, deadline=None)
@given(
    v0=value, v1=value, u0=channel, u1=channel, c=st.integers(0, 3), e=st.integers(0, 3), r=st.integers(0, 2)
)
def test_delta_round_trip(v0, v1, u0, u1, c, e, r):
    ids = IdFactory(1)
    old = make_belief(ids, revision=2, properties={"p": v0}, uncertainty=unc(ua=u0))
    new = make_belief(
        ids,
        belief_id=old.belief_id,
        revision=5,
        properties={"p": v1},
        uncertainty=unc(ua=u1),
        n_conflicts=c,
        n_evidence=e,
        relationships=[ids.new() for _ in range(r)],
    )
    assert apply_deltas(belief_view(old), compute_deltas(belief_view(old), new)) == belief_view(new)


@settings(max_examples=30, deadline=None)
@given(
    bw=st.floats(0.0, 50_000.0),
    n=st.integers(1, 8),
    values=st.lists(channel, min_size=8, max_size=8),
    policy=st.sampled_from([BAAC_POLICY, *BASELINE_POLICIES.values()]),
)
def test_scheduler_never_exceeds_link_budget(bw, n, values, policy):
    ids = IdFactory(2)
    cfg = BAACConfig()
    ch = ChannelSim([LinkProfile(name="a", bandwidth_bps=bw, packet_loss=0.1)], seed=0)
    builder = UnitBuilder(ids, cfg)
    entries = []
    for i in range(n):
        built = builder.belief_unit(make_belief(ids), None, values[i], stamp(0.0, "SIM"))
        assert built is not None
        entries.append(QueueEntry(content=built[0]))
    plan = schedule(entries, ch.link_states(0.0), ch, ReceiverKnowledge(), 0, 1.0, policy, cfg)
    assert sum(p.reserved_bits for p in plan) <= bw + 1e-6
    assert all(p.chunk_bits >= 1 for p in plan)


@settings(max_examples=30, deadline=None)
@given(values=st.lists(channel, min_size=1, max_size=10), cap=st.integers(1, 200_000))
def test_queue_overload_never_drops_critical(values, cap):
    ids = IdFactory(3)
    builder = UnitBuilder(ids, BAACConfig())
    q = PersistentQueue(cap)
    crit = []
    for v in values:
        built = builder.belief_unit(make_belief(ids), None, v, stamp(0.0, "SIM"))
        assert built is not None
        content = built[0]
        q.put(content, 0)
        if content.critical:
            crit.append(content.unit.unit_id)
    assert all(q.get(u) is not None for u in crit)
    assert not any(d.critical for d in q.dropped)
    assert q.queued_bits <= cap or all(e.critical for e in q.entries())
