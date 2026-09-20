"""Property tests of the receiver-side machinery gate I7 is scored on.

Invariants (ch16, ch19), for arbitrary increment streams:

* a duplicated or reordered increment stream never double-counts a belief contribution, and the receiver's
  revision never goes backwards;
* the sender's model of the receiver (``ReceiverKnowledge``) is never AHEAD of the receiver itself;
* with the BAAC policy a mission-critical unit's F0/F1 increment is always scheduled first;
* the durable queue round-trips through disk unchanged;
* mission information retained is monotone as increments arrive.
"""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conrad.communication import (
    BAAC_POLICY,
    BAACConfig,
    ChannelSim,
    LinkProfile,
    PersistentQueue,
    ReceiverKnowledge,
    ReceiverStore,
    UnitBuilder,
    belief_view,
    compute_deltas,
    schedule,
)
from conrad.communication.queue import QueueEntry
from conrad.evaluation.decision_experiments.fixtures import make_belief, unc
from conrad.evaluation.oracle.comm_oracle import mission_information_retained
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp

CFG = BAACConfig()
NOW = stamp(0.0, "SIM")
fraction = st.floats(0.0, 1.0, allow_nan=False)


def _chain(ids, n):
    """One belief revised n times, and the increments that carry each step."""
    msgs = [make_belief(ids, revision=0, n_evidence=1)]
    for r in range(1, n):
        msgs.append(
            make_belief(
                ids,
                belief_id=msgs[0].belief_id,
                revision=r,
                n_evidence=1,
                uncertainty=unc(ua=0.05 * r),
                properties={"condition": "DEGRADED" if r % 2 else "NOMINAL"},
            )
        )
    increments = []
    for i, m in enumerate(msgs):
        known = None if i == 0 else belief_view(msgs[i - 1])
        increments.append(
            {"kind": "deltas", "deltas": [d.model_dump(mode="json") for d in compute_deltas(known, m)]}
        )
    return msgs, increments


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(order=st.lists(st.integers(0, 3), min_size=1, max_size=14), n=st.integers(2, 4))
def test_duplicated_or_reordered_increments_never_double_count(order, n):
    ids = IdFactory(1)
    msgs, increments = _chain(ids, n)
    store = ReceiverStore()
    bid = msgs[0].belief_id
    seen = []
    for i in order:
        store.receive(increments[i % n])
        rev = store.revision(bid)
        if rev is not None:
            seen.append(rev)
    assert store.duplicate_contributions() == 0
    assert seen == sorted(seen), "the receiver's revision never goes backwards"
    applied = store.applied.get(bid, [])
    assert applied == sorted(set(applied)), "each revision contributes at most once"


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(order=st.lists(st.integers(0, 3), min_size=1, max_size=14), n=st.integers(2, 4))
def test_sender_model_is_never_ahead_of_the_receiver(order, n):
    """K_R may be conservative (it forgets, or falls back to an older revision, on a stale or failed apply)
    but must never claim more than the receiver holds: that direction would suppress a needed send."""
    ids = IdFactory(2)
    msgs, increments = _chain(ids, n)
    store, known = ReceiverStore(), ReceiverKnowledge()
    bid = msgs[0].belief_id
    for i in order:
        inc = increments[i % n]
        store.receive(inc)
        known.acknowledge(inc)  # link-layer ACK of the same increment
        mine, theirs = known.known_revision(bid), store.revision(bid)
        assert mine is None or (theirs is not None and mine <= theirs)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(values=st.lists(fraction, min_size=2, max_size=8), bw=st.floats(2_000.0, 200_000.0))
def test_baac_schedules_a_critical_increment_first(values, bw):
    ids = IdFactory(3)
    builder = UnitBuilder(ids, CFG)
    ch = ChannelSim([LinkProfile(name="a", bandwidth_bps=bw, packet_loss=0.0)], seed=0)
    entries = []
    for v in values:
        built = builder.belief_unit(make_belief(ids, n_evidence=1), None, v, NOW)
        assert built is not None
        entries.append(QueueEntry(content=built[0]))
    plan = schedule(entries, ch.link_states(0.0), ch, ReceiverKnowledge(), 0, 1.0, BAAC_POLICY, CFG)
    critical = {e.unit_id for e in entries if e.critical}
    if critical and plan:
        first = plan[0]
        assert first.unit_id in critical and first.to_level <= 1, (
            f"BAAC must pre-empt with a critical F0/F1 increment, scheduled {first}"
        )


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    values=st.lists(fraction, min_size=1, max_size=6),
    partial=st.integers(0, 4000),
    attempts=st.integers(0, 3),
)
def test_durable_queue_round_trips_through_disk(values, partial, attempts, tmp_path_factory):
    path = tmp_path_factory.mktemp("q") / "queue.json"
    ids = IdFactory(4)
    builder = UnitBuilder(ids, CFG)
    q = PersistentQueue(8_000_000_000, path)
    for v in values:
        built = builder.belief_unit(make_belief(ids, n_evidence=1), None, v, NOW)
        assert built is not None
        q.put(built[0], 0)
    head = q.entries()[0]
    q.replace(head.model_copy(update={"partial_level": 1, "partial_bits": partial, "attempts": attempts}))
    back = PersistentQueue(8_000_000_000, path)
    assert [e.unit_id for e in back.entries()] == [e.unit_id for e in q.entries()]
    assert back.queued_bits == q.queued_bits
    restored = back.get(head.unit_id)
    assert restored is not None
    assert (restored.partial_level, restored.partial_bits, restored.attempts) == (1, partial, attempts)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(levels=st.lists(st.integers(0, 4), min_size=1, max_size=10), value=st.floats(0.81, 1.0))
def test_mission_information_retained_is_monotone_in_arriving_increments(levels, value):
    ids = IdFactory(5)
    builder = UnitBuilder(ids, CFG)
    m = make_belief(ids, n_evidence=2, severity=0.9)
    built = builder.belief_unit(m, None, value, NOW)
    assert built is not None
    store = ReceiverStore()
    latest, values = {m.belief_id: m}, {m.belief_id: 1.0}
    score = 0.0
    for lv in sorted(levels):  # increments arrive in fidelity order, possibly repeated
        store.receive(built[0].increments[lv])
        now = mission_information_retained(store, latest, values, CFG.information_retained)
        assert now >= score - 1e-12, "delivered information can never reduce the retained score"
        score = now
    assert store.duplicate_contributions() == 0
