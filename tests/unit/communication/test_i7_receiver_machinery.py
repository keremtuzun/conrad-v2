"""Receiver-side machinery the gate I7 criteria depend on (ch16, ch19).

The I7 criteria are computed from what the RECEIVER holds: mission information retained, critical latency,
receiver sync error, deadline success, queue behaviour and duplicate contributions. These tests pin the
mechanisms behind those numbers so the gate harness cannot pass on a broken receiver:

  receiver knowledge tracking, semantic deltas, the durable store-and-forward queue, acknowledgements and
  reconciliation (resync), blackout recovery, stale-item re-evaluation, fidelity control, mission-critical
  priority, and that a duplicate or retransmitted packet never double-counts belief evidence.
"""

from __future__ import annotations

import json

import pytest

from conrad.communication import (
    BAACConfig,
    BAACSender,
    ChannelSim,
    LinkProfile,
    PersistentQueue,
    ReceiverKnowledge,
    ReceiverStore,
    UnitBuilder,
    apply_deltas,
    belief_view,
    compute_deltas,
)
from conrad.communication.delta import ResyncRequired
from conrad.communication.units import ALERT_FRAME, alert_frame, increment_bits
from conrad.evaluation.decision_experiments.fixtures import make_belief, unc
from conrad.evaluation.oracle.comm_oracle import mission_information_retained, sync_error
from conrad.schemas.belief import Lifecycle
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp


def _now(t):
    return stamp(float(t), "SIM")


def _deltas_increment(known, message):
    return {"kind": "deltas", "deltas": [d.model_dump(mode="json") for d in compute_deltas(known, message)]}


# ------------------------------------------------------------------ receiver knowledge tracking (sender side)
def test_receiver_knowledge_only_advances_on_acknowledged_delivery():
    """K_R is the SENDER's model of the receiver. An unacknowledged unit keeps its novelty and is retried."""
    ids = IdFactory(1)
    ch = ChannelSim([LinkProfile(name="a", bandwidth_bps=0.0)], seed=0)  # nothing can cross
    sender = BAACSender(ids, ch, BAACConfig())
    m = make_belief(ids, severity=0.9)
    sender.offer(m, 0.9, _now(0))
    for t in range(5):
        sender.step(float(t), 1.0)
    assert sender.receiver_model.known_revision(m.belief_id) is None
    assert sender.receiver_model.alerted == {}
    assert len(sender.queue) == 1, "an undelivered unit stays queued"


def test_receiver_knowledge_mirrors_the_receiver_after_delivery():
    ids = IdFactory(2)
    ch = ChannelSim([LinkProfile(name="a", bandwidth_bps=200_000.0, packet_loss=0.0)], seed=0)
    sender = BAACSender(ids, ch, BAACConfig())
    store = ReceiverStore()
    m = make_belief(ids, severity=0.9)
    sender.offer(m, 0.9, _now(0))
    for t in range(6):
        sender.step(float(t), 1.0, store.receive)
    assert sender.receiver_model.known_revision(m.belief_id) == store.revision(m.belief_id) == m.revision
    assert sender.receiver_model.alerted[m.belief_id] == int(store.alerts[m.belief_id]["revision"])


def test_receiver_knowledge_forget_makes_the_next_unit_carry_full_state():
    ids = IdFactory(3)
    known = ReceiverKnowledge()
    m0 = make_belief(ids, revision=3)
    known.acknowledge(_deltas_increment(None, m0))
    assert known.known_revision(m0.belief_id) == 3
    known.forget(m0.belief_id)
    m1 = make_belief(ids, belief_id=m0.belief_id, revision=4, properties={"condition": "DEGRADED"})
    deltas = compute_deltas(known.known(m1.belief_id), m1)
    assert [d.delta_type.value for d in deltas] == ["NEW_BELIEF"], "a forgotten belief is re-sent in full"


# ------------------------------------------------------------------------------------- semantic deltas
def test_semantic_delta_round_trip_and_retirement():
    ids = IdFactory(4)
    old = make_belief(ids, revision=2)
    new = make_belief(
        ids,
        belief_id=old.belief_id,
        revision=3,
        properties={"condition": "FAILED"},
        lifecycle=Lifecycle.RETIRED,
        uncertainty=unc(uc=0.7),
    )
    deltas = compute_deltas(belief_view(old), new)
    kinds = [d.delta_type.value for d in deltas]
    assert "BELIEF_RETIRED" in kinds
    assert apply_deltas(belief_view(old), deltas) == belief_view(new)


def test_delta_against_a_missing_base_raises_resync_required():
    ids = IdFactory(5)
    old = make_belief(ids, revision=2)
    mid = make_belief(ids, belief_id=old.belief_id, revision=3, properties={"condition": "DEGRADED"})
    new = make_belief(ids, belief_id=old.belief_id, revision=4, properties={"condition": "FAILED"})
    deltas = compute_deltas(belief_view(mid), new)  # base revision 3
    with pytest.raises(ResyncRequired):
        apply_deltas(belief_view(old), deltas)  # receiver only has revision 2


def test_receiver_asks_for_a_resync_and_the_sender_answers_with_full_state():
    ids = IdFactory(6)
    store = ReceiverStore()
    old = make_belief(ids, revision=2)
    mid = make_belief(ids, belief_id=old.belief_id, revision=3)
    new = make_belief(ids, belief_id=old.belief_id, revision=4, properties={"condition": "FAILED"})
    store.receive(_deltas_increment(None, old))
    req = store.receive(_deltas_increment(belief_view(mid), new))
    assert req is not None and req.have_revision == 2 and req.needed_base == 3
    assert len(store.resync_requests) == 1
    ch = ChannelSim([LinkProfile(name="a", bandwidth_bps=200_000.0, packet_loss=0.0)], seed=0)
    sender = BAACSender(ids, ch, BAACConfig())
    sender.offer(new, 0.5, _now(0))
    sender.receiver_model.acknowledge(_deltas_increment(None, mid))  # sender wrongly believes rev 3 is held
    sender.handle_resync(req, _now(1))
    assert sender.receiver_model.known(new.belief_id) is None
    queued = [e for e in sender.queue.entries() if e.content.new_revision == new.revision]
    assert queued and queued[-1].content.increments[1]["deltas"][0]["delta_type"] == "NEW_BELIEF"


# --------------------------------------------------------------------------------------- durable queue
def test_persistent_queue_survives_a_restart_with_partial_fidelity_and_drops(tmp_path):
    """Store-and-forward is durable: entries, per-entry fragment progress and drop records reload from disk."""
    ids = IdFactory(7)
    builder = UnitBuilder(ids, BAACConfig())
    path = tmp_path / "queue.json"
    q = PersistentQueue(2_000_000_000, path)
    built = builder.belief_unit(make_belief(ids, severity=0.9), None, 0.9, _now(0))
    assert built is not None
    q.put(built[0], 0)
    entry = q.entries()[0]
    q.replace(entry.model_copy(update={"partial_level": 1, "partial_bits": 64, "attempts": 2}))
    routine = builder.belief_unit(make_belief(ids), None, 0.1, _now(0))
    assert routine is not None
    q.put(routine[0], 0)
    q.remove(routine[0].unit.unit_id, "DEADLINE_EXPIRED", 5)

    reloaded = PersistentQueue(2_000_000_000, path)
    assert [e.unit_id for e in reloaded.entries()] == [entry.unit_id]
    back = reloaded.entries()[0]
    assert (back.partial_level, back.partial_bits, back.attempts) == (1, 64, 2)
    assert [d.reason for d in reloaded.dropped] == ["DEADLINE_EXPIRED"]
    assert reloaded.queued_bits == q.queued_bits
    assert json.loads(path.read_text(encoding="utf-8"))["entries"], "the queue file is real JSON on disk"


def test_queue_overload_drops_the_lowest_value_routine_unit_and_never_a_critical_one():
    ids = IdFactory(8)
    builder = UnitBuilder(ids, BAACConfig())
    q = PersistentQueue(1)  # every unit is over capacity
    crit = builder.belief_unit(make_belief(ids, severity=0.9), None, 0.95, _now(0))
    low = builder.belief_unit(make_belief(ids), None, 0.05, _now(0))
    high = builder.belief_unit(make_belief(ids), None, 0.50, _now(0))
    assert crit is not None and low is not None and high is not None
    q.put(crit[0], 0)
    q.put(high[0], 0)
    q.put(low[0], 0)
    reasons = {d.unit_id: d.reason for d in q.dropped}
    assert low[0].unit.unit_id in reasons and reasons[low[0].unit.unit_id] == "OVERLOAD_LOWEST_VALUE"
    assert q.get(crit[0].unit.unit_id) is not None
    assert not any(d.critical for d in q.dropped)
    assert q.overflow_events > 0, "a queue left holding only critical units records an explicit overflow"


# ---------------------------------------------------------------------- blackout recovery / re-evaluation
def _blackout_sender(seed=21, outage=(5.0, 30.0)):
    ids = IdFactory(seed)
    ch = ChannelSim(
        [LinkProfile(name="a", bandwidth_bps=1200.0, packet_loss=0.0, outages_s=(outage,))], seed=3
    )
    return ids, ch, BAACSender(ids, ch, BAACConfig()), ReceiverStore()


def test_reevaluation_runs_once_per_reconnection_and_rebuilds_against_receiver_knowledge():
    ids, _ch, sender, store = _blackout_sender()
    m = make_belief(ids, revision=0)
    sender.offer(m, 0.5, _now(0))
    for t in range(12):
        sender.step(float(t), 1.0, store.receive)
    before = sender.reevaluations
    m2 = make_belief(ids, belief_id=m.belief_id, revision=1, properties={"condition": "DEGRADED"})
    sender.offer(m2, 0.5, _now(12))
    for t in range(12, 45):
        sender.step(float(t), 1.0, store.receive)
    assert sender.reevaluations == before + 1, "exactly one re-evaluation at the reconnection"
    assert store.revision(m.belief_id) == 1
    assert store.duplicate_contributions() == 0


def test_reevaluation_drops_a_unit_the_receiver_already_holds():
    """An offer made while the link was up and delivered is not re-sent after the next reconnection."""
    ids, _ch, sender, store = _blackout_sender(seed=22, outage=(10.0, 20.0))
    m = make_belief(ids)
    sender.offer(m, 0.5, _now(0))
    for t in range(30):
        sender.step(float(t), 1.0, store.receive)
    assert store.revision(m.belief_id) == m.revision
    assert len(sender.queue) == 0
    assert not any(d.reason == "DEADLINE_EXPIRED" for d in sender.queue.dropped)


def test_reevaluation_expires_non_critical_deadlines_but_never_a_critical_unit():
    ids, ch, sender, store = _blackout_sender(seed=23, outage=(1.0, 60.0))
    sender = BAACSender(ids, ch, BAACConfig(default_deadline_s=5.0))
    routine = make_belief(ids)
    crit = make_belief(ids, properties={"condition": "FAILED"}, severity=0.9)
    sender.offer(routine, 0.2, _now(0))
    sender.offer(crit, 0.9, _now(0))
    for t in range(70):
        sender.step(float(t), 1.0, store.receive)
    reasons = {d.reason for d in sender.queue.dropped}
    assert "DEADLINE_EXPIRED" in reasons
    assert not any(d.critical and d.reason == "DEADLINE_EXPIRED" for d in sender.queue.dropped)


def test_stale_revisions_are_coalesced_not_queued_twice():
    ids, _ch, sender, store = _blackout_sender(seed=24, outage=(2.0, 40.0))
    m = make_belief(ids)
    for rev in range(6):
        m = make_belief(ids, belief_id=m.belief_id, revision=rev, uncertainty=unc(ua=0.1 * rev))
        sender.offer(m, 0.5, _now(3 + rev))
    queued = [e for e in sender.queue.entries() if e.content.unit.belief_ids == (m.belief_id,)]
    assert len(queued) == 1, "only the freshest revision of a belief waits in the queue"
    assert sender.coalesced == 5
    for t in range(50):
        sender.step(float(t), 1.0, store.receive)
    assert store.revision(m.belief_id) == 5 and store.duplicate_contributions() == 0


def test_stale_increment_arriving_after_a_newer_one_is_ignored():
    ids = IdFactory(25)
    store = ReceiverStore()
    m0 = make_belief(ids, revision=0)
    m1 = make_belief(ids, belief_id=m0.belief_id, revision=1, properties={"condition": "DEGRADED"})
    store.receive(_deltas_increment(None, m0))
    store.receive(_deltas_increment(belief_view(m0), m1))
    store.receive(_deltas_increment(None, m0))  # a late duplicate of the ORIGINAL full-state unit
    assert store.revision(m0.belief_id) == 1, "an older revision never overwrites a newer one"
    assert store.stale_ignored == 1 and store.applied[m0.belief_id] == [0, 1]
    assert store.duplicate_contributions() == 0


def test_alert_store_keeps_the_latest_revision_when_an_older_alert_arrives_late():
    ids = IdFactory(26)
    store = ReceiverStore()
    bid = ids.new()
    store.receive({"kind": "alert", "belief_id": str(bid), "revision": 7, "domain": "TECHNICAL"})
    store.receive({"kind": "alert", "belief_id": str(bid), "revision": 3, "domain": "TECHNICAL"})
    assert int(store.alerts[bid]["revision"]) == 7


# ------------------------------------------------------------------------------------- fidelity control
def test_fidelity_levels_are_cumulative_monotonic_and_measured():
    ids = IdFactory(27)
    cfg = BAACConfig()
    builder = UnitBuilder(ids, cfg)
    m = make_belief(ids, n_evidence=3, severity=0.9)
    built = builder.belief_unit(m, None, 0.9, _now(0))
    assert built is not None
    content = built[0]
    sizes = [(int(o.fidelity), o.size_bits, o.information_retained) for o in content.unit.fidelity_levels]
    assert [s[0] for s in sizes] == [0, 1, 2, 3, 4]
    assert [s[1] for s in sizes] == sorted(s[1] for s in sizes), "cumulative sizes are non-decreasing"
    assert [s[2] for s in sizes] == list(cfg.information_retained)
    assert sizes[0][1] == increment_bits(content.increments[0]) == 8 * ALERT_FRAME.size
    assert len(alert_frame(content.increments[0])) == 23
    # F3 carries compressed evidence, F4 the raw bytes: the jump is the configured fraction of them
    raw = 3 * 8 * cfg.default_raw_evidence_bytes
    assert content.evidence_bits[4] == raw
    assert 0 < content.evidence_bits[3] < raw


def test_a_routine_unit_has_no_f0_alert():
    ids = IdFactory(28)
    builder = UnitBuilder(ids, BAACConfig())
    built = builder.belief_unit(make_belief(ids), None, 0.1, _now(0))
    assert built is not None
    assert 0 not in built[0].increments and built[0].levels[0] == 1
    assert not built[0].critical


def test_progressive_fidelity_delivers_f0_before_f1_on_a_slow_link():
    ids = IdFactory(29)
    ch = ChannelSim([LinkProfile(name="a", bandwidth_bps=60.0, packet_loss=0.0)], seed=0)
    sender = BAACSender(ids, ch, BAACConfig())
    order = []

    def sink(inc):
        order.append(inc["kind"])
        return None

    sender.offer(make_belief(ids, severity=0.9), 0.9, _now(0))
    for t in range(200):
        sender.step(float(t), 1.0, sink)
    assert order[:1] == ["alert"], f"the tiny F0 frame must arrive first, got {order[:3]}"
    assert "deltas" in order


# ------------------------------------------------------------------------------ mission-critical priority
def test_critical_alert_and_delta_preempt_routine_traffic_after_a_blackout():
    ids, _ch, sender, store = _blackout_sender(seed=30, outage=(5.0, 30.0))
    routine = [make_belief(ids) for _ in range(8)]
    crit = make_belief(ids, properties={"condition": "FAILED"}, severity=0.9)
    arrivals = []

    def sink(inc):
        if inc["kind"] == "alert":
            arrivals.append((sender.current_arrival_ns, str(inc["belief_id"]), "alert"))
        elif inc["kind"] == "deltas":
            arrivals.append((sender.current_arrival_ns, str(inc["deltas"][0]["belief_id"]), "deltas"))
        return store.receive(inc)

    for t in range(60):
        if t == 1:  # routine traffic is already queued when the link drops at 5 s
            for m in routine:
                sender.offer(m, 0.5, _now(t))
        if t == 10:  # the mission-critical finding is made while the link is DOWN
            sender.offer(crit, 0.9, _now(t))
        sender.step(float(t), 1.0, sink)
    assert not [a for a in arrivals if a[1] == str(crit.belief_id) and a[0] < 30 * 10**9]
    after = [a for a in arrivals if a[0] >= 30 * 10**9]
    first_delta = next(i for i, a in enumerate(after) if a[1] == str(crit.belief_id) and a[2] == "deltas")
    assert all(a[1] == str(crit.belief_id) for a in after[: first_delta + 1])
    assert store.revision(crit.belief_id) == crit.revision


def test_mission_value_marks_a_critical_unit_and_drives_the_f0_frame():
    ids = IdFactory(31)
    cfg = BAACConfig()
    builder = UnitBuilder(ids, cfg)
    m = make_belief(ids, properties={"condition": "FAILED"}, severity=0.9)
    just_under = builder.belief_unit(m, None, cfg.critical_value - 1e-9, _now(0))
    at_threshold = builder.belief_unit(m, None, cfg.critical_value, _now(0))
    assert just_under is not None and at_threshold is not None
    assert not just_under[0].critical and 0 not in just_under[0].increments
    assert at_threshold[0].critical and 0 in at_threshold[0].increments


# -------------------------------------------------------------- duplicates never double-count evidence
def test_retransmission_of_every_fidelity_level_never_double_counts():
    ids = IdFactory(32)
    store = ReceiverStore()
    builder = UnitBuilder(ids, BAACConfig())
    m = make_belief(ids, n_evidence=2, severity=0.9)
    built = builder.belief_unit(m, None, 0.9, _now(0))
    assert built is not None
    for _ in range(3):  # the whole unit retransmitted three times, every level
        for level in sorted(built[0].increments):
            store.receive(built[0].increments[level])
    assert store.applied[m.belief_id] == [m.revision]
    assert store.duplicate_contributions() == 0
    assert store.stale_ignored == 2 * 1  # two extra copies of the F1 delta, both ignored
    assert len(store.known_evidence_ids) == 2, "evidence IDs are a set, not a multiset"


def test_duplicate_contributions_counts_a_genuine_double_apply():
    """The metric must be able to FAIL: hand-built double bookkeeping is reported, not hidden."""
    ids = IdFactory(33)
    store = ReceiverStore()
    m = make_belief(ids)
    store.receive(_deltas_increment(None, m))
    store.applied[m.belief_id].append(m.revision)  # simulate a broken receiver
    assert store.duplicate_contributions() == 1


def test_oracle_scores_and_sync_error_follow_what_the_receiver_holds():
    ids = IdFactory(34)
    cfg = BAACConfig()
    store = ReceiverStore()
    a = make_belief(ids, revision=2, n_evidence=1)
    b = make_belief(ids, revision=1, n_evidence=1)
    latest = {a.belief_id: a, b.belief_id: b}
    values = {a.belief_id: 1.0, b.belief_id: 1.0}
    assert mission_information_retained(store, latest, values, cfg.information_retained) == 0.0
    assert sync_error(store, latest) == 1.0
    store.receive(_deltas_increment(None, a))
    assert sync_error(store, latest) == 0.5
    retained = mission_information_retained(store, latest, values, cfg.information_retained)
    # The F1 delta of a NEW belief already carries evidence_support, so the receiver knows the evidence IDs
    # and the oracle scores it at the F2 ("+ evidence summary") level. See test_f2_adds_no_oracle_score_...
    assert retained == pytest.approx(cfg.information_retained[2] / 2)
    store.receive(_deltas_increment(None, b))
    assert sync_error(store, latest) == 0.0


def test_f2_adds_no_oracle_score_once_f1_carried_the_evidence_ids():
    """Documented coarseness of the scoring model, not of the wire format (see docs/audits/I7_COMMUNICATIONS.md).

    ``ReceiverStore._apply`` records the evidence IDs listed in the applied view, and ``belief_score``
    promotes a belief to the F2 level as soon as every evidence ID is known. For a belief whose F1 delta
    changed ``evidence_support`` (every NEW_BELIEF), the F2 evidence-summary increment therefore adds nothing
    to ``mission_information_retained``: the oracle distinguishes four levels, not five. F3 and F4 still
    score, because they need the evidence BYTES. The effect is identical for every arm."""
    ids = IdFactory(35)
    cfg = BAACConfig()
    builder = UnitBuilder(ids, cfg)
    m = make_belief(ids, n_evidence=2, severity=0.9)
    built = builder.belief_unit(m, None, 0.9, _now(0))
    assert built is not None
    store = ReceiverStore()
    latest, values = {m.belief_id: m}, {m.belief_id: 1.0}
    store.receive(built[0].increments[1])
    after_f1 = mission_information_retained(store, latest, values, cfg.information_retained)
    store.receive(built[0].increments[2])
    after_f2 = mission_information_retained(store, latest, values, cfg.information_retained)
    store.receive(built[0].increments[3])
    after_f3 = mission_information_retained(store, latest, values, cfg.information_retained)
    store.receive(built[0].increments[4])
    after_f4 = mission_information_retained(store, latest, values, cfg.information_retained)
    assert after_f1 == after_f2 == cfg.information_retained[2]
    assert after_f3 == cfg.information_retained[3] and after_f4 == cfg.information_retained[4]
