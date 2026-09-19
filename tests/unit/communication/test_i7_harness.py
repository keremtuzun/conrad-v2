"""Gate-I7 harness mechanics: selectable policy, critical pre-emption after reconnection, packet carry, no
double-counting at the receiver."""

import pytest

from conrad.communication import (
    ALL_POLICIES,
    BAACConfig,
    BAACSender,
    ChannelSim,
    LinkProfile,
    ReceiverStore,
    policy_by_name,
)
from conrad.evaluation.decision_experiments.fixtures import make_belief, unc
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp


def _now(t):
    return stamp(float(t), "SIM")


def test_policy_is_selected_from_config():
    ch = ChannelSim([LinkProfile(name="a", bandwidth_bps=1000.0)], seed=0)
    for name in ALL_POLICIES:
        s = BAACSender(IdFactory(1), ch, BAACConfig(scheduler_policy=name))
        assert s.policy.name == name
    with pytest.raises(KeyError):
        policy_by_name("nope")


def _outage_run(policy_name, routine_first=8):
    ids = IdFactory(7)
    ch = ChannelSim(
        [LinkProfile(name="a", bandwidth_bps=1200.0, packet_loss=0.0, outages_s=((5.0, 30.0),))], seed=3
    )
    sender = BAACSender(ids, ch, BAACConfig(scheduler_policy=policy_name))
    store = ReceiverStore()
    log = []

    def sink(inc):
        if inc["kind"] == "alert":
            log.append((sender.current_arrival_ns, inc["belief_id"], "alert"))
        elif inc["kind"] == "deltas":
            log.append((sender.current_arrival_ns, inc["deltas"][0]["belief_id"], "deltas"))
        return store.receive(inc)

    routine = [make_belief(ids, uncertainty=unc(ua=0.2)) for _ in range(routine_first)]
    crit0 = make_belief(ids)
    for t in range(60):
        if t == 1:
            for m in routine:
                sender.offer(m, 0.5, _now(t))
        if t == 10:  # finding made while the link is down
            crit = make_belief(
                ids, belief_id=crit0.belief_id, revision=1, properties={"condition": "FAILED"}, severity=0.9
            )
            sender.offer(crit, 0.9, _now(t))
        if t in (15, 20, 25):  # routine updates keep arriving during the outage
            for i, m in enumerate(routine[:3]):
                routine[i] = make_belief(
                    ids, belief_id=m.belief_id, revision=m.revision + 1, uncertainty=unc(ua=0.3)
                )
                sender.offer(routine[i], 0.5, _now(t))
        sender.step(float(t), 1.0, sink)
    return sender, store, log, crit0.belief_id


def test_baac_delivers_critical_first_after_reconnection_and_coalesces():
    sender, store, log, crit = _outage_run("C-B10_baac")
    after = [e for e in log if e[0] >= 30 * 10**9]
    first_crit_delta = next(i for i, e in enumerate(after) if e[1] == str(crit) and e[2] == "deltas")
    assert all(e[1] == str(crit) for e in after[: first_crit_delta + 1])
    assert after[0] == (after[0][0], str(crit), "alert")
    assert sender.coalesced > 0  # stale revisions superseded, not queued twice
    assert store.duplicate_contributions() == 0
    assert store.revision(crit) == 1


def test_fifo_baseline_does_not_preempt():
    """FIFO sends full fidelity in arrival order: the older routine unit blocks the critical finding."""
    sender, _, log, crit = _outage_run("C-B1_fifo")
    assert not any(e[1] == str(crit) for e in log)
    after = [t for t in sender.transmissions if t.sent_time_ns >= 30 * 10**9]
    assert after and all(sender.unit_created[t.unit_id][2] != crit for t in after)


def test_slow_link_carries_capacity_until_a_packet_fits():
    ids = IdFactory(9)
    ch = ChannelSim([LinkProfile(name="a", bandwidth_bps=1.2, packet_loss=0.0)], seed=0)
    sender = BAACSender(ids, ch, BAACConfig())
    store = ReceiverStore()
    sender.offer(make_belief(ids), 0.9, _now(0))
    for t in range(400):
        sender.step(float(t), 1.0, store.receive)
    assert store.alerts, "a 1.2 bit/s link must still deliver the 184-bit alert within 400 s"
    # the link cannot bank idle capacity
    assert sender.carry_bits["a"] <= ch.expected_bits("a", ch.profiles["a"].packet_bits)


def test_retransmitted_delta_is_not_counted_twice():
    ids = IdFactory(11)
    m = make_belief(ids)
    from conrad.communication import compute_deltas

    inc = {"kind": "deltas", "deltas": [d.model_dump(mode="json") for d in compute_deltas(None, m)]}
    store = ReceiverStore()
    store.receive(inc)
    store.receive(inc)
    assert store.applied[m.belief_id] == [m.revision]
    assert store.stale_ignored == 1 and store.duplicate_contributions() == 0
