"""The 2026-09-20 BAAC scheduler repair (gate I7 criterion 3 on development world 5100001).

Two mechanisms, both BAAC-only and both switchable from ``BAACConfig`` so the ablation stays reproducible:

1. ``preempt_until_delivered``: a critical unit's ch19 Level 0/1 pre-emption lasts until the receiver holds
   that belief. Unbounded pre-emption let one belief revised 383 times in 240 s hold the whole link.
2. ``receiver_relative_value``: an increment is scored by the information the RECEIVER GAINS, discounting
   what it already holds about that belief, instead of by the unit's absolute information content.

Both defaults are on. With both off the scheduler must reproduce the pre-repair choice, each mechanism on its
own must reproduce the measured ablation, and the baseline policies must be untouched by either flag.
"""

import pytest

from conrad.communication import (
    BAAC_POLICY,
    BASELINE_POLICIES,
    BAACConfig,
    ChannelSim,
    LinkProfile,
    ReceiverKnowledge,
    UnitBuilder,
    compute_deltas,
    schedule,
)
from conrad.communication.queue import QueueEntry
from conrad.communication.scheduler import _critical_undelivered, receiver_credit
from conrad.evaluation.decision_experiments.fixtures import make_belief, unc
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp

NOW = stamp(0.0, "SIM")


def _entry(builder, message, known, value):
    built = builder.belief_unit(message, known, value, NOW)
    assert built is not None
    return QueueEntry(content=built[0])


def _ack_view(receiver: ReceiverKnowledge, message, known=None) -> None:
    """Mirror what an ACKed F1 delta of ``message`` leaves in the sender's model of the receiver."""
    deltas = compute_deltas(known, message)
    receiver.acknowledge({"kind": "deltas", "deltas": [d.model_dump(mode="json") for d in deltas]})


def _setup(cfg: BAACConfig):
    """One critical belief the receiver already holds and that keeps being revised, plus an unseen belief."""
    ids = IdFactory(5)
    builder = UnitBuilder(ids, cfg)
    receiver = ReceiverKnowledge()
    crit0 = make_belief(ids, properties={"condition": "FAILED"}, severity=0.9)
    receiver.acknowledge({"kind": "alert", "belief_id": str(crit0.belief_id), "revision": crit0.revision})
    _ack_view(receiver, crit0)  # the finding itself already reached the receiver, alert and delta
    crit_new = make_belief(
        ids,
        belief_id=crit0.belief_id,
        revision=crit0.revision + 40,
        properties={"condition": "FAILED", "corrosion_depth_m": 0.02},
        severity=0.95,
        uncertainty=unc(ua=0.3, ue=0.2),
    )
    fresh = make_belief(ids)  # never offered before: the receiver knows nothing about it
    entries = [
        _entry(builder, crit_new, receiver.known(crit0.belief_id), 0.9),
        _entry(builder, fresh, None, 0.5),
    ]
    channel = ChannelSim(
        [LinkProfile(name="a", bandwidth_bps=20_000.0, packet_loss=0.0, packet_bits=256)], seed=0
    )
    return entries, receiver, channel, crit0.belief_id, fresh.belief_id


def _first_belief(cfg: BAACConfig) -> str:
    entries, receiver, channel, crit_id, _ = _setup(cfg)
    plan = schedule(entries, channel.link_states(0.0), channel, receiver, 0, 1.0, BAAC_POLICY, cfg, None)
    assert plan, "the link has capacity for both units, so something must be scheduled"
    by_unit = {e.unit_id: e for e in entries}
    return "critical" if by_unit[plan[0].unit_id].content.unit.belief_ids[0] == crit_id else "fresh"


def test_pre_repair_scheduler_re_sends_the_already_delivered_critical_belief():
    """Both flags off reproduces the defect: the re-offer wins over the first delivery of an unseen belief."""
    assert (
        _first_belief(BAACConfig(preempt_until_delivered=False, receiver_relative_value=False)) == "critical"
    )


def test_bounded_preemption_alone_does_not_fix_the_ordering():
    """Measured ablation: without the value model the critical re-offer still wins on value per bit."""
    assert (
        _first_belief(BAACConfig(preempt_until_delivered=True, receiver_relative_value=False)) == "critical"
    )


def test_receiver_relative_value_alone_does_not_fix_the_ordering():
    """Measured ablation: pre-emption is lexicographically first, so the value model alone cannot reorder."""
    assert (
        _first_belief(BAACConfig(preempt_until_delivered=False, receiver_relative_value=True)) == "critical"
    )


def test_both_mechanisms_together_send_the_unseen_belief_first():
    """The repaired default: the first delivery of an unseen belief goes before re-tracking a known one."""
    assert _first_belief(BAACConfig()) == "fresh"


def test_preemption_still_applies_until_the_critical_finding_reaches_the_receiver():
    """A critical belief the receiver has never seen keeps absolute Level 0/1 priority (ch19)."""
    cfg = BAACConfig()
    ids = IdFactory(6)
    builder = UnitBuilder(ids, cfg)
    receiver = ReceiverKnowledge()
    crit = make_belief(ids, properties={"condition": "FAILED"}, severity=0.9)
    routine = make_belief(ids)
    entries = [_entry(builder, routine, None, 0.5), _entry(builder, crit, None, 0.9)]
    for level in (0, 1):
        assert _critical_undelivered(entries[1], level, receiver)
    channel = ChannelSim(
        [LinkProfile(name="a", bandwidth_bps=20_000.0, packet_loss=0.0, packet_bits=256)], seed=0
    )
    plan = schedule(entries, channel.link_states(0.0), channel, receiver, 0, 1.0, BAAC_POLICY, cfg, None)
    by_unit = {e.unit_id: e for e in entries}
    assert by_unit[plan[0].unit_id].content.unit.belief_ids[0] == crit.belief_id


def test_critical_undelivered_turns_off_once_the_alert_and_the_view_have_arrived():
    cfg = BAACConfig()
    entries, receiver, _, crit_id, _ = _setup(cfg)
    crit_entry = next(e for e in entries if e.content.unit.belief_ids[0] == crit_id)
    assert not _critical_undelivered(crit_entry, 0, receiver)
    assert not _critical_undelivered(crit_entry, 1, receiver)


def test_receiver_credit_is_zero_for_an_unseen_belief_and_partial_for_a_stale_view():
    cfg = BAACConfig()
    ids = IdFactory(7)
    builder = UnitBuilder(ids, cfg)
    receiver = ReceiverKnowledge()
    m0 = make_belief(ids)
    unseen = _entry(builder, m0, None, 0.5)
    assert receiver_credit(unseen, 1, receiver, cfg) == 0.0
    _ack_view(receiver, m0)
    m1 = make_belief(ids, belief_id=m0.belief_id, revision=m0.revision + 3)
    stale = _entry(builder, m1, receiver.known(m0.belief_id), 0.5)
    credit = receiver_credit(stale, 1, receiver, cfg)
    f1 = stale.content.option(1)
    assert f1 is not None
    assert credit == pytest.approx(f1.information_retained * cfg.stale_view_credit)
    assert 0.0 < credit < f1.information_retained


def test_the_first_alert_about_a_belief_is_never_discounted_but_later_alerts_are():
    """ch19 Level 0 is a control message: a stale view the receiver holds must not delay the first alert."""
    cfg = BAACConfig()
    ids = IdFactory(8)
    builder = UnitBuilder(ids, cfg)
    receiver = ReceiverKnowledge()
    routine = make_belief(ids)
    _ack_view(receiver, routine)  # the belief was reported as routine before it became critical
    found = make_belief(
        ids,
        belief_id=routine.belief_id,
        revision=routine.revision + 5,
        properties={"condition": "FAILED"},
        severity=0.9,
    )
    alerting = _entry(builder, found, receiver.known(routine.belief_id), 0.9)
    assert receiver_credit(alerting, 0, receiver, cfg) == 0.0
    assert receiver_credit(alerting, 1, receiver, cfg) > 0.0
    receiver.acknowledge({"kind": "alert", "belief_id": str(found.belief_id), "revision": found.revision})
    later = make_belief(
        ids,
        belief_id=routine.belief_id,
        revision=found.revision + 5,
        properties={"condition": "FAILED"},
        severity=0.95,
    )
    re_alert = _entry(builder, later, receiver.known(routine.belief_id), 0.9)
    assert receiver_credit(re_alert, 0, receiver, cfg) > cfg.information_retained[0]


@pytest.mark.parametrize("policy", list(BASELINE_POLICIES.values()))
def test_the_repair_flags_never_change_a_baseline_plan(policy):
    """Both mechanisms are gated on the BAAC ingredients, so no baseline may move when they are toggled."""
    plans = []
    for on in (False, True):
        cfg = BAACConfig(preempt_until_delivered=on, receiver_relative_value=on)
        entries, receiver, channel, _, _ = _setup(cfg)
        plans.append(
            [
                (p.unit_id, p.from_level, p.to_level, p.chunk_bits, p.reserved_bits)
                for p in schedule(
                    entries, channel.link_states(0.0), channel, receiver, 0, 1.0, policy, cfg, None
                )
            ]
        )
    assert plans[0] == plans[1]


def test_completion_feasibility_chooses_a_useful_unit_that_can_finish_before_horizon():
    cfg = BAACConfig(completion_horizon_s=1.0)
    ids = IdFactory(9)
    builder = UnitBuilder(ids, cfg)
    receiver = ReceiverKnowledge()
    large = _entry(builder, make_belief(ids, n_evidence=3), None, 0.7)
    small = _entry(builder, make_belief(ids), None, 0.3)
    large_bits = large.content.option(1).size_bits
    small_bits = small.content.option(1).size_bits
    assert large_bits > small_bits
    bandwidth = float(small_bits + 100)
    channel = ChannelSim(
        [LinkProfile(name="a", bandwidth_bps=bandwidth, latency_s=0.0, packet_loss=0.0, packet_bits=64)],
        seed=0,
    )
    plan = schedule(
        [large, small], channel.link_states(0.0), channel, receiver, 0, 1.0, BAAC_POLICY, cfg, None
    )
    assert plan
    assert plan[0].unit_id == small.unit_id


def test_completion_feasibility_protects_a_nearly_complete_increment():
    cfg = BAACConfig(completion_horizon_s=1.0)
    ids = IdFactory(10)
    builder = UnitBuilder(ids, cfg)
    receiver = ReceiverKnowledge()
    large = _entry(builder, make_belief(ids, n_evidence=3), None, 0.7)
    size = large.content.option(1).size_bits
    nearly_done = large.model_copy(update={"partial_level": 1, "partial_bits": size - 128})
    channel = ChannelSim(
        [LinkProfile(name="a", bandwidth_bps=256.0, latency_s=0.0, packet_loss=0.0, packet_bits=64)],
        seed=0,
    )
    plan = schedule(
        [nearly_done], channel.link_states(0.0), channel, receiver, 0, 1.0, BAAC_POLICY, cfg, None
    )
    assert plan and plan[0].unit_id == nearly_done.unit_id and plan[0].completes


def test_completion_feasibility_counts_pre_serialised_carry_once():
    cfg = BAACConfig(completion_horizon_s=1.0)
    ids = IdFactory(11)
    builder = UnitBuilder(ids, cfg)
    receiver = ReceiverKnowledge()
    entry = _entry(builder, make_belief(ids, n_evidence=0), None, 0.5)
    size = entry.content.option(1).size_bits
    channel = ChannelSim(
        [LinkProfile(name="a", bandwidth_bps=float(size), latency_s=0.0, packet_loss=0.0, packet_bits=64)],
        seed=0,
    )
    # Only half a second of new capacity remains before the horizon, but the other half has already been
    # serialised into carry. Counting the carry twice would incorrectly reject this completion.
    plan = schedule(
        [entry],
        channel.link_states(0.5),
        channel,
        receiver,
        int(0.5e9),
        0.1,
        BAAC_POLICY,
        cfg,
        {"a": size * 0.5},
    )
    assert plan


@pytest.mark.parametrize("policy", list(BASELINE_POLICIES.values()))
def test_completion_horizon_never_changes_a_baseline_plan(policy):
    plans = []
    for enabled in (False, True):
        cfg = BAACConfig(completion_feasibility=enabled, completion_horizon_s=0.1)
        entries, receiver, channel, _, _ = _setup(cfg)
        plans.append(
            [
                (p.unit_id, p.from_level, p.to_level, p.chunk_bits, p.reserved_bits)
                for p in schedule(
                    entries, channel.link_states(0.0), channel, receiver, 0, 1.0, policy, cfg, None
                )
            ]
        )
    assert plans[0] == plans[1]
