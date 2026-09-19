"""BAAC: deltas, sizes, scheduler budgets, collapse ordering, persistence/reconnect, overload, evidence-on-demand."""

import itertools
import pathlib
import re

import numpy as np
import torch

from conrad.communication import (
    BAAC_POLICY,
    BASELINE_POLICIES,
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
    payload_bits,
    schedule,
)
from conrad.communication.learned import BAACBatch, BAACHeadsConfig, baac_loss, build_heads, train_heads
from conrad.communication.queue import QueueEntry
from conrad.communication.units import UnitContent
from conrad.evaluation.decision_experiments.fixtures import make_belief, unc
from conrad.schemas.belief import Lifecycle
from conrad.schemas.comms import DeltaType
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import ProvenanceRecord
from conrad.schemas.timebase import stamp

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _now(t=0.0):
    return stamp(t, "SIM")


def _built(
    result: tuple[UnitContent, ProvenanceRecord] | None,
) -> tuple[UnitContent, ProvenanceRecord]:
    assert result is not None, "belief_unit produced no unit"
    return result


def test_delta_round_trip_covers_every_delta_type():
    ids = IdFactory(1)
    old = make_belief(ids, revision=0, properties={"condition": "NOMINAL"})
    new = make_belief(
        ids,
        belief_id=old.belief_id,
        revision=1,
        properties={"condition": "DAMAGED"},
        uncertainty=unc(uc=0.7),
        n_conflicts=1,
        relationships=[ids.new()],
        lifecycle=Lifecycle.RETIRED,
    )
    deltas = compute_deltas(belief_view(old), new)
    kinds = {d.delta_type for d in deltas}
    assert kinds == set(DeltaType) - {DeltaType.NEW_BELIEF}
    assert apply_deltas(belief_view(old), deltas) == belief_view(new)
    first = compute_deltas(None, old)
    assert first[0].delta_type is DeltaType.NEW_BELIEF and apply_deltas(None, first) == belief_view(old)
    assert compute_deltas(belief_view(new), new) == []


def test_missing_base_revision_requests_resync():
    ids = IdFactory(2)
    b0 = make_belief(ids, revision=0)
    b1 = make_belief(ids, belief_id=b0.belief_id, revision=1, uncertainty=unc(ua=0.5))
    b2 = make_belief(ids, belief_id=b0.belief_id, revision=2, uncertainty=unc(ua=0.9))
    store = ReceiverStore()
    assert (
        store.receive(
            {"kind": "deltas", "deltas": [d.model_dump(mode="json") for d in compute_deltas(None, b0)]}
        )
        is None
    )
    skip = compute_deltas(belief_view(b1), b2)  # receiver never got revision 1
    req = store.receive({"kind": "deltas", "deltas": [d.model_dump(mode="json") for d in skip]})
    assert req is not None and req.have_revision == 0 and req.needed_base == 1
    assert store.revision(b0.belief_id) == 0  # nothing overwritten


def test_sizes_are_measured_from_payloads_and_monotonic():
    ids = IdFactory(3)
    cfg = BAACConfig()
    b = make_belief(ids, n_evidence=2)
    sizes = dict.fromkeys(b.evidence_support, 1000)
    content, prov = _built(UnitBuilder(ids, cfg).belief_unit(b, None, 0.95, _now(), sizes))
    opts = content.unit.fidelity_levels
    assert [int(o.fidelity) for o in opts] == [0, 1, 2, 3, 4]
    assert all(a.size_bits < c.size_bits for a, c in itertools.pairwise(opts))
    assert opts[0].size_bits == payload_bits(content.increments[0])
    assert opts[4].size_bits - opts[3].size_bits == payload_bits(content.increments[4]) + 2 * 8 * 1000
    routine, _ = _built(UnitBuilder(ids, cfg).belief_unit(b, None, 0.2, _now(), sizes))
    assert int(routine.unit.fidelity_levels[0].fidelity) == 1  # no F0 alert for non-critical units
    assert prov.parent_records == b.provenance_refs


def test_scheduler_respects_bandwidth_and_energy_budgets():
    ids = IdFactory(4)
    cfg = BAACConfig(energy_budget_j_per_step=0.5)
    ch = ChannelSim([LinkProfile(name="a", bandwidth_bps=4000.0, energy_per_bit_j=1e-4)], seed=0)
    builder = UnitBuilder(ids, cfg)
    entries = [
        QueueEntry(content=_built(builder.belief_unit(make_belief(ids), None, 0.5, _now()))[0])
        for _ in range(6)
    ]
    plan = schedule(entries, ch.link_states(0.0), ch, ReceiverKnowledge(), 0, 1.0, BAAC_POLICY, cfg)
    assert plan
    assert sum(p.reserved_bits for p in plan) <= 4000.0
    assert sum(p.reserved_bits for p in plan) * 1e-4 <= 0.5 + 1e-9


def _run(policy, bandwidth, stream, seconds, path=None, outage=()):
    ids = IdFactory(5)
    ch = ChannelSim(
        [LinkProfile(name="a", bandwidth_bps=bandwidth, packet_loss=0.0, outages_s=outage)], seed=1
    )
    sender = BAACSender(ids, ch, BAACConfig(), policy, queue_path=path)
    store = ReceiverStore()
    for t in range(seconds):
        for m, v in stream.get(t, []):
            sender.offer(m, v, _now(float(t)))
        sender.step(float(t), 1.0, store.receive)
    return sender, store


def test_critical_info_survives_bandwidth_collapse_longer_than_bulk():
    ids = IdFactory(6)
    crit = make_belief(ids, uncertainty=unc(uc=0.7))
    bulk = [make_belief(ids) for _ in range(10)]
    stream = {0: [(m, 0.2) for m in bulk] + [(crit, 0.95)]}
    _, store = _run(BAAC_POLICY, 400.0, stream, 20)  # 400 bps: a few hundred bytes per step
    assert crit.belief_id in store.alerts or store.revision(crit.belief_id) is not None
    assert sum(store.revision(m.belief_id) is not None for m in bulk) < len(bulk)
    _, fifo = _run(BASELINE_POLICIES["C-B1_fifo"], 400.0, stream, 20)
    assert crit.belief_id not in fifo.alerts


def test_queue_persists_across_restart_and_reconnects(tmp_path):
    ids = IdFactory(7)
    msgs = [make_belief(ids) for _ in range(3)]
    path = tmp_path / "queue.json"
    sender, store = _run(
        BAAC_POLICY, 1000.0, {0: [(m, 0.5) for m in msgs]}, 5, path=path, outage=((0.0, 100.0),)
    )
    assert len(sender.queue) == 3 and not store.views  # link down: everything stored
    reloaded = PersistentQueue(BAACConfig().queue_capacity_bits, path)
    assert {e.unit_id for e in reloaded.entries()} == {e.unit_id for e in sender.queue.entries()}
    ch = ChannelSim([LinkProfile(name="a", bandwidth_bps=50_000.0, packet_loss=0.0)], seed=2)
    resumed = BAACSender(ids, ch, BAACConfig(), BAAC_POLICY, queue_path=path)
    for m in msgs:
        resumed.latest[m.belief_id] = (m, 0.5, None)
    for t in range(5):
        resumed.step(float(t), 1.0, store.receive)
    assert resumed.reevaluations == 1
    assert all(store.revision(m.belief_id) == m.revision for m in msgs)


def test_overload_drops_lowest_value_never_critical():
    ids = IdFactory(8)
    builder = UnitBuilder(ids, BAACConfig())
    q = PersistentQueue(capacity_bits=1)
    crit = _built(builder.belief_unit(make_belief(ids), None, 0.95, _now()))[0]
    low = _built(builder.belief_unit(make_belief(ids), None, 0.1, _now()))[0]
    q.put(crit, 0)
    dropped = q.put(low, 0)
    assert [d.unit_id for d in dropped] == [low.unit.unit_id]
    assert q.get(crit.unit.unit_id) is not None and q.overflow_events >= 1
    assert all(not d.critical for d in q.dropped)


def test_redundant_update_is_not_resent_and_evidence_is_discoverable():
    ids = IdFactory(9)
    m = make_belief(ids)
    sender, store = _run(BAAC_POLICY, 1e6, {0: [(m, 0.5)], 3: [(m, 0.5)]}, 6)
    assert store.revision(m.belief_id) == m.revision
    delta_tx = [t for t in sender.transmissions if "deltas" in t.payload["increments"]]
    assert len(delta_tx) == 1  # the repeat offer had zero novelty
    wanted = store.evidence_to_request(m.belief_id)
    assert set(wanted) <= set(m.evidence_support)


def test_multi_link_prefers_cheap_link_for_bulk_and_fast_link_for_critical():
    ids = IdFactory(10)
    ch = ChannelSim(
        [
            LinkProfile(
                name="acoustic", bandwidth_bps=1e5, latency_s=1.0, energy_per_bit_j=1e-4, packet_loss=0.0
            ),
            LinkProfile(
                name="optical", bandwidth_bps=1e5, latency_s=0.01, energy_per_bit_j=1e-6, packet_loss=0.0
            ),
        ],
        seed=0,
    )
    builder = UnitBuilder(ids, BAACConfig())
    crit = QueueEntry(content=_built(builder.belief_unit(make_belief(ids), None, 0.95, _now()))[0])
    plan = schedule([crit], ch.link_states(0.0), ch, ReceiverKnowledge(), 0, 1.0, BAAC_POLICY, BAACConfig())
    assert plan[0].link_name == "optical"


def test_channel_is_seeded_and_outage_blocks():
    p = LinkProfile(name="a", bandwidth_bps=1000.0, packet_loss=0.3, outages_s=((5.0, 6.0),))
    r1 = [ChannelSim([p], seed=3).transmit("a", 5000, 0.0) for _ in range(1)]
    r2 = [ChannelSim([p], seed=3).transmit("a", 5000, 0.0) for _ in range(1)]
    assert r1 == r2
    assert not ChannelSim([p], seed=3).transmit("a", 100, 5.5).delivered


def test_learned_value_heads_smoke():
    cfg = BAACHeadsConfig(unit_feature_dim=6, receiver_feature_dim=5, model_dim=16, num_heads=2, num_blocks=2)
    g = torch.Generator().manual_seed(0)
    batch = BAACBatch(
        units=torch.randn(2, 4, 6, generator=g),
        unit_mask=torch.ones(2, 4, dtype=torch.bool),
        receiver=torch.randn(2, 3, 5, generator=g),
        receiver_mask=torch.ones(2, 3, dtype=torch.bool),
        novelty=torch.ones(2, 4),
        value=torch.randn(2, 4, generator=g),
        info_loss=torch.rand(2, 4, 5, generator=g),
        deadline_violation=torch.zeros(2, 4),
    )
    model = build_heads(cfg)
    out = model(batch)
    assert out.info_loss.shape == (2, 4, 5)
    baac_loss(out, batch).backward()
    losses = train_heads(build_heads(cfg), [batch], epochs=10)
    assert losses[-1] < losses[0]


def test_communication_never_commands_hardware():
    pattern = re.compile(r"AllocatedCommand|WrenchCommand|command_gateway|RobotHardwareInterface")
    for path in (ROOT / "conrad" / "communication").rglob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path
    assert np.isfinite(BAACConfig().critical_value)
