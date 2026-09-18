"""Deterministic constrained greedy scheduler (ch33 BAAC exact implementation) and baselines.

    priority = (mission_value * novelty * confidence_adjustment) / max(predicted_bits, 1)

Units are selected in descending priority while respecting per-link bandwidth budget (with expected
ARQ overhead), link availability, deadlines, the energy budget and dependencies. BAAC selects
progressive fidelity increments; baselines are the same engine with the BAAC ingredients switched off:

    C-B0 send-all      full fidelity, no receiver knowledge, arrival order
    C-B1 FIFO          full fidelity, arrival order
    C-B2 fixed priority full fidelity, critical first then mission value
    C-B3 fixed compression  always F2, arrival order
    C-B4 value-per-bit  value/bits, single fidelity jump, no novelty/uncertainty term

implementation_status: EXPERIMENTAL_CANDIDATE (value terms) / FROZEN_CONTRACT (hard budgets)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from conrad.communication.channel import ChannelSim
from conrad.communication.config import BAACConfig
from conrad.communication.queue import QueueEntry
from conrad.communication.receiver import ReceiverKnowledge
from conrad.schemas.base import ConradModel
from conrad.schemas.comms import InformationType, LinkState, LinkStatus


class SchedulingPolicy(ConradModel):
    name: str
    progressive: bool = True
    fixed_level: int | None = None  # None = highest available level
    order: str = "value_per_bit"  # value_per_bit | fifo | class
    use_novelty: bool = True
    use_uncertainty: bool = True
    use_receiver_knowledge: bool = True


BAAC_POLICY = SchedulingPolicy(name="C-B10_baac")
BASELINE_POLICIES: dict[str, SchedulingPolicy] = {
    "C-B0_send_all": SchedulingPolicy(
        name="C-B0_send_all",
        progressive=False,
        order="fifo",
        use_novelty=False,
        use_uncertainty=False,
        use_receiver_knowledge=False,
    ),
    "C-B1_fifo": SchedulingPolicy(
        name="C-B1_fifo", progressive=False, order="fifo", use_novelty=False, use_uncertainty=False
    ),
    "C-B2_fixed_priority": SchedulingPolicy(
        name="C-B2_fixed_priority", progressive=False, order="class", use_novelty=False, use_uncertainty=False
    ),
    "C-B3_fixed_compression": SchedulingPolicy(
        name="C-B3_fixed_compression",
        progressive=False,
        fixed_level=2,
        order="fifo",
        use_novelty=False,
        use_uncertainty=False,
    ),
    "C-B4_value_per_bit": SchedulingPolicy(
        name="C-B4_value_per_bit", progressive=False, use_novelty=False, use_uncertainty=False
    ),
}


@dataclass(frozen=True)
class ScheduledIncrement:
    unit_id: UUID
    from_level: int
    to_level: int
    link_name: str
    bits: int  # nominal size of the whole increment
    chunk_bits: int  # nominal bits sent in this step (fragmentation)
    reserved_bits: int  # chunk inflated by expected ARQ retransmissions
    priority: float

    @property
    def completes(self) -> bool:
        return self.chunk_bits >= self.bits


def confidence_adjustment(entry: QueueEntry, to_level: int, cfg: BAACConfig) -> float:
    """ch19 'Uncertainty-conditioned communication'."""
    ua, ue, uc, _ = entry.content.uncertainty
    adj = 1.0
    if entry.critical and uc >= cfg.high_contradiction:
        adj *= 1.2  # operator input on a contested critical belief is valuable
    if ua >= cfg.high_aleatoric and to_level >= 3:
        adj *= 0.3  # do not spend bandwidth on noisy raw data
    if ue >= cfg.high_epistemic and to_level in (2, 3):
        adj *= 1.5  # selected evidence for external review
    return adj


def novelty(entry: QueueEntry, to_level: int, receiver: ReceiverKnowledge) -> float:
    unit = entry.content.unit
    rev = entry.content.new_revision
    if unit.content_type is InformationType.EVIDENCE:
        return 0.0 if all(str(e) in receiver.evidence for e in unit.evidence_ids) else 1.0
    if not unit.belief_ids or rev is None:
        return 1.0
    bid = unit.belief_ids[0]
    if to_level == 0:
        return 0.0 if receiver.alerted.get(bid, -1) >= rev else 1.0
    known = receiver.known_revision(bid)
    return 0.0 if (known is not None and known >= rev and to_level <= 1) else 1.0


def _options(entry: QueueEntry, level: int, policy: SchedulingPolicy) -> list[int]:
    levels = [lv for lv in entry.content.levels if lv > level]
    if not levels or (not policy.progressive and level >= 0):
        return []
    if policy.fixed_level is not None:
        allowed = [lv for lv in levels if lv <= policy.fixed_level]
        return [max(allowed)] if allowed else []
    if policy.progressive:
        return levels
    return [levels[-1]] if policy.order != "value_per_bit" else levels


def schedule(
    entries: Sequence[QueueEntry],
    links: Sequence[LinkState],
    channel: ChannelSim,
    receiver: ReceiverKnowledge,
    now_ns: int,
    dt_s: float,
    policy: SchedulingPolicy,
    cfg: BAACConfig,
) -> list[ScheduledIncrement]:
    """Allocate this step's link capacity (bandwidth x dt) to increments, fragmenting large ones.

    An increment already in progress can only be continued; its priority uses its REMAINING bits.
    FIFO / fixed-priority orders are strict (head-of-line blocking); value orders skip what cannot go.
    """
    budget = {ln.link_name: ln.bandwidth_bps * dt_s for ln in links if ln.status is not LinkStatus.DOWN}
    strict = policy.order in ("fifo", "class")
    by_link = {ln.link_name: ln for ln in links}
    energy_left = cfg.energy_budget_j_per_step
    planned: dict[UUID, int] = {e.unit_id: e.level for e in entries}
    partial: dict[UUID, tuple[int, int]] = {
        e.unit_id: (e.partial_level, e.partial_bits) for e in entries if e.partial_level is not None
    }
    saturated: set[UUID] = set()
    queued = {e.unit_id for e in entries}
    out: list[ScheduledIncrement] = []
    while True:
        best: tuple[tuple[float, ...], ScheduledIncrement] | None = None
        for e in entries:
            if e.unit_id in saturated:
                continue
            if any(d in queued and planned.get(d, -1) < 1 for d in e.content.unit.dependencies):
                continue
            cur = planned[e.unit_id]
            cur_opt = e.content.option(cur)
            cur_bits = 0 if cur_opt is None else cur_opt.size_bits
            cur_ret = 0.0 if cur_opt is None else cur_opt.information_retained
            in_progress = partial.get(e.unit_id)
            levels = [in_progress[0]] if in_progress is not None else _options(e, cur, policy)
            for lv in levels:
                opt = e.content.option(lv)
                assert opt is not None
                bits = opt.size_bits - cur_bits
                done = in_progress[1] if in_progress is not None else 0
                remaining = max(bits - done, 1)
                value = e.content.unit.mission_value * max(opt.information_retained - cur_ret, 0.0)
                if policy.use_novelty:
                    value *= novelty(e, lv, receiver)
                if policy.use_uncertainty:
                    value *= confidence_adjustment(e, lv, cfg)
                if policy.use_novelty and value <= 0:
                    continue
                pick = _pick_link(remaining, e, budget, by_link, channel, now_ns, energy_left)
                if pick is None and not strict:
                    continue
                prio = value / remaining
                t = e.content.unit.created_time_ns
                key: tuple[float, ...]
                if policy.order == "fifo":
                    key = (-float(t), -float(e.unit_id.int % 10**9), prio)
                elif policy.order == "class":
                    key = (float(e.critical), e.content.unit.mission_value, -float(t), prio)
                else:
                    key = (prio, -float(t), -float(lv))
                name, chunk = pick if pick is not None else ("", 0)
                inc = ScheduledIncrement(
                    e.unit_id,
                    cur,
                    lv,
                    name,
                    remaining,
                    chunk,
                    0 if not name else channel.expected_bits(name, chunk),
                    prio,
                )
                if best is None or key > best[0]:
                    best = (key, inc)
        if best is None or not best[1].link_name:
            return out
        inc = best[1]
        budget[inc.link_name] -= inc.reserved_bits
        if energy_left is not None:
            energy_left -= inc.reserved_bits * (by_link[inc.link_name].energy_per_bit_j or 0.0)
        if inc.completes:
            planned[inc.unit_id] = inc.to_level
            partial.pop(inc.unit_id, None)
        else:
            saturated.add(inc.unit_id)
        out.append(inc)


def _pick_link(
    remaining: int,
    entry: QueueEntry,
    budget: dict[str, float],
    links: dict[str, LinkState],
    channel: ChannelSim,
    now_ns: int,
    energy_left: float | None,
) -> tuple[str, int] | None:
    """Best link with capacity left this step, and the nominal chunk it can carry now."""
    feasible: list[tuple[LinkState, int]] = []
    for name, left in budget.items():
        pf = min(channel.packet_failure(name), 0.99)
        ln = links[name]
        cap = left
        if energy_left is not None and (ln.energy_per_bit_j or 0.0) > 0:
            cap = min(cap, energy_left / (ln.energy_per_bit_j or 1.0))
        chunk = min(remaining, int(cap * (1.0 - pf)))
        while chunk >= 1 and channel.expected_bits(name, chunk) > cap:  # reservation must fit the budget
            chunk -= 1
        if chunk < 1:
            continue
        deadline = entry.content.unit.deadline_ns
        if deadline is not None and ln.bandwidth_bps > 0:
            arrival = now_ns + int(
                (ln.latency_s + channel.expected_bits(name, remaining) / ln.bandwidth_bps) * 1e9
            )
            if arrival > deadline:
                continue
        feasible.append((ln, chunk))
    if not feasible:
        return None
    if entry.critical:
        ln, chunk = min(feasible, key=lambda f: (f[0].latency_s, -f[1], f[0].link_name))
    else:
        ln, chunk = min(feasible, key=lambda f: (f[0].energy_per_bit_j or 0.0, -f[1], f[0].link_name))
    return ln.link_name, chunk
