"""BAAC sender: offer -> persistent queue -> re-evaluate -> schedule -> channel -> receiver (ch16, ch19).

Model 1 decides communication INTENT (and mission value); BAAC decides representation, fidelity,
timing and link. Nothing here chooses mission goals.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from conrad.communication.channel import ChannelSim
from conrad.communication.config import BAACConfig
from conrad.communication.queue import DropRecord, PersistentQueue, QueueEntry
from conrad.communication.receiver import ReceiverKnowledge, ResyncRequest
from conrad.communication.scheduler import ScheduledIncrement, SchedulingPolicy, policy_by_name, schedule
from conrad.communication.units import EvidenceSizes, UnitBuilder, UnitContent
from conrad.schemas.belief import BeliefMessage
from conrad.schemas.comms import CommunicationState, Fidelity, LinkState, LinkStatus, Transmission
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import NS_PER_S, TimeStamp, stamp

Sink = Callable[[dict[str, Any]], ResyncRequest | None]


@dataclass
class _InFlight:
    arrive_ns: int
    increments: list[dict[str, Any]]


class BAACSender:
    def __init__(
        self,
        id_factory: IdFactory,
        channel: ChannelSim,
        config: BAACConfig | None = None,
        policy: SchedulingPolicy | None = None,
        queue_path: str | Path | None = None,
    ) -> None:
        self._ids = id_factory
        self.config = config or BAACConfig()
        self.channel = channel
        self.policy = policy if policy is not None else policy_by_name(self.config.scheduler_policy)
        self.carry_bits: dict[str, float] = {}  # unused sub-packet capacity carried to the next step
        self.coalesced = 0  # queued units superseded by a fresher revision of the same belief
        self._deferred: dict[UUID, float] = {}  # fresher revisions waiting for an in-progress increment
        self.builder = UnitBuilder(id_factory, self.config)
        self.queue = PersistentQueue(self.config.queue_capacity_bits, queue_path)
        self.receiver_model = ReceiverKnowledge()
        self.latest: dict[UUID, tuple[BeliefMessage, float, EvidenceSizes | None]] = {}
        self.in_flight: list[_InFlight] = []
        self.transmissions: list[Transmission] = []
        self.first_delivery: dict[UUID, int] = {}  # unit_id -> first arrival ns at any fidelity
        self.unit_created: dict[UUID, tuple[int, bool, UUID | None]] = {}
        self.energy_used_j = 0.0
        self._link_was_up = False
        self.reevaluations = 0
        self.current_arrival_ns = 0  # arrival time of the increment currently handed to the sink

    # ------------------------------------------------------------------ intake
    def offer(
        self,
        message: BeliefMessage,
        mission_value: float,
        now: TimeStamp,
        evidence_sizes: EvidenceSizes | None = None,
    ) -> list[DropRecord]:
        self.latest[message.belief_id] = (message, mission_value, evidence_sizes)
        created = now.time_ns
        if self.policy.use_novelty:  # redundancy-aware coalescing is a BAAC feature, not a baseline one
            mine = [
                e
                for e in self.queue.entries()
                if e.content.unit.belief_ids == (message.belief_id,) and e.content.new_revision is not None
            ]
            if any(e.partial_level is not None for e in mine):
                # An increment of this belief is mid-transmission: superseding it would throw away the delivered
                # fragments, and a belief revised faster than one increment can cross the link would never
                # arrive (livelock). Finish it; the fresher revision is rebuilt as a delta against what the
                # receiver then holds (``_release_deferred``).
                self._deferred[message.belief_id] = max(
                    mission_value, self._deferred.get(message.belief_id, 0.0)
                )
                return []
            mission_value = max(mission_value, self._deferred.pop(message.belief_id, 0.0))
            for e in mine:
                created = min(created, e.content.unit.created_time_ns)
                mission_value = max(mission_value, e.content.unit.mission_value)
                self.queue.remove(e.unit_id, "SUPERSEDED_BY_NEWER_REVISION", now.time_ns, record=False)
                self.coalesced += 1
        known = self.receiver_model.known(message.belief_id) if self.policy.use_receiver_knowledge else None
        built = self.builder.belief_unit(
            message, known, mission_value, now, evidence_sizes, created_time_ns=created
        )
        if built is None:
            return []
        return self._enqueue(built[0], now.time_ns)

    def offer_evidence(
        self, evidence_ids: list[UUID], mission_value: float, now: TimeStamp, sizes: EvidenceSizes
    ) -> None:
        self._enqueue(self.builder.evidence_unit(evidence_ids, mission_value, now, sizes), now.time_ns)

    def _enqueue(self, content: UnitContent, now_ns: int) -> list[DropRecord]:
        u = content.unit
        self.unit_created[u.unit_id] = (
            u.created_time_ns,
            content.critical,
            u.belief_ids[0] if u.belief_ids else None,
        )
        return self.queue.put(content, now_ns)

    def handle_resync(self, request: ResyncRequest, now: TimeStamp) -> None:
        self.receiver_model.forget(request.belief_id)
        latest = self.latest.get(request.belief_id)
        if latest is not None:
            self.offer(latest[0], latest[1], now, latest[2])

    # ------------------------------------------------------------------ re-evaluation
    def reevaluate(self, now: TimeStamp) -> None:
        """On reconnection: drop stale/redundant entries and rebuild deltas against current K_R."""
        self.reevaluations += 1
        for e in self.queue.entries():
            unit = e.content.unit
            deadline = unit.deadline_ns
            if deadline is not None and deadline < now.time_ns and not e.critical:
                self.queue.remove(e.unit_id, "DEADLINE_EXPIRED", now.time_ns)
                continue
            if (
                not unit.belief_ids
                or e.content.new_revision is None
                or not self.policy.use_receiver_knowledge
            ):
                continue
            bid = unit.belief_ids[0]
            message, value, sizes = self.latest[bid]
            known = self.receiver_model.known(bid)
            rebuilt = self.builder.belief_unit(
                message,
                known,
                max(value, unit.mission_value),
                now,
                sizes,
                created_time_ns=unit.created_time_ns,
            )
            self.queue.remove(e.unit_id, "REBUILT_ON_REEVALUATION", now.time_ns, record=False)
            if rebuilt is None:
                self.queue.dropped.append(
                    DropRecord(
                        unit_id=e.unit_id,
                        reason="RECEIVER_ALREADY_HAS_IT",
                        critical=e.critical,
                        mission_value=unit.mission_value,
                        time_ns=now.time_ns,
                    )
                )
                continue
            content = rebuilt[0]
            if e.first_delivery_ns is not None or e.unit_id in self.first_delivery:
                # keep the latency bookkeeping of the original unit
                self.first_delivery[content.unit.unit_id] = self.first_delivery.get(e.unit_id, now.time_ns)
            self._enqueue(content, now.time_ns)

    # ------------------------------------------------------------------ transmission
    def step(self, now_s: float, dt_s: float, sink: Sink | None = None) -> list[Transmission]:
        now = stamp(now_s, self.channel.clock_domain)
        links = self.channel.link_states(now_s)
        up = any(ln.status is not LinkStatus.DOWN for ln in links)
        if up and not self._link_was_up:
            self.reevaluate(now)
        self._link_was_up = up
        entries = self.queue.entries()
        carried = dict(self.carry_bits)
        plan = schedule(
            entries,
            links,
            self.channel,
            self.receiver_model,
            now.time_ns,
            dt_s,
            self.policy,
            self.config,
            self.carry_bits,
        )
        self._update_carry(links, dt_s, plan, bool(entries))
        sent: list[Transmission] = []
        for inc in plan:
            entry = self.queue.get(inc.unit_id)
            if entry is None or entry.level != inc.from_level:
                continue
            result = self.channel.transmit(inc.link_name, inc.chunk_bits, now_s)
            self.energy_used_j += result.energy_j
            # capacity carried from earlier steps was already being serialised then: do not count that time twice
            credit = min(carried.get(inc.link_name, 0.0), float(inc.reserved_bits))
            carried[inc.link_name] = carried.get(inc.link_name, 0.0) - credit
            bw = self.channel.bandwidth(inc.link_name, now_s)
            if result.delivered_time_s is not None and credit > 0 and bw > 0:
                floor_s = now_s + self.channel.profiles[inc.link_name].latency_s
                result = result.model_copy(
                    update={"delivered_time_s": max(floor_s, result.delivered_time_s - credit / bw)}
                )
            completed = result.delivered and inc.completes
            increments = [
                entry.content.increments[lv]
                for lv in sorted(entry.content.increments)
                if inc.from_level < lv <= inc.to_level
            ]
            tx = Transmission(
                transmission_id=self._ids.new(),
                unit_id=inc.unit_id,
                trace_id=entry.content.unit.trace_id,
                link_name=inc.link_name,
                fidelity=Fidelity(inc.to_level),
                bits=max(1, result.bits_used),
                sent_time_ns=int(now_s * NS_PER_S),
                delivered=completed,
                delivered_time_ns=int(result.delivered_time_s * NS_PER_S)
                if completed and result.delivered_time_s is not None
                else None,
                payload={
                    "increments": [str(i.get("kind")) for i in increments],
                    "fragment": not inc.completes,
                    "chunk_delivered": result.delivered,
                },
            )
            sent.append(tx)
            if not result.delivered:  # chunk lost after ARQ retries: progress of this chunk is lost
                self.queue.replace(entry.model_copy(update={"attempts": entry.attempts + 1}))
                continue
            if not completed:
                done = entry.partial_bits if entry.partial_level == inc.to_level else 0
                self.queue.replace(
                    entry.model_copy(
                        update={"partial_level": inc.to_level, "partial_bits": done + inc.chunk_bits}
                    )
                )
                continue
            assert tx.delivered_time_ns is not None
            for incr in increments:
                self.receiver_model.acknowledge(incr)  # link-layer ACK (ARQ success)
            self.first_delivery.setdefault(inc.unit_id, tx.delivered_time_ns)
            self.in_flight.append(_InFlight(tx.delivered_time_ns, increments))
            updated = entry.model_copy(
                update={
                    "level": inc.to_level,
                    "partial_level": None,
                    "partial_bits": 0,
                    "first_delivery_ns": entry.first_delivery_ns or tx.delivered_time_ns,
                }
            )
            if updated.complete:
                self.queue.remove(inc.unit_id, "DELIVERED", now.time_ns, record=False)
            else:
                self.queue.replace(updated)
        self.transmissions.extend(sent)
        self._release_deferred(now)
        if sink is not None:
            self.poll(int((now_s + dt_s) * NS_PER_S), sink, now)
        return sent

    def _release_deferred(self, now: TimeStamp) -> None:
        """Offer deferred fresher revisions once no increment of their belief is mid-transmission."""
        busy = {
            e.content.unit.belief_ids[0]
            for e in self.queue.entries()
            if e.partial_level is not None and e.content.unit.belief_ids
        }
        for bid in [b for b in self._deferred if b not in busy]:
            message, value, sizes = self.latest[bid]
            self.offer(message, max(value, self._deferred[bid]), now, sizes)

    def _update_carry(
        self, links: list[LinkState], dt_s: float, plan: list[ScheduledIncrement], backlog: bool
    ) -> None:
        """Carry unused capacity only while data waits on an UP link, capped at one packet's reservation.

        A link cannot bank idle capacity: the carry is cleared when the queue is empty or the link is down.
        """
        used: dict[str, int] = {}
        for inc in plan:
            used[inc.link_name] = used.get(inc.link_name, 0) + inc.reserved_bits
        for ln in links:
            name = ln.link_name
            if ln.status is LinkStatus.DOWN or not backlog:
                self.carry_bits[name] = 0.0
                continue
            left = ln.bandwidth_bps * dt_s + self.carry_bits.get(name, 0.0) - used.get(name, 0)
            cap = float(self.channel.expected_bits(name, self.channel.profiles[name].packet_bits))
            self.carry_bits[name] = max(0.0, min(left, cap))

    def poll(self, until_ns: int, sink: Sink, now: TimeStamp) -> None:
        ready = sorted((f for f in self.in_flight if f.arrive_ns <= until_ns), key=lambda f: f.arrive_ns)
        self.in_flight = [f for f in self.in_flight if f.arrive_ns > until_ns]
        for flight in ready:
            self.current_arrival_ns = flight.arrive_ns
            for incr in flight.increments:
                req = sink(incr)
                if req is not None:
                    self.handle_resync(req, now)

    def state(self, now_s: float) -> CommunicationState:
        return CommunicationState(
            timestamp=stamp(now_s, self.channel.clock_domain),
            links=tuple(self.channel.link_states(now_s)),
            queue_depth=len(self.queue),
            queued_bits=self.queue.queued_bits,
            last_contact_ns=max(
                (t.delivered_time_ns or 0 for t in self.transmissions if t.delivered), default=None
            ),
        )

    def pending(self) -> list[QueueEntry]:
        return self.queue.entries()
