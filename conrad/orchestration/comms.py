"""BAAC over a constrained ChannelSim link to a shore-side ReceiverStore. DEPLOYMENT PLANE.

Every changed belief is offered with a mission value from config; critical findings (a mission-critical
component whose condition is directly observed as not INTACT) are offered at the critical value.
Critical-alert latency follows the COM-BAAC-E001 convention: first arrival of the same belief at a
revision >= the critical one, minus the offer time.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from conrad.communication import BAACConfig, BAACSender, ChannelSim, LinkProfile, ReceiverStore
from conrad.communication.receiver import ResyncRequest
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.orchestration.services import RuntimeServices
from conrad.schemas.belief import BeliefMessage, KnowledgeStatus
from conrad.schemas.comms import LinkState
from conrad.schemas.events import EventType
from conrad.schemas.timebase import NS_PER_S, TimeStamp
from conrad.schemas.world import Domain

MODULE = "conrad.orchestration.comms"
NOT_INTACT = ("DEGRADED", "SEVERE", "FAILED")


class ShoreLink:
    def __init__(
        self, s: RuntimeServices, cfg: MissionRuntimeConfig, seed: int, critical_registry_ids: Sequence[UUID]
    ) -> None:
        self.s, self.cfg = s, cfg
        link = cfg.link
        self.profile = LinkProfile(
            name=link.name,
            bandwidth_bps=link.bandwidth_bps,
            latency_s=link.latency_s,
            packet_loss=link.packet_loss,
            bit_error_rate=link.bit_error_rate,
            energy_per_bit_j=link.energy_per_bit_j,
            packet_bits=link.packet_bits,
            outages_s=link.outages_s,
        )
        self.channel = ChannelSim([self.profile], seed)
        self.baac_config = BAACConfig(**cfg.baac)
        self.sender = BAACSender(s.ids.child("baac"), self.channel, self.baac_config)
        self.receiver = ReceiverStore()
        self.critical_ids = set(critical_registry_ids)
        self.critical_offers: list[tuple[UUID, int, int]] = []
        self.arrivals: dict[tuple[UUID, int], int] = {}
        self.offered = 0
        self.dropped = 0
        self._pending: dict[UUID, tuple[BeliefMessage, float]] = {}
        self._last_offer_ns: dict[UUID, int] = {}
        self._critical_offered: set[UUID] = set()

    def mission_value(self, m: BeliefMessage) -> float:
        if m.domain is Domain.TECHNICAL and m.world_entity_id in self.critical_ids:
            cond = next((c for c in m.state_summary if c.name == "condition"), None)
            if cond is not None and cond.status is KnowledgeStatus.OBSERVED and str(cond.value) in NOT_INTACT:
                return self.cfg.critical_finding_value
        return float(self.cfg.routine_value.get(m.domain.value, 0.1))

    def offer(self, messages: Sequence[BeliefMessage], now: TimeStamp, floor: float = 0.0) -> None:
        """Coalesce: keep the latest revision per belief; ``flush`` hands them to BAAC at a bounded rate."""
        for m in messages:
            value = max(self.mission_value(m), floor)
            prev = self._pending.get(m.belief_id)
            if prev is None or m.revision >= prev[0].revision:
                self._pending[m.belief_id] = (m, max(value, prev[1] if prev else 0.0))

    def flush(self, now: TimeStamp) -> None:
        critical = self.baac_config.critical_value
        for bid in sorted(self._pending, key=str):
            m, value = self._pending[bid]
            last = self._last_offer_ns.get(bid)
            first_critical = value >= critical and bid not in self._critical_offered
            if (
                not first_critical
                and last is not None
                and (now.time_ns - last) / NS_PER_S < self.cfg.reoffer_interval_s
            ):
                continue
            del self._pending[bid]
            self._last_offer_ns[bid] = now.time_ns
            drops = self.sender.offer(m, value, now)
            self.offered += 1
            self.dropped += len(drops)
            if first_critical:  # latency is measured from the FIRST critical offer of each belief
                self._critical_offered.add(bid)
                self.critical_offers.append((m.belief_id, m.revision, now.time_ns))

    def _sink(self, increment: dict[str, Any]) -> ResyncRequest | None:
        kind = increment.get("kind")
        key: tuple[UUID, int] | None = None
        if kind == "alert":
            key = (UUID(str(increment["belief_id"])), int(increment["revision"]))
        elif kind == "deltas" and increment.get("deltas"):
            d = increment["deltas"][0]
            key = (UUID(str(d["belief_id"])), int(d["new_revision"]))
        if key is not None and self.sender.current_arrival_ns is not None:
            self.arrivals.setdefault(key, int(self.sender.current_arrival_ns))
        return self.receiver.receive(increment)

    def step(self, now_s: float, dt_s: float) -> None:
        self.flush(TimeStamp(time_ns=round(now_s * NS_PER_S), clock_domain=self.channel.clock_domain))
        for tx in self.sender.step(now_s, dt_s, self._sink):
            self.s.emit(
                EventType.TRANSMISSION,
                MODULE,
                tx.trace_id,
                {
                    "transmission_id": str(tx.transmission_id),
                    "unit_id": str(tx.unit_id),
                    "link": tx.link_name,
                    "fidelity": int(tx.fidelity),
                    "bits": tx.bits,
                    "delivered": tx.delivered,
                    "sent_time_ns": tx.sent_time_ns,
                    "delivered_time_ns": tx.delivered_time_ns,
                },
                measurement_time_ns=tx.sent_time_ns,
            )

    def link_state(self, t_s: float) -> LinkState:
        return self.channel.link_state(self.profile.name, t_s)

    def reported_revision(self, belief_id: UUID) -> int | None:
        return self.receiver.revision(belief_id)

    def metrics(self) -> dict[str, Any]:
        latencies = []
        for bid, rev, t_offer in self.critical_offers:
            got = [ns for (b, r), ns in self.arrivals.items() if b == bid and r >= rev]
            if got:
                latencies.append((min(got) - t_offer) / NS_PER_S)
        txs = self.sender.transmissions
        return {
            "bits_sent": int(sum(t.bits for t in txs)),
            "transmissions": len(txs),
            "delivered": sum(1 for t in txs if t.delivered),
            "offers": self.offered,
            "dropped": self.dropped,
            "critical_offers": len(self.critical_offers),
            "critical_delivered": len(latencies),
            "critical_alert_latency_s": latencies,
            "critical_alert_latency_min_s": min(latencies) if latencies else None,
            "energy_j": float(self.sender.energy_used_j),
            "receiver_beliefs": len(self.receiver.views),
            "receiver_alerts": len(self.receiver.alerts),
            "queue_bits": int(self.sender.queue.queued_bits),
        }

    def receiver_state(self) -> dict[str, Any]:
        return {
            "views": {str(k): self.receiver.revision(k) for k in sorted(self.receiver.views, key=str)},
            "alerts": {str(k): v for k, v in sorted(self.receiver.alerts.items(), key=lambda kv: str(kv[0]))},
            "stale_ignored": self.receiver.stale_ignored,
            "resync_requests": len(self.receiver.resync_requests),
        }
