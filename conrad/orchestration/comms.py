"""BAAC over a constrained ChannelSim link to a shore-side ReceiverStore. DEPLOYMENT PLANE.

Every changed belief is offered with a mission value from config; critical findings (a mission-critical
component whose condition is directly observed as not INTACT) are offered at the critical value.
Critical-alert latency follows the COM-BAAC-E001 convention: first arrival of the same belief at a
revision >= the critical one, minus the offer time.

Scheduler policy is config (``baac.scheduler_policy``: C-B10_baac or a baseline C-B0..C-B4). For the gate-I7
harness, ``baac.shadow_arms`` adds counterfactual communication arms: each has its own BAAC sender, channel
and receiver, receives the IDENTICAL offer stream at the identical times over a link with the identical
profile and channel seed, and never feeds back into the mission. Only the primary arm's receiver and link
state are visible to decision routing, so every arm is scored on the same mission.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from pydantic import Field

from conrad.communication import BAACConfig, BAACSender, ChannelSim, LinkProfile, ReceiverStore
from conrad.communication.receiver import ResyncRequest
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.orchestration.services import RuntimeServices
from conrad.schemas.base import ConradModel
from conrad.schemas.belief import BeliefMessage, KnowledgeStatus
from conrad.schemas.comms import LinkState, Transmission
from conrad.schemas.events import EventType
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import NS_PER_S, TimeStamp
from conrad.schemas.world import Domain

MODULE = "conrad.orchestration.comms"
NOT_INTACT = ("DEGRADED", "SEVERE", "FAILED")
PRIMARY = "primary"


class ShadowArm(ConradModel):
    """A counterfactual communication arm of the I7 harness (``baac.shadow_arms`` entries)."""

    name: str
    policy: str
    bandwidth_factor: float = Field(default=1.0, ge=0, description="of the mission link bandwidth")


class CommsArm:
    """One sender -> channel -> receiver chain, with the traces the I7 harness reports."""

    def __init__(
        self, name: str, profile: LinkProfile, seed: int, ids: IdFactory, config: BAACConfig
    ) -> None:
        self.name = name
        self.profile = profile
        self.channel = ChannelSim([profile], seed)
        self.sender = BAACSender(ids, self.channel, config)
        self.receiver = ReceiverStore()
        self.arrivals: dict[tuple[UUID, int], int] = {}
        self.receipts: list[dict[str, Any]] = []
        self.queue_trace: list[tuple[float, int, int, int]] = []
        self.dropped = 0

    def sink(self, increment: dict[str, Any]) -> ResyncRequest | None:
        kind = increment.get("kind")
        key: tuple[UUID, int] | None = None
        if kind == "alert":
            key = (UUID(str(increment["belief_id"])), int(increment["revision"]))
        elif kind == "deltas" and increment.get("deltas"):
            d = increment["deltas"][0]
            key = (UUID(str(d["belief_id"])), int(d["new_revision"]))
        arrival = int(self.sender.current_arrival_ns)
        if key is not None:
            self.arrivals.setdefault(key, arrival)
        before = None if key is None else self.receiver.revision(key[0])
        req = self.receiver.receive(increment)
        self.receipts.append(
            {
                "t_ns": arrival,
                "kind": str(kind),
                "belief_id": None if key is None else str(key[0]),
                "revision": None if key is None else key[1],
                "applied": key is not None and kind == "deltas" and self.receiver.revision(key[0]) != before,
            }
        )
        return req

    def step(self, now_s: float, dt_s: float) -> list[Transmission]:
        sent = self.sender.step(now_s, dt_s, self.sink)
        q = self.sender.queue
        self.queue_trace.append(
            (round(now_s, 3), len(q), int(q.queued_bits), sum(1 for e in q.entries() if e.critical))
        )
        return sent

    def report(self) -> dict[str, Any]:
        """JSON-safe traces (deployment-side facts only; scoring against truth happens in evaluation)."""
        s = self.sender
        txs = []
        for t in s.transmissions:
            created = s.unit_created.get(t.unit_id)
            txs.append(
                {
                    "sent_ns": t.sent_time_ns,
                    "delivered_ns": t.delivered_time_ns,
                    "delivered": t.delivered,
                    "bits": t.bits,
                    "fidelity": int(t.fidelity),
                    "belief_id": None if created is None or created[2] is None else str(created[2]),
                    "critical": bool(created[1]) if created is not None else False,
                    "fragment": bool(t.payload.get("fragment", False)),
                }
            )
        drops: dict[str, int] = {}
        for d in s.queue.dropped:
            drops[d.reason] = drops.get(d.reason, 0) + 1
        return {
            "policy": s.policy.name,
            "bandwidth_bps": self.profile.bandwidth_bps,
            "transmissions": txs,
            "receipts": self.receipts,
            "queue_trace": self.queue_trace,
            "arrivals": [
                [str(b), r, ns] for (b, r), ns in sorted(self.arrivals.items(), key=lambda kv: kv[1])
            ],
            "receiver_revisions": {
                str(k): self.receiver.revision(k) for k in sorted(self.receiver.views, key=str)
            },
            "receiver_alerts": {
                str(k): int(v["revision"])
                for k, v in sorted(self.receiver.alerts.items(), key=lambda kv: str(kv[0]))
            },
            "receiver_applied": {str(k): list(v) for k, v in self.receiver.applied.items()},
            "duplicate_contributions": self.receiver.duplicate_contributions(),
            "stale_ignored": self.receiver.stale_ignored,
            "resync_requests": len(self.receiver.resync_requests),
            "sender_latest_revisions": {str(k): v[0].revision for k, v in s.latest.items()},
            "coalesced": s.coalesced,
            "drop_reasons": drops,
            "reevaluations": s.reevaluations,
            "energy_j": float(s.energy_used_j),
        }


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
        raw = dict(cfg.baac)
        shadows = [ShadowArm.model_validate(a) for a in raw.pop("shadow_arms", [])]
        self.baac_config = BAACConfig(**raw)
        self.arms: dict[str, CommsArm] = {
            PRIMARY: CommsArm(PRIMARY, self.profile, seed, s.ids.child("baac"), self.baac_config)
        }
        for arm in shadows:
            if arm.name in self.arms:
                raise ValueError(f"duplicate comms arm name {arm.name!r}")
            profile = self.profile.model_copy(
                update={"bandwidth_bps": self.profile.bandwidth_bps * arm.bandwidth_factor}
            )
            conf = self.baac_config.model_copy(update={"scheduler_policy": arm.policy})
            self.arms[arm.name] = CommsArm(arm.name, profile, seed, s.ids.child(f"baac-{arm.name}"), conf)
        primary = self.arms[PRIMARY]
        self.channel, self.sender, self.receiver = primary.channel, primary.sender, primary.receiver
        self.arrivals = primary.arrivals
        self.critical_ids = set(critical_registry_ids)
        self.critical_offers: list[tuple[UUID, int, int]] = []
        self.offer_log: list[tuple[UUID, int, int, float]] = []  # (belief, revision, t_ns, value) to BAAC
        self.intent_latest: dict[UUID, tuple[BeliefMessage, float]] = {}  # everything Model 1 wants reported
        self.offered = 0
        self._pending: dict[UUID, tuple[BeliefMessage, float]] = {}
        self._last_offer_ns: dict[UUID, int] = {}
        self._critical_offered: set[UUID] = set()

    @property
    def dropped(self) -> int:
        return self.arms[PRIMARY].dropped

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
            known = self.intent_latest.get(m.belief_id)
            if known is None or m.revision >= known[0].revision:
                self.intent_latest[m.belief_id] = (m, max(value, known[1] if known else 0.0))

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
            for arm in self.arms.values():  # identical offer stream for every arm
                arm.dropped += len(arm.sender.offer(m, value, now))
            self.offered += 1
            self.offer_log.append((bid, m.revision, now.time_ns, value))
            if first_critical:  # latency is measured from the FIRST critical offer of each belief
                self._critical_offered.add(bid)
                self.critical_offers.append((m.belief_id, m.revision, now.time_ns))

    def step(self, now_s: float, dt_s: float) -> None:
        self.flush(TimeStamp(time_ns=round(now_s * NS_PER_S), clock_domain=self.channel.clock_domain))
        for name, arm in self.arms.items():
            sent = arm.step(now_s, dt_s)
            if name != PRIMARY:
                continue
            for tx in sent:
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
            "policy": self.sender.policy.name,
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
            "shadow_arms": sorted(n for n in self.arms if n != PRIMARY),
        }

    def receiver_state(self) -> dict[str, Any]:
        return {
            "views": {str(k): self.receiver.revision(k) for k in sorted(self.receiver.views, key=str)},
            "alerts": {str(k): v for k, v in sorted(self.receiver.alerts.items(), key=lambda kv: str(kv[0]))},
            "stale_ignored": self.receiver.stale_ignored,
            "resync_requests": len(self.receiver.resync_requests),
        }

    def harness_report(self) -> dict[str, Any]:
        """Per-arm traces plus the shared offer stream (gate I7 harness)."""
        return {
            "link": self.profile.model_dump(mode="json"),
            "critical_value": self.baac_config.critical_value,
            "critical_offers": [[str(b), r, ns] for b, r, ns in self.critical_offers],
            "offer_log": [[str(b), r, ns, v] for b, r, ns, v in self.offer_log],
            "intent_latest": {str(b): [m.revision, v] for b, (m, v) in self.intent_latest.items()},
            "arms": {name: arm.report() for name, arm in self.arms.items()},
        }
