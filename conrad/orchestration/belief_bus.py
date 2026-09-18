"""Versioned Belief Bus (ch2, ch16 'Do not dump the entire world into Model 1', ch28 runtime contract).

In-process, deterministic and asynchronous in style: ``publish`` only validates, versions and enqueues;
``dispatch`` later fans the pending messages out. One failing consumer never blocks another (the
failure is recorded, the consumer is marked DEGRADED, the remaining consumers still get the message).

Boundaries enforced here:

* a belief ID is owned by the domain that first published it; another domain can never write it;
* revisions are strictly monotonic per belief;
* Model 2 children receive other domains' messages only through ``receive_context`` and every such
  delivery is logged as ``SourceType.CROSS_DOMAIN_CONTEXT``; the bus never touches child state;
* ``query`` returns one coherent ``BeliefSnapshot`` that carries every revision plus per-domain
  availability, so a consumer can detect mixed-time or stale input.

BELIEF PLANE: no truth types may be imported here.

implementation_status: FROZEN_CONTRACT (interfaces) - in-process transport only
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from uuid import UUID

from pydantic import Field

from conrad.schemas.base import ConradModel, digest_of
from conrad.schemas.belief import Availability, BeliefMessage, BeliefQuery, BeliefSnapshot, KnowledgeStatus
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import SourceType
from conrad.schemas.timebase import NS_PER_S
from conrad.schemas.world import Domain

Subscriber = Callable[[BeliefMessage], None]
ContextReceiver = Callable[[Sequence[BeliefMessage]], None]

REASON_ACCEPTED = "ACCEPTED"
REASON_NON_MONOTONIC = "NON_MONOTONIC_REVISION"
REASON_FOREIGN_WRITE = "FOREIGN_DOMAIN_WRITE"
REASON_DUPLICATE_MESSAGE = "DUPLICATE_MESSAGE_ID"


class BeliefBusConfig(ConradModel):
    """Freshness is per publishing domain and comes from configuration, never from a constant."""

    default_freshness_s: float = Field(default=30.0, gt=0)
    freshness_s: dict[str, float] = Field(default_factory=dict, description="Domain.value -> seconds")
    history_limit: int = Field(default=8, gt=0, description="revisions retained per belief")

    def freshness_for(self, domain: Domain) -> float:
        return self.freshness_s.get(domain.value, self.default_freshness_s)


class PublishReceipt(ConradModel):
    message_id: UUID
    belief_id: UUID
    revision: int
    accepted: bool
    reason_code: str
    bus_sequence: int | None = None


class BusFailure(ConradModel):
    """A consumer raised while handling a message. Recorded, never hidden, never propagated."""

    consumer: str
    message_id: UUID
    error_type: str
    error: str
    bus_sequence: int


class ContextDelivery(ConradModel):
    target_domain: Domain
    source_domain: Domain
    message_id: UUID
    belief_id: UUID
    revision: int
    source_type: SourceType = SourceType.CROSS_DOMAIN_CONTEXT


class _Subscription:
    def __init__(self, name: str, callback: Subscriber, domains: frozenset[Domain] | None) -> None:
        self.name = name
        self.callback = callback
        self.domains = domains
        self.failed = False


class _Child:
    def __init__(self, domain: Domain, receiver: ContextReceiver) -> None:
        self.domain = domain
        self.receiver = receiver
        self.failed = False


def _overlaps(a: SpatialSupport, b: SpatialSupport) -> bool:
    if a.frame_id != b.frame_id:
        return False
    return all(
        abs(a.center_m[i] - b.center_m[i]) <= a.half_extent_m[i] + b.half_extent_m[i] + 1e-9 for i in range(3)
    )


class BeliefBus:
    def __init__(self, id_factory: IdFactory, config: BeliefBusConfig | None = None) -> None:
        self._ids = id_factory
        self.config = config or BeliefBusConfig()
        self._heads: dict[UUID, BeliefMessage] = {}
        self._history: dict[UUID, deque[BeliefMessage]] = {}
        self._owner: dict[UUID, Domain] = {}
        self._seen_messages: set[UUID] = set()
        self._pending: deque[tuple[int, BeliefMessage]] = deque()
        self._subs: list[_Subscription] = []
        self._children: dict[Domain, _Child] = {}
        self._reported: dict[Domain, Availability] = {}
        self._last_publish_ns: dict[Domain, int] = {}
        self._sequence = 0
        self.failures: list[BusFailure] = []
        self.context_deliveries: list[ContextDelivery] = []
        self.rejected: list[PublishReceipt] = []

    # ------------------------------------------------------------------ publishing
    def publish(self, message: BeliefMessage) -> PublishReceipt:
        reason = REASON_ACCEPTED
        if message.message_id in self._seen_messages:
            reason = REASON_DUPLICATE_MESSAGE
        elif self._owner.get(message.belief_id, message.domain) is not message.domain:
            reason = REASON_FOREIGN_WRITE
        else:
            head = self._heads.get(message.belief_id)
            if head is not None and message.revision <= head.revision:
                reason = REASON_NON_MONOTONIC
        if reason != REASON_ACCEPTED:
            receipt = PublishReceipt(
                message_id=message.message_id,
                belief_id=message.belief_id,
                revision=message.revision,
                accepted=False,
                reason_code=reason,
            )
            self.rejected.append(receipt)
            return receipt
        self._sequence += 1
        self._seen_messages.add(message.message_id)
        self._owner.setdefault(message.belief_id, message.domain)
        self._heads[message.belief_id] = message
        hist = self._history.setdefault(message.belief_id, deque(maxlen=self.config.history_limit))
        hist.append(message)
        self._last_publish_ns[message.domain] = max(
            self._last_publish_ns.get(message.domain, 0), message.timestamp.time_ns
        )
        self._pending.append((self._sequence, message))
        return PublishReceipt(
            message_id=message.message_id,
            belief_id=message.belief_id,
            revision=message.revision,
            accepted=True,
            reason_code=REASON_ACCEPTED,
            bus_sequence=self._sequence,
        )

    def publish_many(self, messages: Sequence[BeliefMessage]) -> list[PublishReceipt]:
        return [self.publish(m) for m in messages]

    def report_availability(self, domain: Domain, availability: Availability) -> None:
        """Children report their own health; an UNAVAILABLE child is never queried for fresh state."""
        self._reported[domain] = availability

    # ------------------------------------------------------------------ consumers
    def subscribe(self, name: str, callback: Subscriber, domains: Sequence[Domain] | None = None) -> None:
        if any(s.name == name for s in self._subs):
            raise ValueError(f"subscriber {name!r} already registered")
        self._subs.append(_Subscription(name, callback, None if domains is None else frozenset(domains)))

    def unsubscribe(self, name: str) -> None:
        self._subs = [s for s in self._subs if s.name != name]

    def attach_child(self, domain: Domain, receive_context: ContextReceiver) -> None:
        """Register a Model 2 child. It will only ever see OTHER domains, and only as context."""
        self._children[domain] = _Child(domain, receive_context)

    def dispatch(self, max_messages: int | None = None) -> int:
        """Deliver pending messages in publish order. Returns the number of messages dispatched."""
        count = 0
        while self._pending and (max_messages is None or count < max_messages):
            sequence, message = self._pending.popleft()
            count += 1
            for sub in self._subs:
                if sub.domains is not None and message.domain not in sub.domains:
                    continue
                try:
                    sub.callback(message)
                except Exception as exc:  # isolation boundary: record and continue with the others
                    sub.failed = True
                    self._record_failure(f"subscriber:{sub.name}", message, exc, sequence)
            for child in self._children.values():
                if child.domain is message.domain:
                    continue
                try:
                    child.receiver((message,))
                except Exception as exc:  # isolation boundary
                    child.failed = True
                    self._record_failure(f"child:{child.domain.value}", message, exc, sequence)
                    continue
                self.context_deliveries.append(
                    ContextDelivery(
                        target_domain=child.domain,
                        source_domain=message.domain,
                        message_id=message.message_id,
                        belief_id=message.belief_id,
                        revision=message.revision,
                    )
                )
        return count

    def _record_failure(self, consumer: str, message: BeliefMessage, exc: Exception, sequence: int) -> None:
        self.failures.append(
            BusFailure(
                consumer=consumer,
                message_id=message.message_id,
                error_type=type(exc).__name__,
                error=str(exc),
                bus_sequence=sequence,
            )
        )

    def clear_failure(self, domain: Domain) -> None:
        if domain in self._children:
            self._children[domain].failed = False

    @property
    def pending(self) -> int:
        return len(self._pending)

    # ------------------------------------------------------------------ availability
    def availability(self, now_ns: int) -> dict[str, Availability]:
        out: dict[str, Availability] = {}
        for domain in Domain:
            out[domain.value] = self._domain_availability(domain, now_ns)
        return out

    def _domain_availability(self, domain: Domain, now_ns: int) -> Availability:
        reported = self._reported.get(domain)
        if reported is Availability.UNAVAILABLE:
            return Availability.UNAVAILABLE
        last = self._last_publish_ns.get(domain)
        if last is None:
            return Availability.UNAVAILABLE
        if (now_ns - last) / NS_PER_S > self.config.freshness_for(domain):
            return Availability.STALE
        if reported is Availability.STALE:
            return Availability.STALE
        child = self._children.get(domain)
        heads_degraded = any(
            m.domain is domain and m.publisher_availability is Availability.DEGRADED
            for m in self._heads.values()
        )
        if reported is Availability.DEGRADED or heads_degraded or (child is not None and child.failed):
            return Availability.DEGRADED
        return Availability.AVAILABLE

    def is_stale(self, message: BeliefMessage, now_ns: int, max_age_s: float | None = None) -> bool:
        limit = self.config.freshness_for(message.domain) if max_age_s is None else max_age_s
        return (now_ns - message.timestamp.time_ns) / NS_PER_S > limit

    # ------------------------------------------------------------------ queries
    def head(self, belief_id: UUID) -> BeliefMessage | None:
        return self._heads.get(belief_id)

    def history(self, belief_id: UUID) -> tuple[BeliefMessage, ...]:
        return tuple(self._history.get(belief_id, ()))

    def query(self, query: BeliefQuery, now_ns: int, max_age_s: float | None = None) -> BeliefSnapshot:
        """Coherent reply taken from one bus sequence. ``max_age_s`` drops older messages entirely."""
        selected: list[BeliefMessage] = []
        for message in self._heads.values():
            if not self._matches(message, query):
                continue
            if max_age_s is not None and self.is_stale(message, now_ns, max_age_s):
                continue
            selected.append(message)
        selected.sort(key=lambda m: (-m.timestamp.time_ns, m.domain.value, m.belief_id.int))
        truncated = len(selected) > query.max_results
        selected = selected[: query.max_results]
        if query.requested_fields:
            wanted = set(query.requested_fields)
            selected = [
                m.model_copy(update={"state_summary": tuple(c for c in m.state_summary if c.name in wanted)})
                for m in selected
            ]
        provenance: dict[str, object] = {
            "bus_sequence": self._sequence,
            "revisions": {str(m.belief_id): m.revision for m in selected},
            "stale_belief_ids": [str(m.belief_id) for m in selected if self.is_stale(m, now_ns)],
            "truncated": truncated,
            "query_digest": digest_of(query.model_dump(mode="json")),
        }
        if query.include_provenance:
            provenance["provenance_refs"] = {
                str(m.belief_id): [str(p) for p in m.provenance_refs] for m in selected
            }
        return BeliefSnapshot(
            snapshot_id=self._ids.new(),
            created_time_ns=now_ns,
            messages=tuple(selected),
            domain_availability=self.availability(now_ns),
            provenance=provenance,
        )

    @staticmethod
    def _matches(m: BeliefMessage, q: BeliefQuery) -> bool:
        if q.domain is not None and m.domain is not q.domain:
            return False
        if q.belief_ids and m.belief_id not in q.belief_ids:
            return False
        if q.entity_ids and (m.world_entity_id is None or m.world_entity_id not in q.entity_ids):
            return False
        if q.region is not None and (m.spatial_support is None or not _overlaps(m.spatial_support, q.region)):
            return False
        if q.time_range_ns is not None and not (
            q.time_range_ns[0] <= m.timestamp.time_ns <= q.time_range_ns[1]
        ):
            return False
        if not q.include_predictions and m.knowledge_status is KnowledgeStatus.PREDICTED:
            return False
        if (
            q.min_observational_uncertainty is not None
            and m.uncertainty.observational < q.min_observational_uncertainty
        ):
            return False
        return not (
            q.min_contradiction_uncertainty is not None
            and m.uncertainty.contradiction < q.min_contradiction_uncertainty
        )
