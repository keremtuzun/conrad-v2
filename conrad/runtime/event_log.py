"""Append-only structured event log (JSONL + optional DB index) (ch34, ch36).

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TextIO
from uuid import UUID

from conrad.schemas.base import digest_of
from conrad.schemas.belief import Availability
from conrad.schemas.envelope import MessageEnvelope
from conrad.schemas.events import EventType, RuntimeEvent, Severity
from conrad.schemas.ids import IdFactory
from conrad.settings import redact

Clock = Callable[[], int]


class RunContext:
    """Identity shared by every record of one run: IDs, clock, deterministic ID factory."""

    def __init__(
        self,
        run_id: UUID,
        mission_id: UUID,
        scenario_id: UUID | None,
        ids: IdFactory,
        clock_ns: Clock,
        clock_domain: str,
        producer_version: str,
    ) -> None:
        self.run_id = run_id
        self.mission_id = mission_id
        self.scenario_id = scenario_id
        self.ids = ids
        self.clock_ns = clock_ns
        self.clock_domain = clock_domain
        self.producer_version = producer_version

    def envelope(
        self,
        measurement_time_ns: int | None = None,
        correlation_id: UUID | None = None,
        causation: tuple[UUID, ...] = (),
    ) -> MessageEnvelope:
        now = self.clock_ns()
        m = now if measurement_time_ns is None else measurement_time_ns
        return MessageEnvelope(
            message_id=self.ids.new(),
            scenario_id=self.scenario_id,
            mission_id=self.mission_id,
            run_id=self.run_id,
            measurement_time_ns=m,
            created_time_ns=max(now, m),
            clock_domain=self.clock_domain,
            producer_version=self.producer_version,
            correlation_id=correlation_id or self.ids.new(),
            causation_ids=causation,
        )


class EventLog:
    def __init__(
        self, ctx: RunContext, path: Path | None = None, sink: Callable[[RuntimeEvent], None] | None = None
    ) -> None:
        self.ctx = ctx
        self.path = path
        self._sink = sink
        self._sequence = 0
        self.events: list[RuntimeEvent] = []
        self._handle: TextIO | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")

    def emit(
        self,
        event_type: EventType,
        module: str,
        trace_id: UUID,
        payload: dict[str, Any] | None = None,
        severity: Severity = Severity.INFO,
        measurement_time_ns: int | None = None,
        availability: Availability = Availability.AVAILABLE,
        causation: tuple[UUID, ...] = (),
    ) -> RuntimeEvent:
        body = redact(payload or {})
        event = RuntimeEvent(
            event_id=self.ctx.ids.new(),
            sequence=self._sequence,
            envelope=self.ctx.envelope(measurement_time_ns, correlation_id=trace_id, causation=causation),
            event_type=event_type,
            severity=severity,
            availability=availability,
            trace_id=trace_id,
            module=module,
            module_version=self.ctx.producer_version,
            payload_digest=digest_of(body),
            payload=body,
        )
        self._sequence += 1
        self.events.append(event)
        if self._handle is not None:
            self._handle.write(event.canonical_json() + "\n")
            self._handle.flush()
        if self._sink is not None:
            self._sink(event)
        return event

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def by_trace(self, trace_id: UUID) -> list[RuntimeEvent]:
        return [e for e in self.events if e.trace_id == trace_id]

    def of_type(self, event_type: EventType) -> list[RuntimeEvent]:
        return [e for e in self.events if e.event_type is event_type]


def read_events(path: Path) -> Iterator[RuntimeEvent]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield RuntimeEvent.model_validate(json.loads(line))


def event_signature(events: list[RuntimeEvent]) -> list[tuple[int, str, str, str]]:
    """Order, type, module and payload digest: what replay must reproduce (CC-10, SS-02)."""
    return [(e.sequence, e.event_type.value, e.module, e.payload_digest) for e in events]
