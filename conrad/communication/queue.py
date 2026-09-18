"""Persistent store-and-forward queue (ch16 'Persistent queue', ch19 'Store-and-forward').

Every mutation is written to disk atomically (temp file + ``os.replace``) when a path is given, so
the queue survives outages and restarts. Overload is bounded: the lowest-value NON-critical entries
are dropped first and every drop is recorded with a reason. Critical entries are never silently
dropped; if only critical entries remain above capacity the queue records an explicit overflow.

implementation_status: FROZEN_CONTRACT (semantics) / deterministic
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import UUID

from pydantic import Field

from conrad.communication.units import UnitContent
from conrad.schemas.base import ConradModel


class QueueEntry(ConradModel):
    content: UnitContent
    level: int = Field(default=-1, ge=-1, le=4, description="highest fidelity the receiver has ACKed")
    first_delivery_ns: int | None = None
    attempts: int = Field(default=0, ge=0)
    partial_level: int | None = Field(default=None, description="increment being fragmented across steps")
    partial_bits: int = Field(default=0, ge=0, description="nominal bits of that increment already delivered")

    @property
    def unit_id(self) -> UUID:
        return self.content.unit.unit_id

    @property
    def critical(self) -> bool:
        return self.content.critical

    def remaining_bits(self) -> int:
        top = self.content.unit.fidelity_levels[-1].size_bits
        cur = self.content.option(self.level)
        return top - (0 if cur is None else cur.size_bits)

    @property
    def complete(self) -> bool:
        return self.level >= self.content.levels[-1]


class DropRecord(ConradModel):
    unit_id: UUID
    reason: str
    critical: bool
    mission_value: float
    time_ns: int


class PersistentQueue:
    def __init__(self, capacity_bits: int, path: str | Path | None = None) -> None:
        self.capacity_bits = capacity_bits
        self.path = None if path is None else Path(path)
        self._entries: dict[UUID, QueueEntry] = {}
        self.dropped: list[DropRecord] = []
        self.overflow_events = 0
        if self.path is not None and self.path.exists():
            self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        assert self.path is not None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._entries = {}
        for raw in data["entries"]:
            e = QueueEntry.model_validate(raw)
            self._entries[e.unit_id] = e
        self.dropped = [DropRecord.model_validate(d) for d in data.get("dropped", [])]
        self.overflow_events = int(data.get("overflow_events", 0))

    def save(self) -> None:
        if self.path is None:
            return
        data = {
            "entries": [e.model_dump(mode="json") for e in self._entries.values()],
            "dropped": [d.model_dump(mode="json") for d in self.dropped],
            "overflow_events": self.overflow_events,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ mutation
    def put(self, content: UnitContent, now_ns: int) -> list[DropRecord]:
        self._entries[content.unit.unit_id] = QueueEntry(content=content)
        dropped = self.enforce_capacity(now_ns)
        self.save()
        return dropped

    def replace(self, entry: QueueEntry) -> None:
        self._entries[entry.unit_id] = entry
        self.save()

    def remove(self, unit_id: UUID, reason: str, now_ns: int, record: bool = True) -> None:
        entry = self._entries.pop(unit_id, None)
        if entry is not None and record:
            self.dropped.append(
                DropRecord(
                    unit_id=unit_id,
                    reason=reason,
                    critical=entry.critical,
                    mission_value=entry.content.unit.mission_value,
                    time_ns=now_ns,
                )
            )
        self.save()

    def enforce_capacity(self, now_ns: int) -> list[DropRecord]:
        before = len(self.dropped)
        while self.queued_bits > self.capacity_bits:
            victims = [e for e in self._entries.values() if not e.critical]
            if not victims:
                self.overflow_events += 1  # explicit, never a silent critical drop
                break
            victim = min(
                victims, key=lambda e: (e.content.unit.mission_value, -e.content.unit.created_time_ns)
            )
            self.remove(victim.unit_id, "OVERLOAD_LOWEST_VALUE", now_ns)
        return self.dropped[before:]

    # ------------------------------------------------------------------ views
    def entries(self) -> list[QueueEntry]:
        return sorted(self._entries.values(), key=lambda e: (e.content.unit.created_time_ns, e.unit_id.int))

    def get(self, unit_id: UUID) -> QueueEntry | None:
        return self._entries.get(unit_id)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def queued_bits(self) -> int:
        return sum(e.remaining_bits() for e in self._entries.values())
