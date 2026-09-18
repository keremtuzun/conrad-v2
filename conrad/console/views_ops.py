"""Operational views: runtime timeline, command outcomes, faults, BAAC, mission files, truth records."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from conrad.persistence import db
from conrad.persistence.repository import Repository
from conrad.schemas.events import EventType, RuntimeEvent, Severity

# Raw actuator payloads are never displayed (ch37 DTOs: "no raw actuator payload").
REDACTED_KEYS = frozenset({"thruster_commands", "raw_command", "actuator_payload"})
STATE_TYPES = {
    EventType.RUN_STARTED,
    EventType.STATE_CHANGED,
    EventType.OPERATOR_ACTION,
    EventType.SAFE_HOLD_ENTERED,
    EventType.RUN_TERMINATED,
}
FAULT_TYPES = {EventType.FAULT_DETECTED, EventType.FAULT_INJECTED, EventType.SAFE_HOLD_ENTERED}
MISSION_CATEGORIES = (
    ("mcbr", ("candidate", "mcbr", "plan", "observation_plan")),
    ("needs", ("need",)),
    ("decisions", ("decision",)),
    ("contradictions", ("contradiction",)),
    ("navigation", ("trajector", "goal", "nav")),
    ("baac", ("transmission", "baac", "receiver", "queue", "comm")),
    ("faults", ("fault",)),
)


def redact(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            k: ("[raw actuator payload not shown]" if k in REDACTED_KEYS else redact(v))
            for k, v in payload.items()
        }
    if isinstance(payload, list):
        return [redact(v) for v in payload]
    return payload


@dataclass(frozen=True)
class CommandRow:
    command_id: str
    seq: int | None
    time_ns: int | None
    outcome: str  # ACCEPTED | REJECTED | SENT_NO_ACK
    reason_codes: tuple[str, ...]
    provenance_root: str | None
    source: str


def event_rows(events: list[RuntimeEvent], types: set[EventType], warn: bool) -> list[RuntimeEvent]:
    order = list(Severity)
    return [
        e
        for e in events
        if e.event_type in types
        or (warn and order.index(e.severity) >= order.index(Severity.WARNING))
        or (warn and e.availability.value != "AVAILABLE")
    ]


def lanes(events: list[RuntimeEvent]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for e in events:
        out.setdefault(e.event_type.value, []).append(e.sequence)
    return out


def command_rows(events: list[RuntimeEvent], repo: Repository | None) -> list[CommandRow]:
    rows: dict[str, dict[str, Any]] = {}
    for e in events:
        cid = str(e.payload.get("command_id", ""))
        if e.event_type is EventType.COMMAND_SENT:
            rows.setdefault(cid, {}).update(
                seq=e.sequence,
                time_ns=e.envelope.measurement_time_ns,
                outcome="SENT_NO_ACK",
                prov=e.payload.get("provenance_root"),
                source="event log",
            )
        elif e.event_type is EventType.COMMAND_ACK:
            r = rows.setdefault(cid, {"seq": e.sequence, "source": "event log"})
            r["outcome"] = "ACCEPTED" if e.payload.get("accepted") else "REJECTED"
            r["reasons"] = tuple(e.payload.get("reason_codes", ()))
        elif e.event_type is EventType.COMMAND_REJECTED:
            rows.setdefault(cid, {}).update(
                seq=e.sequence,
                time_ns=e.envelope.measurement_time_ns,
                outcome="REJECTED",
                reasons=tuple(e.payload.get("reason_codes", ())),
                source="event log",
            )
    if repo is not None:
        with repo.engine.connect() as conn:
            for cid, issued, accepted, reasons in conn.execute(
                select(
                    db.commands.c.command_id,
                    db.commands.c.issued_time_ns,
                    db.commands.c.accepted,
                    db.commands.c.reason_codes,
                ).order_by(db.commands.c.issued_time_ns)
            ):
                r = rows.setdefault(str(cid), {"source": "database"})
                r.setdefault("time_ns", issued)
                r["outcome"] = "ACCEPTED" if accepted else "REJECTED"
                r["reasons"] = tuple(json.loads(reasons))
                if r.get("source") == "event log":
                    r["source"] = "event log + database"
    return [
        CommandRow(
            cid,
            r.get("seq"),
            r.get("time_ns"),
            r.get("outcome", "SENT_NO_ACK"),
            tuple(r.get("reasons", ())),
            r.get("prov"),
            r.get("source", "event log"),
        )
        for cid, r in rows.items()
    ]


def categorize(files: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {name: {} for name, _ in MISSION_CATEGORIES}
    out["other"] = {}
    for rel, data in (files or {}).items():
        low = rel.lower()
        cat = next((name for name, keys in MISSION_CATEGORIES if any(k in low for k in keys)), "other")
        out[cat][rel] = data
    return out


def walk_dicts(obj: Any) -> Iterator[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_dicts(v)


def records(data: Any) -> list[dict[str, Any]]:
    """The record list of a mission file: a top-level list, or the first list of dicts inside a dict."""
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and all(isinstance(d, dict) for d in v):
                return list(v)
        return [data]
    return []


def plans(files: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        d
        for data in files.values()
        for d in walk_dicts(data)
        if "plan_id" in d and ("primary_action" in d or "rejected" in d)
    ]


def _position(d: dict[str, Any]) -> tuple[float, float, float] | None:
    pos = d.get("position_m") or d.get("position")
    if isinstance(pos, list | tuple) and len(pos) >= 2 and all(isinstance(v, int | float) for v in pos):
        return (float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0)
    return None


def positions(data: Any) -> list[tuple[float, float, float]]:
    """Every position found in a JSON tree, in document order (trajectory points, truth pose records)."""
    return [p for d in walk_dicts(data) if (p := _position(d)) is not None]


def flatten(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(obj, dict):
        return [row for k, v in obj.items() for row in flatten(v, f"{prefix}.{k}" if prefix else str(k))]
    if isinstance(obj, list) and not all(isinstance(v, int | float | str | bool) for v in obj):
        return [row for i, v in enumerate(obj) for row in flatten(v, f"{prefix}[{i}]")]
    return [(prefix, str(obj))]
