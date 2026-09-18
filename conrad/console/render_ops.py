"""Operational sections: runtime timeline, contradictions, needs/MCBR, navigation, commands, BAAC, faults."""

from __future__ import annotations

from typing import Any

from conrad.console import svg
from conrad.console.html import badge, compact, esc, missing, records_table, secs, section, table
from conrad.console.views_belief import BeliefView
from conrad.console.views_ops import (
    FAULT_TYPES,
    STATE_TYPES,
    CommandRow,
    event_rows,
    flatten,
    lanes,
    plans,
    records,
    redact,
    walk_dicts,
)
from conrad.schemas.events import EventType, RuntimeEvent


def _event_table(evs: list[RuntimeEvent]) -> str:
    rows = [
        [
            esc(e.sequence),
            esc(secs(e.envelope.measurement_time_ns)),
            esc(e.event_type.value),
            badge(e.severity.value, "sev"),
            esc(e.availability.value),
            esc(e.module),
            f"<code>{esc(compact(redact(e.payload)))}</code>",
        ]
        for e in evs
    ]
    return table(
        ["seq", "time", "type", "severity", "availability", "module", "payload"],
        rows,
        [e.sequence for e in evs],
    )


def json_view(data: Any) -> str:
    """Scalars as key/value rows; every list of records as its own table (shape-agnostic)."""
    if not isinstance(data, dict):
        return records_table(records(data))
    lists = {
        k: v for k, v in data.items() if isinstance(v, list) and v and all(isinstance(d, dict) for d in v)
    }
    rest = {k: v for k, v in data.items() if k not in lists}
    out = (
        table(["key", "value"], [[f"<code>{esc(k)}</code>", esc(v)] for k, v in flatten(rest)])
        if rest
        else ""
    )
    return out + "".join(f"<h4>{esc(k)}</h4>{records_table(v)}" for k, v in lists.items())


def mission_files(files: dict[str, Any], what: str, prefix: str = "mission/") -> str:
    if not files:
        return missing(what)
    return "".join(f"<h3>{esc(prefix + name)}</h3>{json_view(data)}" for name, data in files.items())


def health_section(events: list[RuntimeEvent]) -> str:
    if not events:
        return section("health", "Robot health and mission/runtime state", missing("event log"))
    state = event_rows(events, STATE_TYPES, warn=True)
    body = svg.event_lanes("event timeline by type", lanes(events), len(events))
    body += "<h3>State changes, warnings and degraded availability</h3>"
    body += _event_table(state) if state else missing("state events", "none recorded")
    return section("health", "Robot health and mission/runtime state timeline", body)


def contradictions_section(
    beliefs: list[BeliefView], files: dict[str, Any], decisions: dict[str, Any]
) -> str:
    rows = []
    for b in beliefs:
        if b.revisions and b.revisions[-1].uncertainty[2] > 0:
            head = b.revisions[-1]
            rows.append(
                [f"<code>{esc(b.belief_id)}</code>", esc(head.revision), esc(f"{head.uncertainty[2]:.4g}")]
            )
    body = (
        "<p>Beliefs whose head revision reports U_C &gt; 0, as recorded by the belief plane. The console does "
        "not judge contradictions itself.</p>"
    )
    body += (
        table(["belief", "head rev", "U_C"], rows)
        if rows
        else missing("active contradictions", "no head has U_C > 0")
    )
    edges = [
        d for data in decisions.values() for d in walk_dicts(data) if d.get("edge_type") == "CONTRADICTS"
    ]
    body += "<h3>CONTRADICTS edges in decision claim graphs</h3>" + (
        records_table(edges) if edges else missing("CONTRADICTS edges", "none in mission/ decision files")
    )
    return section(
        "contradictions", "Active contradictions", body + mission_files(files, "mission contradiction files")
    )


def mcbr_section(events: list[RuntimeEvent], cats: dict[str, dict[str, Any]], mission_present: bool) -> str:
    needs = [
        d
        for data in cats["needs"].values()
        for d in walk_dicts(data)
        if "need_id" in d and "question_type" in d
    ]
    needs += [
        d
        for data in cats["decisions"].values()
        for d in walk_dicts(data)
        if "need_id" in d and "question_type" in d
    ]
    need_events = [
        e
        for e in events
        if e.event_type is EventType.PLAN_PROPOSED
        or (e.event_type is EventType.DECISION_MADE and e.payload.get("action") == "REQUEST_INFORMATION")
    ]
    body = "<h3>InformationNeeds</h3>" + (
        records_table(needs) if needs else missing("InformationNeed records")
    )
    body += "<h3>Need / plan events</h3>" + (
        _event_table(need_events) if need_events else missing("need events", "none")
    )
    body += "<h3>MCBR plans: selected observation and rejected candidates</h3>"
    found = plans(cats["mcbr"]) + plans(cats["decisions"])
    for p in found:
        act = p.get("primary_action") or {}
        pose = (act.get("pose") or {}).get("position_m")
        body += (
            f"<p>plan <code>{esc(p.get('plan_id'))}</code> status {esc(p.get('status'))}: selected view "
            f"<code>{esc(pose)}</code>, visibility {esc(act.get('predicted_visibility'))}, expected gain "
            f"{esc(act.get('expected_information_gain'))}, cost <code>{esc(compact(act.get('expected_cost')))}</code></p>"
        )
        rej = [
            [
                f"<code>{esc((r.get('action') or {}).get('action_id'))}</code>",
                f"<code>{esc(((r.get('action') or {}).get('pose') or {}).get('position_m'))}</code>",
                esc(", ".join(r.get("reason_codes", ()))),
            ]
            for r in p.get("rejected", ())
        ]
        body += (
            table(["rejected candidate", "pose", "reason codes"], rej)
            if rej
            else missing("rejected candidates", "none")
        )
    if not found:
        body += missing(
            "MCBR plan records", "no mission/ plan file present" if not mission_present else "none found"
        )
    body += mission_files(cats["mcbr"], "MCBR candidate tables")
    return section("mcbr", "InformationNeeds and MCBR candidates", body)


def navigation_section(events: list[RuntimeEvent], files: dict[str, Any]) -> str:
    nav = [e for e in events if e.event_type in (EventType.GOAL_ACCEPTED, EventType.TRAJECTORY_PLANNED)]
    body = "<h3>Goal / trajectory events</h3>" + (
        _event_table(nav) if nav else missing("navigation events", "none")
    )
    return section(
        "navigation",
        "Navigation goals and trajectories",
        body + mission_files(files, "mission trajectory files"),
    )


def commands_section(rows: list[CommandRow], events: list[RuntimeEvent]) -> str:
    acc = sum(r.outcome == "ACCEPTED" for r in rows)
    rej = sum(r.outcome == "REJECTED" for r in rows)
    body = (
        f"<p>{len(rows)} commands: {badge('ACCEPTED', 'out')} {acc} · {badge('REJECTED', 'out')} {rej} · "
        f"{badge('SENT_NO_ACK', 'out')} {len(rows) - acc - rej}. Raw actuator payloads are not shown.</p>"
    )
    table_rows = [
        [
            esc(r.seq),
            esc(secs(r.time_ns)),
            f"<code>{esc(r.command_id)}</code>",
            badge(r.outcome, "out"),
            esc(", ".join(r.reason_codes) or "none"),
            f"<code>{esc(r.provenance_root)}</code>",
            esc(r.source),
        ]
        for r in rows
    ]
    body += (
        table(
            ["seq", "time", "command", "outcome", "reason codes", "provenance root", "source"],
            table_rows,
            [r.seq for r in rows],
        )
        if rows
        else missing("commands", "none recorded")
    )
    safety = [e for e in events if e.event_type in (EventType.ACTION_REJECTED, EventType.SAFE_HOLD_ENTERED)]
    body += "<h3>Safety rejections of proposed actions</h3>" + (
        _event_table(safety) if safety else missing("safety rejections", "none recorded")
    )
    return section("command-outcomes", "Safety and command outcomes", body)


def baac_section(events: list[RuntimeEvent], files: dict[str, Any]) -> str:
    tx = [e for e in events if e.event_type is EventType.TRANSMISSION]
    bits = sum(int(e.payload.get("bits", 0) or 0) for e in tx)
    body = f"<p>{len(tx)} transmission events, {bits} bits in total (from event payloads).</p>"
    body += _event_table(tx) if tx else missing("transmissions", "none recorded")
    return section(
        "baac",
        "BAAC queue, transmissions and receiver state",
        body + mission_files(files, "BAAC queue / receiver files"),
    )


def faults_section(events: list[RuntimeEvent], files: dict[str, Any]) -> str:
    faults = event_rows(events, FAULT_TYPES, warn=False)
    body = _event_table(faults) if faults else missing("faults", "no fault events recorded")
    return section("faults", "Faults", body + mission_files(files, "mission fault files"))


def metrics_section(metrics: Any) -> str:
    if metrics is None:
        return section("metrics", "Run metrics (reports/metrics.json)", missing("reports/metrics.json"))
    rows = [[f"<code>{esc(k)}</code>", esc(v)] for k, v in flatten(metrics)]
    return section("metrics", "Run metrics (reports/metrics.json)", table(["key", "value"], rows))


TRUTH_HIDDEN_NOTE = (
    "An evaluation-only truth record exists in this bundle. It is hidden outside evaluation mode (ch37 UI-08); "
    "re-render with include_truth=True to evaluate."
)


def truth_section(truth: dict[str, Any] | None, present: bool) -> str:
    """Only called with ``truth`` in evaluation mode; otherwise it shows at most an existence note."""
    title = "TRUTH (evaluation only)"
    if truth is None:
        body = f'<p class="missing">{esc(TRUTH_HIDDEN_NOTE)}</p>' if present else missing("truth/")
        return section("truth-evaluation-only", "Truth record (evaluation mode only)", body)
    body = (
        '<p class="truth-label">TRUTH (evaluation only). Hidden simulator state recorded for scoring. The robot '
        "never saw it, and it is never mixed into any belief view on this page.</p>"
    )
    body += mission_files(truth, "truth records", prefix="truth/")
    return section("truth-evaluation-only", title, body, "truth")
