"""Assemble the single self-contained console page, or the integrity-error page."""

from __future__ import annotations

from pathlib import Path

from conrad.console import render_belief as rb
from conrad.console import render_ops as ro
from conrad.console.assets import CSP, CSS, JS
from conrad.console.bundle import IntegrityFailure, LoadedBundle, open_bundle
from conrad.console.html import esc, missing, script_json, section, table
from conrad.console.views_belief import belief_views, pose_track, provenance_view
from conrad.console.views_ops import categorize, command_rows, redact

EVAL_BANNER = (
    '<header class="banner truth-label" id="evaluation-mode">EVALUATION MODE: this page includes '
    "evaluation-only TRUTH from truth/. The robot never saw it; it is not part of any belief view.</header>"
)

BANNER = (
    '<header class="banner">Conrad V2 Operator and Replay Console'
    "<small>Read-only explanatory surface over a replay bundle. It is not a source of truth and not a decision "
    "authority, and it has no command capability.</small></header>"
)
NAV = (
    ("overview", "Run bundle"),
    ("replay-controls", "Replay"),
    ("health", "Health / state"),
    ("pose-track", "Pose track"),
    ("belief-status-timeline", "Belief status"),
    ("uncertainty-channels", "Uncertainty"),
    ("provenance-explorer", "Provenance"),
    ("contradictions", "Contradictions"),
    ("mcbr", "Needs / MCBR"),
    ("navigation", "Navigation"),
    ("command-outcomes", "Commands"),
    ("baac", "BAAC"),
    ("faults", "Faults"),
    ("metrics", "Metrics"),
    ("truth-evaluation-only", "Truth (eval mode)"),
)


def _doc(title: str, body: str, script: str = "") -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{CSP}">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="referrer" content="no-referrer">'
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body>{body}{script}</body></html>"
    )


def integrity_page(failure: IntegrityFailure) -> str:
    items = "".join(f"<li><code>{esc(p)}</code></li>" for p in failure.problems)
    body = (
        '<div class="error-page" id="integrity-error"><h1>BUNDLE INTEGRITY FAILURE</h1>'
        f"<p>The run bundle at <code>{esc(failure.run_dir)}</code> failed digest verification. Nothing from it "
        "is displayed. Missing or corrupt items:</p>"
        f"<ul>{items}</ul></div>"
    )
    return _doc("Bundle integrity failure", body)


def _overview(b: LoadedBundle) -> str:
    m = b.manifest
    parts = {
        "events.jsonl": f"{len(b.events)} events" if b.events else "not present",
        "database": b.db_note,
        "mission/": f"{len(b.mission)} files" if b.mission is not None else "not present",
        "truth/ (evaluation only)": (
            f"{len(b.truth)} files (evaluation mode)"
            if b.truth is not None
            else "present, hidden outside evaluation mode"
            if b.truth_present
            else "not present"
        ),
        "reports/metrics.json": "present" if b.metrics is not None else "not present",
        "config.resolved.yaml": "present" if b.config is not None else "not present",
    }
    body = (
        f"<p>run <code>{esc(m.run_id)}</code> · {esc(m.architecture_id)} / {esc(m.stack_id)} · schema "
        f"{esc(m.schema_version)} · manifest digest <code>{esc(m.manifest_digest())}</code></p>"
        f"<p>Verified: {len(m.files)} run files and {len(m.object_digests)} referenced objects hash correctly.</p>"
    )
    body += table(["bundle part", "state"], [[esc(k), esc(v)] for k, v in parts.items()])
    body += "<h3>Replay inputs</h3>" + table(
        ["key", "value"], [[f"<code>{esc(k)}</code>", esc(v)] for k, v in sorted(m.replay_inputs.items())]
    )
    if b.warnings:
        body += "<h3>Warnings</h3><ul>" + "".join(f"<li>{esc(w)}</li>" for w in b.warnings) + "</ul>"
    return section("overview", "Run bundle", body)


def _replay_controls(n: int) -> str:
    if n == 0:
        return section("replay-controls", "Replay controls", missing("event scrubber", "no events present"))
    body = (
        '<div class="controls"><button id="b-first" type="button">First</button>'
        '<button id="b-prev" type="button">Prev</button><button id="b-play" type="button">Play</button>'
        '<button id="b-next" type="button">Next</button><button id="b-last" type="button">Last</button>'
        f'<input id="scrub" type="range" min="0" max="{n - 1}" step="1" value="{n - 1}" '
        'aria-label="event sequence index"></div>'
        '<p id="scrub-label" class="mono"></p>'
        "<p>Scrubbing by sequence index dims every row and revision recorded after the cursor. The selected "
        "event is shown in the inspector.</p>"
    )
    return section("replay-controls", "Replay controls (event scrubber)", body)


def render_bundle(b: LoadedBundle) -> str:
    """Evaluation mode is exactly ``b.truth is not None`` (set by ``open_bundle(include_truth=True)``)."""
    evaluation = b.truth is not None
    beliefs = belief_views(b.repo, b.events)
    prov = provenance_view(b.repo, beliefs, b.events)
    cmd_rows = command_rows(b.events, b.repo)
    cats = categorize(b.mission)
    command_roots = [(label, rid) for label, rid in prov.roots if label.startswith("command ")]
    sections = [
        _overview(b),
        _replay_controls(len(b.events)),
        ro.health_section(b.events),
        rb.pose_section(pose_track(b.repo), b.truth, cats["navigation"], b.truth_present and not evaluation),
        rb.belief_status_section(beliefs, b.db_note),
        rb.uncertainty_section(beliefs, b.db_note),
        rb.provenance_section(prov, command_roots, b.db_note),
        ro.contradictions_section(beliefs, cats["contradictions"], cats["decisions"]),
        ro.mcbr_section(b.events, cats, b.mission is not None),
        ro.navigation_section(b.events, cats["navigation"]),
        ro.commands_section(cmd_rows, b.events),
        ro.baac_section(b.events, cats["baac"]),
        ro.faults_section(b.events, cats["faults"]),
        ro.metrics_section(b.metrics),
        ro.truth_section(b.truth, b.truth_present),
    ]
    if cats["other"]:
        sections.append(
            section("mission-other", "Other mission files", ro.mission_files(cats["other"], "other files"))
        )
    nav = "".join(f'<a href="#{sid}">{esc(label)}</a>' for sid, label in NAV)
    events_json = [
        [
            e.sequence,
            e.event_type.value,
            e.module,
            e.severity.value,
            e.availability.value,
            str(e.trace_id),
            e.envelope.measurement_time_ns,
            redact(e.payload),
        ]
        for e in b.events
    ]
    body = (
        f'{BANNER}{EVAL_BANNER if evaluation else ""}<div class="layout"><nav aria-label="sections">{nav}</nav><main>{"".join(sections)}</main>'
        '<aside aria-label="inspector"><h2>Inspector</h2><pre id="inspector-body">select an event with the '
        "replay scrubber</pre></aside></div>"
    )
    script = script_json("events-data", events_json) + script_json("prov-data", rb.prov_json(prov))
    title = f"{'EVALUATION MODE - ' if evaluation else ''}Conrad console {b.manifest.run_id}"
    return _doc(title, body, script + f"<script>{JS}</script>")


def render_run(
    run_dir: str | Path, object_store: str | Path | None = None, include_truth: bool = False
) -> str:
    """Default (ch37 UI-08): no truth-derived value in the page. ``include_truth`` = evaluation mode."""
    loaded = open_bundle(run_dir, object_store, include_truth)
    if isinstance(loaded, IntegrityFailure):
        return integrity_page(loaded)
    try:
        return render_bundle(loaded)
    finally:
        loaded.close()


def build_console(
    run_dir: str | Path,
    out_html: str | Path,
    object_store: str | Path | None = None,
    include_truth: bool = False,
) -> Path:
    """Render ``run_dir`` into one self-contained HTML file and return its path."""
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_run(run_dir, object_store, include_truth), encoding="utf-8")
    return out
