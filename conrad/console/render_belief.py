"""Belief-plane sections: status timeline, uncertainty channels, provenance explorer, pose track."""

from __future__ import annotations

from typing import Any

from conrad.console import svg
from conrad.console.html import badge, esc, missing, secs, section, table
from conrad.console.views_belief import CHANNELS, BeliefView, PosePoint, ProvenanceView, chain_lines
from conrad.console.views_ops import positions

CHANNEL_CSS = ("c-ua", "c-ue", "c-uc", "c-uo")
CHANNEL_DASH = ("", "6 3", "2 3", "10 3 2 3")


def belief_status_section(beliefs: list[BeliefView], db_note: str) -> str:
    if not beliefs:
        return section(
            "belief-status-timeline", "2S/2T/2E belief status timeline", missing("beliefs", db_note)
        )
    parts = [
        "<p>Knowledge status per claim over revisions. OBSERVED, INFERRED, PREDICTED (dashed) and UNKNOWN are "
        "shown as text as well as colour. UNKNOWN is an epistemic status, not a value.</p>"
    ]
    for b in beliefs:
        head = [f"rev {r.revision}" for r in b.revisions]
        rows = [["<b>belief summary</b>", *(badge(r.knowledge_status) for r in b.revisions)]]
        rows.append(["lifecycle", *(esc(r.lifecycle) for r in b.revisions)])
        rows.append(["update kind", *(esc(r.update_kind + (" LATE" if r.late else "")) for r in b.revisions)])
        rows.append(["measurement time", *(esc(secs(r.time_ns)) for r in b.revisions)])
        for name in b.claim_names:
            cells = []
            for r in b.revisions:
                c = r.claims.get(name)
                cells.append(
                    "absent" if c is None else f"{badge(c.status)} <span class='mono'>{esc(c.value)}</span>"
                )
            rows.append([f"claim <b>{esc(name)}</b>", *cells])
        reg = f", registry {esc(b.registry_entity_id)}" if b.registry_entity_id else ""
        parts.append(
            f"<h3>{esc(b.domain)} · {esc(b.entity_type)} · <code>{esc(b.belief_id)}</code>{reg}</h3>"
            + _pivot(["field", *head], rows, [None, *(r.seq for r in b.revisions)])
        )
    return section("belief-status-timeline", "2S/2T/2E belief status timeline", "".join(parts))


def _pivot(headers: list[str], rows: list[list[str]], col_seqs: list[int | None]) -> str:
    """Revision-as-column table; each column carries its commit sequence for the scrubber."""
    head = "".join(
        f'<th data-seq="{s}">{esc(h)}</th>' if s is not None else f"<th>{esc(h)}</th>"
        for h, s in zip(headers, col_seqs, strict=True)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td data-seq="{s}">{c}</td>' if s is not None else f"<td>{c}</td>"
            for c, s in zip(row, col_seqs, strict=True)
        )
        + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def uncertainty_section(beliefs: list[BeliefView], db_note: str) -> str:
    title = "Uncertainty decomposition U_A / U_E / U_C / U_O"
    if not beliefs:
        return section("uncertainty-channels", title, missing("beliefs", db_note))
    parts = [
        "<p>The four channels are separate quantities and do not sum to one. No single confidence scalar is "
        "shown because none exists in the contract.</p>"
    ]
    for b in beliefs:
        xs = [r.time_ns / 1e9 for r in b.revisions]
        series = [
            svg.Series(
                f"{code} {name}",
                tuple((x, r.uncertainty[i]) for x, r in zip(xs, b.revisions, strict=True)),
                CHANNEL_CSS[i],
                CHANNEL_DASH[i],
            )
            for i, (code, name) in enumerate(CHANNELS)
        ]
        rows = [
            [esc(r.revision), esc(secs(r.time_ns)), *(esc(f"{u:.4g}") for u in r.uncertainty)]
            for r in b.revisions
        ]
        parts.append(
            f"<h3>{esc(b.domain)} <code>{esc(b.belief_id)}</code></h3>"
            + svg.line_chart(f"uncertainty channels {b.belief_id}", series, "measurement time (s)", "value")
            + table(["rev", "time", *(f"{c} {n}" for c, n in CHANNELS)], rows, [r.seq for r in b.revisions])
        )
    return section("uncertainty-channels", title, "".join(parts))


def _chain_html(view: ProvenanceView, root: str) -> str:
    lines = chain_lines(view, root)
    return "".join(
        f'<div class="{"obs" if text.startswith("RAW OBSERVATION") else ""}">'
        f"{esc('  ' * depth + ('└ ' if depth else '') + text)}</div>"
        for depth, text in lines
    )


def provenance_section(view: ProvenanceView, command_roots: list[tuple[str, str]], db_note: str) -> str:
    title = "Provenance explorer"
    if not view.nodes:
        return section("provenance-explorer", title, missing("provenance records", db_note))
    opts = "".join(f'<option value="{esc(rid)}">{esc(label)}</option>' for label, rid in view.roots)
    parts = [
        "<p>Select a claim, belief revision, decision, plan or command to walk its provenance DAG down to raw "
        "observations. The chain comes from <code>Repository.provenance_closure</code>.</p>",
        f'<select id="prov-root"><option value="">choose a root...</option>{opts}</select>',
        '<div id="prov-tree" class="tree"></div>',
    ]
    if command_roots:
        label, rid = command_roots[-1]
        parts.append(
            f"<h3>Last command chain: {esc(label)}</h3>"
            f'<div id="provenance-last-command" class="tree">{_chain_html(view, rid)}</div>'
        )
    else:
        parts.append(missing("command provenance", "no command in this run carries a provenance root"))
    if view.errors:
        parts.append("<h3>Provenance problems</h3><ul>" + "".join(f"<li>{esc(e)}</li>" for e in view.errors))
        parts.append("</ul>")
    return section("provenance-explorer", title, "".join(parts))


def prov_json(view: ProvenanceView) -> dict[str, Any]:
    return {
        "nodes": {
            k: {"t": n.source_type, "op": n.operation, "m": n.module, "s": n.source_ids, "p": n.parents}
            for k, n in view.nodes.items()
        },
        "obs": view.observations,
    }


def pose_section(
    est: list[PosePoint],
    truth: dict[str, Any] | None,
    trajectories: dict[str, Any],
    truth_hidden: bool = False,
) -> str:
    """``truth`` is passed only in evaluation mode; ``truth_hidden`` notes a record withheld by UI-08."""
    series = [
        svg.Series(
            "estimated pose (belief side)", tuple((p.position[0], p.position[1]) for p in est), "c-est"
        )
    ]
    for name, data in trajectories.items():
        pts = positions(data)
        if pts:
            series.append(
                svg.Series(f"planned: {name}", tuple((p[0], p[1]) for p in pts), "c-traj", "4 3", False)
            )
    truth_pts = [p for data in (truth or {}).values() for p in positions(data)]
    if truth_pts:
        series.append(
            svg.Series(
                "TRUTH (evaluation only)", tuple((p[0], p[1]) for p in truth_pts), "c-truth", "3 3", False
            )
        )
    frames = sorted({p.frame for p in est})
    notes = [
        f"<p>Plan view x/y in frame {esc(', '.join(frames) or 'n/a')}. Estimated poses are the ones stamped on "
        "stored observations.</p>"
    ]
    notes.append(
        '<p class="truth-label">The TRUTH (evaluation only) series comes from truth/ and was never visible to '
        "the robot.</p>"
        if truth_pts
        else missing("truth pose", "hidden outside evaluation mode (ch37 UI-08)")
        if truth_hidden
        else missing("truth pose", "no truth/ pose record present")
    )
    rows = [[esc(secs(p.time_ns)), esc(p.frame), *(esc(f"{v:.3f}") for v in p.position)] for p in est]
    body = "".join(notes) + svg.line_chart("pose track", series, "x (m)", "y (m)", equal_aspect=True)
    body += (
        table(["time", "frame", "x", "y", "z"], rows)
        if rows
        else missing("estimated pose", "no observations")
    )
    return section("pose-track", "Pose track: estimated" + (" vs truth" if truth_pts else ""), body)
