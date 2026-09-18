"""Small escaping HTML helpers. Every dynamic string goes through ``esc``."""

from __future__ import annotations

import json
from collections.abc import Sequence
from html import escape
from typing import Any

MAX_ROWS = 500


def esc(value: object) -> str:
    return escape("" if value is None else str(value), quote=True)


def secs(time_ns: int | None) -> str:
    return "n/a" if time_ns is None else f"{time_ns / 1e9:.3f} s"


def badge(value: str, family: str = "ks") -> str:
    return f'<span class="badge {family}-{esc(value)}">{esc(value)}</span>'


def missing(what: str, why: str = "not present in this bundle") -> str:
    return f'<p class="missing">{esc(what)}: {esc(why)}.</p>'


def section(sid: str, title: str, body: str, css: str = "") -> str:
    cls = f' class="{css}"' if css else ""
    return f'<section id="{esc(sid)}"{cls}><h2>{esc(title)}</h2>{body}</section>'


def table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], seqs: Sequence[int | None] | None = None
) -> str:
    """``rows`` hold already-escaped HTML cells. ``seqs`` tag rows for the replay scrubber."""
    if not rows:
        return '<p class="missing">no rows.</p>'
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows[:MAX_ROWS]):
        seq = seqs[i] if seqs is not None else None
        attr = f' data-seq="{seq}"' if seq is not None else ""
        body.append(f"<tr{attr}>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    more = (
        f'<p class="missing">{len(rows) - MAX_ROWS} further rows not shown.</p>'
        if len(rows) > MAX_ROWS
        else ""
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>{more}'


def compact(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def records_table(recs: Sequence[dict[str, Any]], max_cols: int = 10) -> str:
    """Generic table for JSON records of unknown shape (mission/ and truth/ files)."""
    if not recs:
        return '<p class="missing">no records.</p>'
    cols: list[str] = []
    for r in recs[:25]:
        for k in r:
            if k not in cols and len(cols) < max_cols:
                cols.append(k)
    rows = [[esc(compact(r.get(c, ""))) for c in cols] for r in recs]
    return table(cols, rows)


def script_json(element_id: str, data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    raw = raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f'<script type="application/json" id="{esc(element_id)}">{raw}</script>'
