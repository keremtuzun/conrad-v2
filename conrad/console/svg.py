"""Inline SVG charts generated in Python (no JS chart library, no network).

Every series carries a text label and a dash pattern so that colour is never the only signal (ch37 Visual
and interaction requirements).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

W, H, PAD = 640, 220, 36


@dataclass(frozen=True)
class Series:
    label: str
    points: tuple[tuple[float, float], ...]
    css_class: str
    dash: str = ""
    markers: bool = True


def _bounds(series: Sequence[Series]) -> tuple[float, float, float, float]:
    xs = [p[0] for s in series for p in s.points]
    ys = [p[1] for s in series for p in s.points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 - x0 < 1e-12:
        x0, x1 = x0 - 1.0, x1 + 1.0
    if y1 - y0 < 1e-12:
        y0, y1 = y0 - 1.0, y1 + 1.0
    return x0, x1, y0, y1


def _fmt(v: float) -> str:
    return f"{v:.3g}"


def _frame(title: str, body: list[str], x_label: str, y_label: str, bounds: tuple[float, ...]) -> str:
    x0, x1, y0, y1 = bounds
    axes = [
        f'<line class="axis" x1="{PAD}" y1="{H - PAD}" x2="{W - 8}" y2="{H - PAD}"/>',
        f'<line class="axis" x1="{PAD}" y1="8" x2="{PAD}" y2="{H - PAD}"/>',
        f'<text class="tick" x="{PAD}" y="{H - PAD + 14}">{_fmt(x0)}</text>',
        f'<text class="tick" x="{W - 8}" y="{H - PAD + 14}" text-anchor="end">{_fmt(x1)}</text>',
        f'<text class="tick" x="{PAD - 4}" y="{H - PAD}" text-anchor="end">{_fmt(y0)}</text>',
        f'<text class="tick" x="{PAD - 4}" y="16" text-anchor="end">{_fmt(y1)}</text>',
        f'<text class="axlabel" x="{(W + PAD) / 2}" y="{H - 6}" text-anchor="middle">{escape(x_label)}</text>',
        f'<text class="axlabel" x="10" y="{H / 2}" transform="rotate(-90 10 {H / 2})" '
        f'text-anchor="middle">{escape(y_label)}</text>',
    ]
    return (
        f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title>{''.join(axes)}{''.join(body)}</svg>"
    )


def line_chart(
    title: str, series: Sequence[Series], x_label: str, y_label: str, equal_aspect: bool = False
) -> str:
    """Polyline chart. With ``equal_aspect`` both axes share one scale (plan-view tracks)."""
    live = [s for s in series if s.points]
    if not live:
        return f'<p class="missing">{escape(title)}: no data points present.</p>'
    x0, x1, y0, y1 = _bounds(live)
    sx, sy = (W - PAD - 8) / (x1 - x0), (H - PAD - 8) / (y1 - y0)
    if equal_aspect:
        s = min(sx, sy)
        sx = sy = s
    body: list[str] = []
    for s_ in live:
        pts = [(PAD + (x - x0) * sx, H - PAD - (y - y0) * sy) for x, y in s_.points]
        path = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        dash = f' stroke-dasharray="{s_.dash}"' if s_.dash else ""
        body.append(
            f'<polyline class="series {s_.css_class}" points="{path}"{dash}>'
            f"<title>{escape(s_.label)}</title></polyline>"
        )
        if s_.markers:
            body += [
                f'<circle class="dot {s_.css_class}" cx="{px:.1f}" cy="{py:.1f}" r="2.5"/>' for px, py in pts
            ]
    items = []
    for s_ in live:
        dash = f' stroke-dasharray="{s_.dash}"' if s_.dash else ""
        items.append(
            f'<span class="legend-item"><svg width="28" height="10"><line class="series {s_.css_class}" '
            f'x1="0" y1="5" x2="28" y2="5"{dash}/></svg>{escape(s_.label)}</span>'
        )
    legend = "".join(items)
    return _frame(title, body, x_label, y_label, (x0, x1, y0, y1)) + f'<div class="legend">{legend}</div>'


def event_lanes(title: str, lanes: dict[str, list[int]], n_events: int) -> str:
    """One row per event type, one tick per event at its sequence index."""
    if not lanes or n_events == 0:
        return f'<p class="missing">{escape(title)}: no events present.</p>'
    row_h, left = 16, 190
    height = row_h * len(lanes) + 24
    span = max(n_events - 1, 1)
    body: list[str] = []
    for i, (name, seqs) in enumerate(sorted(lanes.items())):
        y = 12 + i * row_h
        body.append(f'<text class="tick" x="{left - 6}" y="{y + 4}" text-anchor="end">{escape(name)}</text>')
        body.append(f'<line class="lane" x1="{left}" y1="{y}" x2="{W - 8}" y2="{y}"/>')
        for seq in seqs:
            x = left + (W - 8 - left) * seq / span
            body.append(
                f'<line class="evtick ev-{escape(name)}" x1="{x:.1f}" y1="{y - 5}" x2="{x:.1f}" y2="{y + 5}">'
                f"<title>#{seq} {escape(name)}</title></line>"
            )
    body.append(f'<text class="tick" x="{left}" y="{height - 4}">seq 0</text>')
    body.append(
        f'<text class="tick" x="{W - 8}" y="{height - 4}" text-anchor="end">seq {n_events - 1}</text>'
    )
    return (
        f'<svg class="chart" viewBox="0 0 {W} {height}" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title>{''.join(body)}</svg>"
    )
