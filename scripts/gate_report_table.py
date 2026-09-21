"""Emit the final gate table from the recorded evidence, never from prose.

For every gate in dependency order it reports the official and formal status, the criteria and how many of
them pass, the evidence path, the execution level (FORMAL = the gate's own path, SURROGATE = python kernel),
the held-out split the evidence names, the commit the evidence was recorded at, and the blocker when the gate
is not PASS. Everything is read from ``artifacts/gates/<gate>/evidence_*.json`` and ``conrad.evaluation.gates``.

Usage: python -m uv run python scripts/gate_report_table.py [--markdown out.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conrad.evaluation.gates import GATES, CriterionStatus, GateStatus, evaluate_gates  # noqa: E402

SPLIT = re.compile(r'"partition[^"]*":\s*"([^"]+)"|partition=([\w/.-]+)|seeds?\s(\d{6,}-\d{6,})')


def _evidence(gate: str, kind: str) -> dict | None:
    p = ROOT / "artifacts" / "gates" / gate / f"evidence_{kind}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _split_of(ev: dict | None) -> str:
    """The held-out split the evidence itself names, from its notes or its recorded meta."""
    if not ev:
        return ""
    hit = SPLIT.search(ev.get("notes", "") or "")
    if hit:
        return next(g for g in hit.groups() if g)
    for a in ev.get("artifacts", ()):
        if "partitions" in a:
            return a
    return ""


def rows() -> list[dict[str, str]]:
    results = evaluate_gates(ROOT / "artifacts" / "gates")
    out = []
    for g in GATES:
        r = results[g.gate_id]
        formal, surrogate = _evidence(g.gate_id, "formal"), _evidence(g.gate_id, "surrogate")
        ev = formal or surrogate
        passed = failed = 0
        if formal:
            for c in formal["criteria"]:
                passed += c["status"] == CriterionStatus.PASS.value
                failed += c["status"] == CriterionStatus.FAIL.value
        blocker = ""
        if r.official_status is not GateStatus.PASS:
            if r.blocking_upstream:
                blocker = "blocked by " + ", ".join(r.blocking_upstream)
            elif g.external_blocker:
                blocker = g.external_blocker.value
            elif formal:
                bad = [
                    c["criterion"] for c in formal["criteria"] if c["status"] == CriterionStatus.FAIL.value
                ]
                blocker = "; ".join(bad) or "no formal evidence"
            else:
                blocker = "no formal evidence"
        out.append(
            {
                "gate": g.gate_id,
                "official": r.official_status.value,
                "formal": r.formal_status.value,
                "surrogate": r.surrogate_status.value,
                "criteria": f"{passed}/{len(g.criteria)} pass" + (f", {failed} fail" if failed else ""),
                "source": g.source,
                "path": g.formal_path,
                "level": (ev or {}).get("evidence_class", "NONE"),
                "split": _split_of(ev),
                "commit": ((ev or {}).get("git_commit") or "")[:8],
                "evidence": f"artifacts/gates/{g.gate_id}/" if ev else "none",
                "blocker": blocker,
            }
        )
    return out


def markdown(data: list[dict[str, str]]) -> str:
    head = ("gate", "official", "formal", "criteria", "level", "split", "commit", "evidence", "blocker")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in data:
        lines.append("| " + " | ".join((r[h] or "").replace("|", "/") for h in head) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", type=Path, help="write the table to this file as markdown")
    args = ap.parse_args()
    data = rows()
    text = markdown(data)
    if args.markdown:
        args.markdown.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
