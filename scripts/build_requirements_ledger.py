"""Generate the requirements ledger (markdown + JSON) from docs/specification/requirements_source.psv.

Status is DERIVED from the repository, never typed by hand, unless the source row carries an explicit
override (INTEGRATED / EVALUATED / VALIDATED / BLOCKED_EXTERNAL / BLOCKED_OPEN_DECISION):

  MISSING      no implementation path exists
  PARTIAL      some implementation paths exist
  IMPLEMENTED  all implementation paths exist, no test path exists
  TESTED       all implementation paths and at least one test path exist
  EVALUATED    TESTED, and every linked experiment ID has a retained result artifact in artifacts/experiments
  INTEGRATED   TESTED, and the linked formal gate (12th column ``gate:<ID>``) has official status PASS
Hand-written INTEGRATED/EVALUATED/VALIDATED overrides are ignored: those states are derived from evidence only.
BLOCKED_* overrides are kept (they record an external dependency, not a claim of success).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "specification" / "requirements_source.psv"
OUT_MD = ROOT / "docs" / "specification" / "CONRAD_V2_REQUIREMENTS_LEDGER.md"
OUT_JSON = ROOT / "artifacts" / "spec" / "requirements-ledger.json"
BLOCKED = {"BLOCKED_EXTERNAL", "BLOCKED_OPEN_DECISION"}
PROMOTED = {"INTEGRATED", "EVALUATED", "VALIDATED"}
ORDER = [
    "MISSING",
    "PARTIAL",
    "IMPLEMENTED",
    "TESTED",
    "INTEGRATED",
    "EVALUATED",
    "VALIDATED",
    "BLOCKED_EXTERNAL",
    "BLOCKED_OPEN_DECISION",
]


def _split(cell: str) -> list[str]:
    return [p.strip() for p in cell.split(";") if p.strip()]


def _exists(rel: str) -> bool:
    p = ROOT / rel
    if p.is_dir():
        return any(p.rglob("*.py")) or any(p.rglob("*.cs")) or any(p.iterdir())
    return p.exists()


EXPERIMENTS_DIR = ROOT / "artifacts" / "experiments"


def _experiment_has_artifact(exp_id: str) -> bool:
    if not EXPERIMENTS_DIR.exists():
        return False
    for path in EXPERIMENTS_DIR.rglob("*.json"):
        if path.stem.startswith(exp_id) or any(part.startswith(exp_id) for part in path.relative_to(EXPERIMENTS_DIR).parts[:-1]):
            return True
    return False


def _gate_status() -> dict[str, str]:
    sys.path.insert(0, str(ROOT))
    from conrad.evaluation.gates import evaluate_gates

    return {k: v.official_status.value for k, v in evaluate_gates().items()}


def parse() -> list[dict[str, object]]:
    gates = _gate_status()
    rows = []
    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cells = [c.strip() for c in raw.split("|")]
        cells += [""] * (12 - len(cells))
        rid, section, authority, owner, summary, impl, tests, exps, keys, ext, override, gate_cell = cells[:12]
        gate_id = gate_cell.removeprefix("gate:").strip() or None
        impl_paths, test_paths = _split(impl), _split(tests)
        present = [p for p in impl_paths if _exists(p)]
        tests_present = [p for p in test_paths if _exists(p)]
        if not present:
            derived = "MISSING"
        elif len(present) < len(impl_paths):
            derived = "PARTIAL"
        elif not tests_present:
            derived = "IMPLEMENTED"
        else:
            derived = "TESTED"
        experiments = _split(exps)
        missing_experiments = [e for e in experiments if not _experiment_has_artifact(e)]
        gate_status = gates.get(gate_id) if gate_id else None
        if override in BLOCKED:
            status = override
        elif derived == "TESTED" and gate_status == "PASS":
            status = "INTEGRATED"
        elif derived == "TESTED" and experiments and not missing_experiments:
            status = "EVALUATED"
        else:
            status = derived
        rows.append(
            {
                "requirement_id": rid,
                "source_section": section,
                "requirement_summary": summary,
                "authority": authority,
                "owner": owner,
                "implementation_status": status,
                "implementation_paths": impl_paths,
                "missing_paths": [p for p in impl_paths if p not in present],
                "test_ids": test_paths,
                "missing_tests": [p for p in test_paths if p not in tests_present],
                "experiment_ids": experiments,
                "missing_experiment_artifacts": missing_experiments,
                "gate_id": gate_id,
                "gate_official_status": gate_status,
                "configuration_keys": _split(keys),
                "assumptions": [],
                "external_dependencies": [ext] if ext else [],
                "evidence_artifacts": [],
                "maturity_level": None,
                "claim_status": "NONE" if status in ("MISSING", "PARTIAL") else "IMPLEMENTED",
            }
        )
    return rows


def main() -> int:
    rows = parse()
    ids = [r["requirement_id"] for r in rows]
    if len(ids) != len(set(ids)):
        print("duplicate requirement ids", file=sys.stderr)
        return 1
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({"source": str(SOURCE.relative_to(ROOT)), "requirements": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    counts = {s: sum(1 for r in rows if r["implementation_status"] == s) for s in ORDER}
    lines = [
        "# Conrad V2 requirements ledger",
        "",
        "Generated by `scripts/build_requirements_ledger.py` from `docs/specification/requirements_source.psv`.",
        "Statuses are derived from the files present in the repository; do not edit this file by hand.",
        "",
        "| status | count |",
        "|---|---|",
        *[f"| {s} | {c} |" for s, c in counts.items() if c],
        "",
        "| ID | Source | Authority | Status | Gate (official) | Summary | Missing |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        missing = ", ".join(r["missing_paths"] + r["missing_tests"]) or ""  # type: ignore[operator]
        lines.append(
            f"| {r['requirement_id']} | {r['source_section']} | {r['authority']} | {r['implementation_status']} | "
            f"{(str(r['gate_id']) + ': ' + str(r['gate_official_status'])) if r['gate_id'] else ''} | {r['requirement_summary']} | {missing} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
