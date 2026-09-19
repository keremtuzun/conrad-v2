"""Deterministic replay of a stored mission run (CC-10, SS-02). TRUTH-SIDE harness.

1. ``verify_bundle`` checks every file and object digest; anything missing or corrupt fails closed before
   any re-execution (SS-04).
2. The run is re-executed from the stored seed and resolved configuration into a scratch location.
3. Event signatures (order, type, module, payload digest), decision sequences and belief revision headers
   must be identical. Any mismatch is reported with its first differing index.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from conrad.persistence.db import make_engine
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.replay_store import ReplayIntegrityError, verify_bundle
from conrad.persistence.repository import Repository
from conrad.runtime.event_log import event_signature, read_events
from conrad.settings import ConradSettings
from conrad.sim.mission.run import run_scenario


def _decisions(run_dir: Path) -> list[tuple[str, str | None, bool]]:
    out = []
    for line in (run_dir / "mission" / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        chosen = r.get("chosen") or {}
        out.append((r["decision_id"], chosen.get("action_type"), bool(r["abstained"])))
    return out


def _revisions(run_dir: Path) -> list[tuple[str, int, str, int]]:
    engine = make_engine(run_dir / "conrad.sqlite")
    try:
        revs = Repository(engine).all_revisions()
    finally:
        engine.dispose()
    return [(str(r.belief_id), r.revision, r.update_kind.value, r.measurement_time_ns) for r in revs]


def _first_diff(a: list[Any], b: list[Any]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def fingerprint(run_dir: Path) -> dict[str, Any]:
    return {
        "events": event_signature(list(read_events(run_dir / "events.jsonl"))),
        "decisions": _decisions(run_dir),
        "revisions": _revisions(run_dir),
    }


def compare(original: Path, replayed: Path) -> dict[str, Any]:
    a, b = fingerprint(original), fingerprint(replayed)
    report: dict[str, Any] = {}
    for key in ("events", "decisions", "revisions"):
        idx = _first_diff(a[key], b[key])
        report[key] = {
            "original": len(a[key]),
            "replayed": len(b[key]),
            "equal": idx is None,
            "first_difference": idx,
        }
    report["equal"] = all(v["equal"] for v in report.values() if isinstance(v, dict))
    return report


def replay_run(run_dir: Path, scratch: Path | None = None) -> dict[str, Any]:
    """Raises ReplayIntegrityError when the bundle does not verify. Returns the comparison report."""
    manifest = verify_bundle(run_dir, ObjectStore(run_dir / "objects"))
    inputs = manifest.replay_inputs
    for key in ("scenario_id", "config_resolved", "run_name"):
        if key not in inputs:
            raise ReplayIntegrityError([f"replay input missing: {key}"])
    settings = ConradSettings.model_validate(inputs["config_resolved"])
    if settings.config_digest() != inputs["config_digest"]:
        raise ReplayIntegrityError(["stored configuration does not match its recorded digest"])
    root = scratch or Path(tempfile.mkdtemp(prefix="conrad-replay-"))
    out = run_scenario(str(inputs["scenario_id"]), settings, run_id=str(inputs["run_name"]), runs_root=root)
    replayed = Path(out["run_dir"])
    report = compare(run_dir, replayed)
    report["replay_dir"] = str(replayed)
    report["verified_files"] = len(manifest.files)
    report["verified_objects"] = len(manifest.object_digests)
    return report
