"""Deterministic replay of a stored mission run (CC-10, SS-02). TRUTH-SIDE harness.

1. ``verify_bundle`` checks every file and object digest; anything missing or corrupt fails closed before
   any re-execution (SS-04).
2. The run is re-executed from the stored seed and resolved configuration into a scratch location. The mission
   world options and runtime configuration are taken from the bundle (``mission_world_options`` /
   ``mission_runtime_config``), not re-resolved from the current scenario table: a changed code default must show
   up as a replay mismatch of the recorded run, not silently re-define it.
3. Event signatures, decisions, belief revisions, persisted Observation/Evidence/command payloads,
   trajectories, view execution, associations and transmissions must be identical. A mismatch reports
   its first differing index.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from conrad.domains.technical.spatial_mission import MODEL_VERSION as SPATIAL_MODEL_VERSION
from conrad.orchestration.mission_config import runtime_config
from conrad.persistence.db import make_engine
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.replay_store import ReplayIntegrityError, verify_bundle
from conrad.persistence.repository import Repository
from conrad.runtime.event_log import event_signature, read_events
from conrad.schemas.structural_sensor import StructuralSensorModelV2
from conrad.settings import ConradSettings
from conrad.sim.mission.options import world_options
from conrad.sim.mission.run import run_scenario, spatial_version_contract, validate_spatial_mission_selection


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


def _payloads(run_dir: Path, table: str, key: str) -> list[tuple[str, str]]:
    """Compare persisted content, including structural supports, not only row counts or headers."""
    if (table, key) not in {
        ("observations", "observation_id"),
        ("evidence", "evidence_id"),
        ("commands", "command_id"),
        ("belief_revisions", "id"),
    }:
        raise ValueError("unsupported replay payload table")
    with sqlite3.connect(run_dir / "conrad.sqlite") as con:
        return [
            (str(row_id), hashlib.sha256(payload.encode("utf-8")).hexdigest())
            for row_id, payload in con.execute(f"SELECT {key}, payload_json FROM {table} ORDER BY {key}")
        ]


def _json_rows(run_dir: Path, relative: str) -> list[str]:
    path = run_dir / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value if isinstance(value, list) else [value]
    return [
        hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        for row in rows
    ]


def _jsonl_rows(run_dir: Path, relative: str) -> list[str]:
    return [
        hashlib.sha256(line.encode("utf-8")).hexdigest()
        for line in (run_dir / relative).read_text(encoding="utf-8").splitlines()
    ]


def fingerprint(run_dir: Path) -> dict[str, Any]:
    return {
        "events": event_signature(list(read_events(run_dir / "events.jsonl"))),
        "decisions": _decisions(run_dir),
        "revisions": _revisions(run_dir),
        "observation_payloads": _payloads(run_dir, "observations", "observation_id"),
        "evidence_payloads": _payloads(run_dir, "evidence", "evidence_id"),
        "command_payloads": _payloads(run_dir, "commands", "command_id"),
        "belief_payloads": _payloads(run_dir, "belief_revisions", "id"),
        "trajectories": _json_rows(run_dir, "mission/trajectories.json"),
        "view_execution": _json_rows(run_dir, "mission/view_execution.json"),
        "structural_associations": _json_rows(run_dir, "mission/structural_associations.json"),
        "baac_transmissions": _jsonl_rows(run_dir, "mission/baac_transmissions.jsonl"),
    }


def compare(original: Path, replayed: Path) -> dict[str, Any]:
    a, b = fingerprint(original), fingerprint(replayed)
    report: dict[str, Any] = {}
    for key in a:
        idx = _first_diff(a[key], b[key])
        report[key] = {
            "original": len(a[key]),
            "replayed": len(b[key]),
            "equal": idx is None,
            "first_difference": idx,
        }
    report["equal"] = all(v["equal"] for v in report.values() if isinstance(v, dict))
    return report


def validate_spatial_replay_contract(inputs: dict[str, Any]) -> None:
    """Reject edited or unsupported spatial declarations before launching replay."""
    raw_world = inputs.get("mission_world_options")
    raw_runtime = inputs.get("mission_runtime_config")
    if not isinstance(raw_world, dict) or not isinstance(raw_runtime, dict):
        if inputs.get("spatial_versions") is None:
            return  # historical legacy bundles may predate stored resolved mission options
        raise ReplayIntegrityError(["spatial replay requires stored mission options"])
    try:
        wopts, rcfg = world_options(raw_world), runtime_config(raw_runtime)
        validate_spatial_mission_selection(wopts, rcfg)
    except (ValueError, KeyError, TypeError) as exc:
        raise ReplayIntegrityError([f"spatial mission selection mismatch: {exc}"]) from exc
    if rcfg.model2t_backend != "spatial_v1":
        if inputs.get("spatial_versions") is not None:
            raise ReplayIntegrityError(["legacy replay has unexpected spatial versions"])
        return
    if rcfg.model2t_spatial is None:
        raise ReplayIntegrityError(["spatial Model2T settings missing"])
    sensor = StructuralSensorModelV2.model_validate(rcfg.model2t_spatial["sensor"])
    expected = spatial_version_contract(sensor)
    if (
        inputs.get("spatial_versions") != expected
        or inputs.get("model_versions", {}).get("model2t") != SPATIAL_MODEL_VERSION
    ):
        raise ReplayIntegrityError(
            ["spatial architecture, sensor configuration, or Model2T version mismatch"]
        )


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
    validate_spatial_replay_contract(inputs)
    try:
        wopts = world_options(inputs["mission_world_options"]) if "mission_world_options" in inputs else None
        rcfg = (
            runtime_config(inputs["mission_runtime_config"]) if "mission_runtime_config" in inputs else None
        )
    except ValueError as exc:
        raise ReplayIntegrityError([f"stored mission options no longer validate: {exc}"]) from exc
    root = scratch or Path(tempfile.mkdtemp(prefix="conrad-replay-"))
    out = run_scenario(
        str(inputs["scenario_id"]),
        settings,
        run_id=str(inputs["run_name"]),
        runs_root=root,
        stored_world=wopts,
        stored_runtime=rcfg,
    )
    replayed = Path(out["run_dir"])
    report = compare(run_dir, replayed)
    report["replay_dir"] = str(replayed)
    report["verified_files"] = len(manifest.files)
    report["verified_objects"] = len(manifest.object_digests)
    return report
