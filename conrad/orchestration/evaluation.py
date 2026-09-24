"""evaluate_run: post-run metrics of a stored mission bundle against its truth record.

Used by ``conrad eval run --run <id>``. It reads files only: the run's SQLite (belief revisions), the
deployment artifacts under ``mission/`` and the evaluation-only ``truth/truth_record.json``. It imports no
truth-plane code and never feeds anything back into a runtime.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from conrad.domains.technical.config import PriorConfig
from conrad.persistence.db import make_engine
from conrad.persistence.repository import Repository
from conrad.schemas.belief import BeliefRevision
from conrad.schemas.world import Domain
from conrad.settings import load_settings

QUANTITIES = ("corrosion_depth_m", "crack_length_m")
TRUTH_RECORD = "truth/truth_record.json"


def _claims(rev: BeliefRevision) -> dict[str, Any]:
    cell = rev.cell
    out: dict[str, Any] = {
        "revision": rev.revision,
        "update_kind": rev.update_kind.value,
        "t_s": rev.measurement_time_ns / 1e9,
        "knowledge_status": cell.knowledge_status.value,
        "U_A": cell.uncertainty.aleatoric,
        "U_E": cell.uncertainty.epistemic,
        "U_C": cell.uncertainty.contradiction,
        "U_O": cell.uncertainty.observational,
        "consumed_evidence": len(rev.consumed_evidence_ids),
    }
    for name in (*QUANTITIES, "condition"):
        c = cell.claim(name)
        out[name] = None if c is None else c.value
        out[f"{name}.status"] = "UNKNOWN" if c is None else c.status.value
        if c is not None and name in QUANTITIES:
            out[f"{name}.U_O"] = c.uncertainty.observational
    return out


def _errors(state: dict[str, Any], truth: dict[str, Any], prior_mean: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for q in QUANTITIES:
        t = truth.get(q)
        v = state.get(q)
        out[f"{q}.abs_error"] = None if (v is None or t is None) else abs(float(v) - float(t))
        out[f"{q}.prior_mean_abs_error"] = (
            None if t is None else abs(float(prior_mean.get(q, 0.0)) - float(t))
        )
    return out


def evaluate_run_dir(run_dir: Path) -> dict[str, Any]:
    truth = json.loads((run_dir / TRUTH_RECORD).read_text(encoding="utf-8"))
    meta = truth["meta"]
    target = UUID(meta["target_registry_id"])
    engine = make_engine(run_dir / "conrad.sqlite")
    try:
        revs = [
            r
            for r in Repository(engine).all_revisions()
            if r.cell.domain is Domain.TECHNICAL and r.cell.registry_entity_id == target
        ]
    finally:
        engine.dispose()
    goals = json.loads((run_dir / "mission" / "trajectories.json").read_text(encoding="utf-8"))["goals"]
    inspect_t = next((g["t_s"] for g in goals if g["purpose"] == "INSPECT" and g["accepted"]), None)
    before_revs = [r for r in revs if inspect_t is None or r.measurement_time_ns / 1e9 <= inspect_t]
    before = _claims(before_revs[-1]) if before_revs else None
    after = _claims(revs[-1]) if revs else None
    direct_after = [
        r.revision
        for r in revs
        if r.update_kind.value == "DIRECT" and (inspect_t is None or r.measurement_time_ns / 1e9 > inspect_t)
    ]
    t_start, t_end = truth["target_states"]["start"], truth["target_states"]["end"]
    prior_mean = dict(PriorConfig().mean)
    patch = truth["series"].get("patch_visibility", [])
    labels = truth["series"].get("structural_label", [])
    spatial_labels = truth["series"].get("spatial_structural_label", [])
    target_spatial_ids = {
        row["observation_id"] for row in spatial_labels if row.get("world_id") == meta.get("target_world_id")
    }
    associations_path = run_dir / "mission" / "structural_associations.json"
    associations = (
        json.loads(associations_path.read_text(encoding="utf-8")) if associations_path.exists() else []
    )
    return {
        "scenario_id": meta["scenario_id"],
        "seed": meta["seed"],
        "target_registry_id": str(target),
        "truth_target_start": t_start,
        "truth_target_end": t_end,
        "inspection_goal_t_s": inspect_t,
        "target_before": before,
        "target_after": after,
        "target_error_before": None if before is None else _errors(before, t_end, prior_mean),
        "target_error_after": None if after is None else _errors(after, t_end, prior_mean),
        "target_direct_revisions_after_inspection": direct_after,
        "target_revision_count": len(revs),
        "patch_max_visible_fraction": max((p["visible_fraction"] for p in patch), default=0.0),
        "patch_first_visible_t_s": next((p["t_s"] for p in patch if p["visible_fraction"] > 0.3), None),
        "target_structural_observations": sum(1 for x in labels if x.get("patch")),
        "structural_observations": len(labels),
        "spatial_structural_observations": len(spatial_labels),
        "spatial_target_structural_observations": len(target_spatial_ids),
        "spatial_target_associated_evidence": sum(
            1
            for row in associations
            if row.get("observation_id") in target_spatial_ids and row.get("registry_id") == str(target)
        ),
        "vehicle": truth["vehicle"],
    }


def evaluate_run(run_id: str, config: str | None = None) -> dict[str, Any]:
    settings = load_settings(config or "configs/sim/default.yaml")
    run_dir = Path(run_id)
    if not run_dir.is_absolute() or not run_dir.exists():
        run_dir = settings.resolve(settings.paths.runs_dir) / run_id
    if not (run_dir / TRUTH_RECORD).exists():
        raise FileNotFoundError(f"run {run_id!r} has no truth record at {run_dir / TRUTH_RECORD}")
    report = evaluate_run_dir(run_dir)
    metrics = run_dir / "mission" / "runtime_metrics.json"
    if metrics.exists():
        report["runtime"] = json.loads(metrics.read_text(encoding="utf-8"))
    return report
