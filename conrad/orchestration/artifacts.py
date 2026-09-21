"""Deployment-side mission artifacts written under ``mission/`` in the run directory. DEPLOYMENT PLANE.

MCBR candidate tables (with rejected candidates and reasons), decision records, goals and trajectories,
BAAC transmissions, the shore receiver state, structural associations, safety events and runtime metrics.
Nothing here is truth.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from conrad.decision.uir import uir_report
from conrad.orchestration.view_execution import summarize as summarize_views

if TYPE_CHECKING:
    from conrad.orchestration.mission import MissionRuntime


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")


def _jsonl(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r + "\n" for r in rows), encoding="utf-8")


def runtime_metrics(rt: MissionRuntime) -> dict[str, Any]:
    records = [d.record for d in rt.deliberation.decisions]
    uir = uir_report(records)
    x, p = rt.executive.stats, rt.perception.stats
    return {
        "decisions": len(records),
        "decision_actions": [None if r.chosen is None else r.chosen.action_type.value for r in records],
        "abstentions": sum(1 for r in records if r.abstained),
        "uir": uir.unsupported_inference_rate,
        "uir_report": uir.model_dump(mode="json"),
        "plans": [
            {
                "plan_id": str(a.plan.plan_id),
                "status": a.plan.status.value,
                "planner": rt.deliberation.planner.name,
                "rejected": len(a.plan.rejected),
                "feasible": sum(1 for r in a.table if r.get("feasible")),
                "candidates": len(a.table),
                "latency_ms": a.latency_ms,
            }
            for a in rt.deliberation.plans
        ],
        "view_execution": summarize_views(rt.executive.views.records),
        "goal_rejections": rt.executive.goal_rejections,
        "empty_plans": rt.routing.empty_plans,
        "view_outcomes": rt.routing.view_outcomes,
        "deferred_replans": rt.routing.deferred_replans,
        "replans": rt.routing.replans,
        "commands_accepted": rt.gateway.accepted,
        "commands_rejected": rt.gateway.rejected,
        "commands_refused_by_safety_supervisor": x.refused_by_supervisor,
        "gateway_rejection_reasons": x.rejection_reasons,
        "safety_events": x.safety_events,
        "runtime_state_history": [(s.value, r) for s, r in rt.supervisor.history],
        "module_health": {k: v.value for k, v in rt.health.snapshot().items()},
        "failed_modules": sorted(rt.runner.failed),
        "perception": p.__dict__,
        "beliefs_published": rt.published,
        "communication": rt.shore.metrics(),
    }


def write_mission_artifacts(rt: MissionRuntime, run_dir: Path) -> dict[str, Any]:
    out = run_dir / "mission"
    tables = []
    for a in rt.deliberation.plans:
        tables.append(
            {
                "plan": a.plan.model_dump(mode="json"),
                "decision_id": str(a.decision_id),
                "adoption_provenance_id": str(a.adoption_record),
                "candidate_table": a.table,
            }
        )
    _dump(out / "mcbr_candidate_tables.json", tables)
    _jsonl(out / "decisions.jsonl", [d.record.canonical_json() for d in rt.deliberation.decisions])
    _dump(
        out / "trajectories.json", {"estimated_track": rt.estimated_track, "goals": rt.executive.stats.goals}
    )
    _dump(out / "view_execution.json", rt.executive.views.to_json())
    _jsonl(out / "baac_transmissions.jsonl", [t.canonical_json() for t in rt.shore.sender.transmissions])
    _dump(out / "receiver_state.json", rt.shore.receiver_state())
    _dump(out / "structural_associations.json", rt.perception.structural_log)
    _dump(out / "requirements.json", [r.model_dump(mode="json") for r in rt.deliberation.requirements])
    metrics = runtime_metrics(rt)
    _dump(out / "runtime_metrics.json", metrics)
    return metrics
