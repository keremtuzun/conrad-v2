"""Join existing I4 development traces into an action-level energy ledger.

This reads completed Python-kernel diagnostics only. It never launches missions or
uses held-out worlds. Truth-side fields in the input are copied for audit, not
passed to a planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from uuid import UUID

from conrad.orchestration.evaluation import evaluate_run_dir
from conrad.persistence.db import make_engine
from conrad.persistence.repository import Repository
from conrad.schemas.world import Domain


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_direct_revisions(run: Path) -> list[float]:
    target = UUID(evaluate_run_dir(run)["target_registry_id"])
    engine = make_engine(run / "conrad.sqlite")
    try:
        return sorted(
            revision.measurement_time_ns / 1e9
            for revision in Repository(engine).all_revisions()
            if revision.cell.domain is Domain.TECHNICAL
            and revision.cell.registry_entity_id == target
            and revision.update_kind.value == "DIRECT"
        )
    finally:
        engine.dispose()


def ledger(root: Path, arm: str = "V4_full") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seed in range(8100000, 8100040):
        run = root / f"I4-ENERGY-DIAG-s{seed}-{arm}"
        report_path = run / "reports" / "i4_energy_diagnostic.json"
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = _read(report_path)
        report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
        direct_times = _target_direct_revisions(run)
        views = _read(run / "mission" / "view_execution.json")
        plans = {
            entry["plan"]["plan_id"]: entry
            for entry in _read(run / "mission" / "mcbr_candidate_tables.json")
            if entry.get("plan", {}).get("plan_id")
        }
        reported_views = {view["plan_id"]: view for view in report["views"]}
        prior: list[dict[str, Any]] = []
        for view in views:
            plan_id = view["plan_id"]
            if plan_id not in reported_views or plan_id not in plans:
                raise ValueError(f"missing action join for seed {seed}, plan {plan_id}")
            diagnostic = reported_views[plan_id]
            plan = plans[plan_id]["plan"]
            action = plan["primary_action"]
            candidates = plans[plan_id]["candidate_table"]
            selected = next((c for c in candidates if c["action_id"] == view["action_id"]), None)
            if selected is None:
                raise ValueError(f"missing selected candidate for seed {seed}, plan {plan_id}")
            pose = view["commanded_position_m"]
            repeat = any(
                math.dist(pose, old["commanded_position_m"]) < 0.1
                and action["sensor_configuration"]["modality"] == old["modality"]
                for old in prior
            )
            out.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "source_report_sha256": report_sha256,
                    "plan_id": plan_id,
                    "action_id": view["action_id"],
                    "need_id": plan["need_id"],
                    "target_beliefs": plan["target_beliefs"],
                    "modality": action["sensor_configuration"]["modality"],
                    "commanded_position_m": pose,
                    "accepted_s": view["t_start_s"],
                    "outcome": view["outcome"],
                    "repeated_pose_within_0_1m": repeat,
                    "predicted_mission_gain": plan["expected_mission_gain"],
                    "predicted_information_gain": plan["expected_information_gain"],
                    "selected_score": selected.get("score"),
                    "selected_observational_gain": selected.get("gain", {}).get("observational"),
                    "selected_redundancy": selected.get("gain", {}).get("redundancy"),
                    "predicted_cost": diagnostic["predicted"],
                    "realized_energy_j": diagnostic["realized_energy_j"],
                    "realized_approach_energy_j": diagnostic["realized_approach_energy_j"],
                    "realized_elapsed_s": diagnostic["realized_elapsed_s"],
                    "first_target_revision_during_view": diagnostic["target_revision_during_view"],
                    "target_direct_revisions_during_view": sum(
                        view["t_start_s"] <= t <= view["t_end_s"] for t in direct_times
                    ),
                }
            )
            prior.append({**view, "modality": action["sensor_configuration"]["modality"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="existing I4-ENERGY-DIAG development bundle directory")
    parser.add_argument("--arm", default="V4_full")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = ledger(args.root, args.arm)
    summary = {
        "evidence_class": "DEVELOPMENT_DIAGNOSTIC; Python kernel, no gate promotion",
        "worlds": 40,
        "actions": len(rows),
        "repeated_pose_actions": sum(r["repeated_pose_within_0_1m"] for r in rows),
        "zero_predicted_mission_gain_actions": sum(r["predicted_mission_gain"] == 0 for r in rows),
        "repeated_pose_energy_j": sum(r["realized_energy_j"] for r in rows if r["repeated_pose_within_0_1m"]),
        "zero_predicted_mission_gain_energy_j": sum(
            r["realized_energy_j"] for r in rows if r["predicted_mission_gain"] == 0
        ),
        "actions_with_target_revision": sum(r["target_direct_revisions_during_view"] > 0 for r in rows),
        "repeated_pose_actions_with_target_revision": sum(
            r["repeated_pose_within_0_1m"] and r["target_direct_revisions_during_view"] > 0 for r in rows
        ),
    }
    body = {"summary": summary, "actions": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(body, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
