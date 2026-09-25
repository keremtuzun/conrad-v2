"""Summarize I4/I7 impact observables from two completed development bundles."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from itertools import pairwise
from pathlib import Path
from typing import Any


def summarize(path: Path) -> dict[str, Any]:
    report = json.loads((path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "bundle_manifest.json").read_text(encoding="utf-8"))
    replay = manifest["replay_inputs"]
    runtime = report["runtime"]
    communication = runtime["communication"]
    trajectory = json.loads((path / "mission" / "trajectories.json").read_text(encoding="utf-8"))
    track = trajectory["estimated_track"]
    travel_m = sum(math.dist(a[1:4], b[1:4]) for a, b in pairwise(track))
    with sqlite3.connect(path / "conrad.sqlite") as db:
        revisions, bytes_total = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload_json)), 0) FROM belief_revisions"
        ).fetchone()
        technical_revisions = 0
        target_coverage = None
        target_condition = None
        target_id = report["target_registry_id"]
        for (payload,) in db.execute("SELECT payload_json FROM belief_revisions ORDER BY id"):
            record = json.loads(payload)
            cell = record.get("cell", {})
            if cell.get("domain") == "TECHNICAL":
                technical_revisions += 1
            if cell.get("registry_entity_id") != target_id:
                continue
            for claim in cell.get("claims", []):
                if claim["name"] == "surface_coverage":
                    target_coverage = claim.get("value")
                elif claim["name"] == "condition":
                    target_condition = claim.get("value")
    link_diagnostics = None
    link_path = path / "reports" / "impact_link_diagnostics.json"
    if link_path.exists():
        link_data = json.loads(link_path.read_text(encoding="utf-8"))
        harness = link_data["harness"]
        arm = harness["arms"]["primary"]
        queue = arm["queue_trace"]
        offers = harness["offer_log"]
        critical_value = harness["critical_value"]
        sender_revisions = arm["sender_latest_revisions"]
        receiver_revisions = arm["receiver_revisions"]
        lags = [
            int(revision) - int(receiver_revisions[belief_id])
            for belief_id, revision in sender_revisions.items()
            if receiver_revisions.get(belief_id) is not None
        ]
        link_diagnostics = {
            "routine_offers": sum(float(o[3]) < critical_value for o in offers),
            "critical_offers": sum(float(o[3]) >= critical_value for o in offers),
            "offered_wire": link_data["offered_wire"],
            "max_queue_entries": max((int(q[1]) for q in queue), default=0),
            "max_queue_bits": max((int(q[2]) for q in queue), default=0),
            "mean_queue_bits": sum(int(q[2]) for q in queue) / len(queue) if queue else 0.0,
            "coalesced": arm["coalesced"],
            "coalesced_queued_bits": arm["coalesced_queued_bits"],
            "receiver_revisions": receiver_revisions,
            "sender_latest_revisions": sender_revisions,
            "sender_belief_count": len(sender_revisions),
            "receiver_belief_count": len(receiver_revisions),
            "mean_receiver_revision_lag": sum(lags) / len(lags) if lags else None,
            "max_receiver_revision_lag": max(lags, default=None),
            "useful_receiver_bits": arm["useful_receiver_bits"],
            "obsolete_on_arrival_bits": arm["obsolete_on_arrival_bits"],
            "duplicate_contributions": arm["duplicate_contributions"],
        }
    return {
        "bundle": str(path),
        "git_commit": replay["git_commit"],
        "seed": replay["scenario_seed"],
        "scenario": replay["scenario_id"],
        "duration_s": replay["mission_runtime_config"]["duration_s"],
        "twin2t_truth_model": replay["mission_world_options"]["twin2t_truth_model"],
        "model2t_backend": replay["mission_runtime_config"]["model2t_backend"],
        "structural_observations": report["structural_observations"],
        "spatial_structural_observations": report["spatial_structural_observations"],
        "target_structural_observations": report["target_structural_observations"],
        "spatial_target_structural_observations": report["spatial_target_structural_observations"],
        "spatial_target_associated_evidence": report["spatial_target_associated_evidence"],
        "target_revision_count": report["target_revision_count"],
        "target_condition": target_condition,
        "target_surface_coverage": target_coverage,
        "target_error_before": report["target_error_before"],
        "target_error_after": report["target_error_after"],
        "mcbr_plans": len(runtime["plans"]),
        "decision_count": runtime["decisions"],
        "view_execution": runtime["view_execution"],
        "energy_j": report["vehicle"]["energy_used_j"],
        "estimated_travel_m": travel_m,
        "belief_revisions": revisions,
        "technical_belief_revisions": technical_revisions,
        "belief_payload_bytes": bytes_total,
        "beliefs_published": runtime["beliefs_published"],
        "communication": {
            key: communication[key]
            for key in (
                "offers",
                "bits_sent",
                "transmissions",
                "delivered",
                "dropped",
                "critical_offers",
                "critical_delivered",
                "receiver_beliefs",
                "receiver_alerts",
                "queue_bits",
            )
        },
        "link_diagnostics": link_diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = [summarize(p.resolve()) for p in args.bundles]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
