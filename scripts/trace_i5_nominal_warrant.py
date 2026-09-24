"""Development-only I5 nominal warrant trace; never gate evidence."""

from __future__ import annotations

import argparse
import json
from typing import Any
from uuid import UUID

import yaml

from conrad.domains.technical.registry import CORROSION_DEPTH, CRACK_LENGTH
from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments import m1_action_integrated as i5
from conrad.settings import REPO_ROOT
from conrad.sim.mission.run import prepare
from conrad.sim.mission.structure import rest_region_of
from conrad.sim.mission.unity_run import player_identity, prepare_unity

OUT = REPO_ROOT / "artifacts" / "experiments" / "I5-NOMINAL-FORENSICS"
CONFIG = yaml.safe_load((REPO_ROOT / "configs/eval/i5_unity.yaml").read_text(encoding="utf-8"))


def _belief(rt: Any, rid: UUID) -> dict[str, Any]:
    b = rt.m2t.beliefs[rid]
    return {
        "covered_cells": sorted(b.covered),
        "coverage": b.coverage_fraction(),
        "complete_fraction": b.coverage_complete,
        "condition_open": b.condition_open(),
        "uncertainty": b.uncertainty().model_dump(mode="json"),
        "corrosion": b.estimates[CORROSION_DEPTH].level,
        "crack": b.estimates[CRACK_LENGTH].level,
    }


def trace(backend: str, seed: int, scenario: str, tag: str = "") -> dict[str, Any]:
    with P.purpose_scope(P.Purpose.DESIGN):
        allowed = P.split(P.I5_V7_DOMAIN, "development", "design").world_seeds
    if seed not in allowed or scenario not in ("I5-NOMINAL", "I5-NOMINAL-READABLE"):
        raise ValueError("diagnostics require v7 development seed and nominal or readable scenario")
    if tag and (not tag.isascii() or not tag.replace("_", "").isalnum()):
        raise ValueError("tag must contain only ASCII letters, digits and underscores")
    suffix = f"-{tag}" if tag else ""
    run_id = f"I5-TRACE-{backend}-{scenario}-s{seed}{suffix}"
    root = OUT / "runs"
    session = (
        prepare_unity(scenario, i5.DEFAULT_CONFIG, run_id=run_id, runs_root=root, seed=seed, capture=False)
        if backend == "unity"
        else prepare(scenario, i5._settings(seed), run_id=run_id, runs_root=root, capture=False)
    )
    rt = session.runtime
    rid = UUID(session.world.recorder.meta["target_registry_id"])
    source = session.world.t2t._source
    assert source is not None
    rest = rest_region_of(source, session.world.target)
    truth_initial = {
        "target": source.structural_state["entities"][str(session.world.target)],
        "rest_region": None if rest is None else source.structural_state["entities"][str(rest)],
    }
    observations: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    original_structural = rt.perception._structural
    original_recorder = i5.RecordingEGDC

    def structural(obs: Any, now: Any) -> Any:
        before = set(rt.m2t.beliefs[rid].covered)
        start = len(rt.perception.structural_log)
        result = original_structural(obs, now)
        after = set(rt.m2t.beliefs[rid].covered)
        logs = rt.perception.structural_log[start:]
        target = [entry for entry in logs if entry["registry_id"] == str(rid)]
        if obs or after != before:
            observations.append(
                {
                    "t_s": now.time_ns / 1e9,
                    "technical_observations": len(obs),
                    "target_associated": len(target),
                    "target_readings": [
                        {
                            "observation_id": entry["observation_id"],
                            "sensor_id": entry["sensor_id"],
                            "measurements": entry["measurements"],
                            "spatial_support": (
                                None
                                if (ev := rt.s.repo.evidence(UUID(entry["evidence_id"]))) is None
                                or ev.spatial_support is None
                                else ev.spatial_support.model_dump(mode="json")
                            ),
                        }
                        for entry in target
                    ],
                    "new_cells": sorted(after - before),
                    "belief": _belief(rt, rid),
                }
            )
        return result

    class TracingRecorder(original_recorder):
        def decide(self, ctx: Any) -> Any:
            outcome = super().decide(ctx)
            state = i5.decision_state(
                ctx, {str(x) for x in rt.ctx.critical_component_ids}, i5.DecisionConfig()
            )
            decisions.append(
                {
                    **state,
                    "belief": _belief(rt, rid),
                    "active_information_needs": len(ctx.active_information_needs),
                    "unavailable_information_targets": len(ctx.unavailable_information_targets),
                    "plans": len(rt.deliberation.plans),
                    "active_goal": None if rt.executive.active is None else rt.executive.active.purpose,
                    "action": i5.label(outcome.record.chosen),
                }
            )
            return outcome

    rt.perception._structural = structural
    i5.RecordingEGDC = TracingRecorder
    try:
        row = i5.drive_and_score(session, seed, scenario, i5.PRIMARY, CONFIG)
        summary = {
            "evidence_class": "DEVELOPMENT DIAGNOSTIC; not gate evidence",
            "partition": "i5_mission_v7/development",
            "partition_digest": P.load_i5_v7()["digest"],
            "backend": backend,
            "seed": seed,
            "scenario": scenario,
            "tag": tag or "incumbent",
            "target_registry_id": str(rid),
            "truth_initial_evaluation_only": truth_initial,
            "player": player_identity() if backend == "unity" else None,
            "score": {"driving": row["driving"], "outcome": row["outcome"]},
            "observations": observations,
            "decisions": decisions,
            "final_belief": _belief(rt, rid),
            "structural_stats": rt.perception.stats.__dict__,
        }
        session.finish()
        summary["run_dir"] = str(session.run_dir)
    except BaseException:
        abort = getattr(session, "abort", None)
        if abort is not None:
            abort()
        raise
    finally:
        i5.RecordingEGDC = original_recorder
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{backend}_{scenario.lower().replace('-', '_')}_s{seed}{suffix}.json"
    path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return {
        "path": str(path),
        "warrant": row["driving"]["warrant_reached"],
        "coverage": summary["final_belief"]["coverage"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("backend", choices=("kernel", "unity"))
    ap.add_argument("seed", type=int)
    ap.add_argument("scenario", choices=("I5-NOMINAL", "I5-NOMINAL-READABLE"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    print(json.dumps(trace(args.backend, args.seed, args.scenario, args.tag)), flush=True)


if __name__ == "__main__":
    main()
