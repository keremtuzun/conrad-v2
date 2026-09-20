"""FLAGSHIP-UNITY: one integrated mission through the built Unity player with all three declared stressors.

Declaration: ``configs/eval/flagship_unity.yaml``. Write-up: ``docs/audits/FLAGSHIP_UNITY.md``.
This is a DEMONSTRATION run, not gate evidence.

The script

1. checks the world against the pinned unity_gate partition and the resolved runtime against the declared
   production defaults (it refuses to run when either differs);
2. flies the mission with ``prepare_unity`` (observation capture ON), sampling on every control step the
   Unity TRUE pose, the estimator pose and position sigma, the supervisor's safety state and the authorised
   thruster command;
3. writes small JSON artifacts under ``artifacts/flagship/`` (the run bundle stays out of git);
4. optionally runs the truth-removal falsification on the bundle.

Usage:
  python -m uv run python scripts/run_flagship_unity.py --partition development --seed 7810007  # dry run
  python -m uv run python scripts/run_flagship_unity.py --partition final_test                  # once
  python -m uv run python scripts/run_flagship_unity.py --analyse-only <bundle>
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs" / "eval" / "flagship_unity.yaml"
OUT_DIR = ROOT / "artifacts" / "flagship"
SCENARIO = "FLAGSHIP-UNITY"
NOT_INTACT = ("DEGRADED", "SEVERE", "FAILED")
QUANTITIES = ("corrosion_depth_m", "crack_length_m")


def load_config() -> dict[str, Any]:
    return dict(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


# ============================================================================================ pre-flight checks
def check_world(cfg: dict[str, Any], seed: int, partition: str) -> dict[str, Any]:
    """The world must come from the declared split of the pinned unity_gate partition file."""
    from conrad.evaluation.partitions import load_unity_gates, split

    purpose = "final_evaluation" if partition == "final_test" else "design"
    allowed = set(split("unity_gate", partition, purpose).world_seeds)
    if seed not in allowed:
        raise SystemExit(f"seed {seed} is not in the unity_gate {partition} split")
    declared = int(cfg["final_world"]) if partition == "final_test" else None
    if declared is not None and seed != declared:
        raise SystemExit(f"final_test runs only the declared world {declared}, not {seed}")
    if partition != "final_test" and seed not in set(cfg["development_worlds"]):
        raise SystemExit(f"development seed {seed} is not declared in {CONFIG.name}")
    return {"partition": partition, "seed": seed, "partition_digest": load_unity_gates()["digest"]}


def check_runtime(rcfg: Any, expect: dict[str, Any]) -> dict[str, Any]:
    """Refuse to fly unless the resolved runtime is the production one the config declares."""
    from conrad.domains.ecological.config import Model2EConfig
    from conrad.domains.spatial.config import spatial_config

    m2e = Model2EConfig.model_validate(dict(rcfg.model2e))
    m2s = spatial_config(rcfg.model2s)
    got = {
        "planner": rcfg.planner,
        "model2t_mode": rcfg.model2t_mode,
        "model2s_refinement_enabled": bool(m2s.refinement.enabled),
        "model2e_enabled": bool(rcfg.model2e_enabled),
        "model2e_observability_context": bool(m2e.switches.observability_context),
        "model2e_ecological_coupling": bool(m2e.switches.ecological_coupling),
        "model2e_model_version": m2e.model_version,
        "model2s_model_version": m2s.model_version,
        "multidomain_enabled": bool(rcfg.multidomain.enabled),
        "duration_s": float(rcfg.duration_s),
        "control_period_s": float(rcfg.control_period_s),
    }
    wrong = {k: (v, got.get(k)) for k, v in expect.items() if got.get(k) != v}
    if wrong:
        raise SystemExit(f"resolved runtime differs from the declared production defaults: {wrong}")
    return got


# ==================================================================================================== the flight
def fly(cfg: dict[str, Any], seed: int, runs_root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    """One Unity mission, sampled on every control step. Returns (bundle dir, per-step trace)."""
    from conrad.orchestration.mission_config import runtime_config
    from conrad.sim.mission.run import settings_for
    from conrad.sim.mission.unity_run import player_identity, prepare_unity, resolve_unity

    settings = settings_for(str(cfg["mission_config"]))
    _, runtime_raw = resolve_unity(SCENARIO, dict(settings.sim.get("mission", {})))
    rcfg = runtime_config(runtime_raw)
    runtime_seen = check_runtime(rcfg, dict(cfg["runtime_expectations"]))
    session = prepare_unity(
        SCENARIO, str(cfg["mission_config"]), run_id=run_id, runs_root=runs_root, seed=seed
    )
    rows: list[dict[str, Any]] = []
    try:
        steps = round(session.rcfg.duration_s / session.rcfg.control_period_s)
        for _ in range(steps):
            session.step()
            rows.append(_sample(session))
        out = session.finish()
    except BaseException:
        session.abort()
        raise
    bundle = Path(out["run_dir"])
    trace = {
        "run_id": out["run_id"],
        "run_dir": str(bundle),
        "seed": seed,
        "scenario": SCENARIO,
        "runtime": runtime_seen,
        "player": player_identity(),
        "steps": rows,
        "comms": session.runtime.shore.harness_report(),
        "report": out["report"],
    }
    return bundle, trace


def _sample(session: Any) -> dict[str, Any]:
    """Pure reads: Unity truth (evaluation only) + deployment-side estimator and supervisor state."""
    rt, world = session.runtime, session.world
    est = rt.stack.estimator
    state = est.get_state()
    true_p = np.asarray(world.hardware.true_state().pose.position_m, dtype=float)
    est_p = np.asarray(state.pose.position_m, dtype=float)
    return {
        "t_s": round(world.t_s, 4),
        "true_m": [float(v) for v in true_p],
        "est_m": [float(v) for v in est_p],
        "err_m": float(np.linalg.norm(true_p - est_p)),
        "sigma_m": float(est.position_sigma_m),
        "estimator_health": state.estimator_health.value,
        "estimator_reasons": list(est.health_reasons()),
        "safety_state": rt.executive.safety_state,
        "runtime_state": rt.supervisor.state.value,
        "link_status": rt.shore.link_state(world.t_s).status.value,
    }


# ======================================================================================================= analysis
def _revisions(bundle: Path) -> tuple[list[Any], UUID]:
    from conrad.persistence.db import make_engine
    from conrad.persistence.repository import Repository
    from conrad.schemas.world import Domain

    truth = json.loads((bundle / "truth" / "truth_record.json").read_text(encoding="utf-8"))
    target = UUID(truth["meta"]["target_registry_id"])
    engine = make_engine(bundle / "conrad.sqlite")
    try:
        revs = [
            r
            for r in Repository(engine).all_revisions()
            if r.cell.domain is Domain.TECHNICAL and r.cell.registry_entity_id == target
        ]
    finally:
        engine.dispose()
    return revs, target


def _claim(rev: Any, name: str) -> dict[str, Any]:
    c = rev.cell.claim(name)
    return {
        "value": None if c is None else c.value,
        "status": "UNKNOWN" if c is None else c.status.value,
        "U_O": None if c is None else c.uncertainty.observational,
        "U_A": None if c is None else c.uncertainty.aleatoric,
    }


def _row(rev: Any) -> dict[str, Any]:
    u = rev.cell.uncertainty
    return {
        "revision": rev.revision,
        "t_s": round(rev.measurement_time_ns / 1e9, 3),
        "update_kind": rev.update_kind.value,
        "U_A": u.aleatoric,
        "U_E": u.epistemic,
        "U_C": u.contradiction,
        "U_O": u.observational,
        "evidence": len(rev.consumed_evidence_ids),
        **{q: _claim(rev, q) for q in QUANTITIES},
        "condition": _claim(rev, "condition"),
    }


def structural_and_contradiction(bundle: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """The structural estimate against truth, and U_C on the target belief around the second payload."""
    revs, target = _revisions(bundle)
    truth = json.loads((bundle / "truth" / "truth_record.json").read_text(encoding="utf-8"))
    end = truth["target_states"]["end"]
    direct = [r for r in revs if r.update_kind.value == "DIRECT"]
    start_s = float(cfg["stressors"]["contradiction"]["second_sensor_start_s"])
    before = [r for r in direct if r.measurement_time_ns / 1e9 < start_s]
    after = [r for r in direct if r.measurement_time_ns / 1e9 >= start_s]
    last = revs[-1] if revs else None
    labels = truth["series"].get("structural_label", [])
    on_target = [x for x in labels if x.get("patch")]
    final = None if last is None else _row(last)
    errors: dict[str, Any] = {}
    for q in QUANTITIES:
        t = end.get(q)
        v = None if final is None else final[q]["value"]
        errors[q] = {
            "truth": t,
            "estimate": v,
            "abs_error_m": None if (v is None or t is None) else abs(float(v) - float(t)),
        }
    return {
        "target_registry_id": str(target),
        "sensor_model": "Twin2T REALISTIC (docs/audits/STRUCTURAL_LINEAGE_AUDIT.md), not the old optimistic model",
        "truth_target_end": end,
        "final_belief": final,
        "errors": errors,
        "revision_count": len(revs),
        "direct_revisions": len(direct),
        "target_structural_readings": len(on_target),
        "aux_readings": sum(1 for x in labels if x.get("aux")),
        "contradiction": {
            "second_sensor_start_s": start_s,
            "direct_before": [_row(r) for r in before],
            "direct_after": [_row(r) for r in after],
            "U_C_before_max": max((r.cell.uncertainty.contradiction for r in before), default=None),
            "U_C_after_max": max((r.cell.uncertainty.contradiction for r in after), default=None),
            "U_C_final": None if last is None else last.cell.uncertainty.contradiction,
        },
    }


def comms_timeline(bundle: Path, trace: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """When the critical finding was made, when the link returned, when the alert and the delta arrived."""
    revs, _ = _revisions(bundle)
    outage = cfg["stressors"]["communication_outage"]
    down, up = float(outage["link_down_s"]), float(outage["link_up_s"])
    finding_s: float | None = None
    finding_rev: int | None = None
    for r in revs:
        cond = r.cell.claim("condition")
        if cond is not None and cond.status.value == "OBSERVED" and str(cond.value) in NOT_INTACT:
            finding_s, finding_rev = r.measurement_time_ns / 1e9, r.revision
            break
    comms = trace["comms"]
    primary = comms["arms"]["primary"]
    crit = comms["critical_offers"]
    alert_ns: int | None = None
    delta_ns: int | None = None
    belief_id: str | None = None
    offer_ns: int | None = None
    if crit:
        belief_id, crit_rev, offer_ns = str(crit[0][0]), int(crit[0][1]), int(crit[0][2])
        arrivals = [(b, r, ns) for b, r, ns in primary["arrivals"] if b == belief_id and r >= crit_rev]
        alert_ns = min((ns for _, _, ns in arrivals), default=None)
        deltas = [
            x
            for x in primary["receipts"]
            if x["kind"] == "deltas" and x["belief_id"] == belief_id and x["applied"]
        ]
        delta_ns = min((int(x["t_ns"]) for x in deltas), default=None)
    routine = [
        int(x["t_ns"])
        for x in primary["receipts"]
        if x["kind"] == "deltas" and x["applied"] and x["belief_id"] != belief_id
    ]
    first_routine_after_up = min((ns for ns in routine if ns / 1e9 >= up), default=None)
    return {
        "link_down_s": down,
        "link_up_s": up,
        "critical_finding_t_s": finding_s,
        "critical_finding_revision": finding_rev,
        "finding_made_during_outage": None if finding_s is None else bool(down <= finding_s < up),
        "critical_belief_id": belief_id,
        "critical_first_offer_t_s": None if offer_ns is None else offer_ns / 1e9,
        "critical_alert_arrival_t_s": None if alert_ns is None else alert_ns / 1e9,
        "critical_delta_arrival_t_s": None if delta_ns is None else delta_ns / 1e9,
        "first_routine_delta_after_reconnection_t_s": (
            None if first_routine_after_up is None else first_routine_after_up / 1e9
        ),
        "alert_latency_from_finding_s": (
            None if (alert_ns is None or finding_s is None) else alert_ns / 1e9 - finding_s
        ),
        "delta_latency_from_finding_s": (
            None if (delta_ns is None or finding_s is None) else delta_ns / 1e9 - finding_s
        ),
        "receiver_revision_at_end": primary["receiver_revisions"].get(belief_id),
        "sender_revision_at_end": primary["sender_latest_revisions"].get(belief_id),
        "duplicate_contributions": primary["duplicate_contributions"],
        "resync_requests": primary["resync_requests"],
        "queue_max_units": max((q[1] for q in primary["queue_trace"]), default=0),
        "policy": primary["policy"],
        "metrics": trace["report"]["runtime"]["communication"],
    }


def _boundary(bundle: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """The mission boundary the SafetySupervisor uses, from the captured (deployment-side) mission context."""
    from conrad.orchestration.mission_context import MissionContext

    ctx = MissionContext.model_validate_json(
        (bundle / "capture" / "mission_context.json").read_text(encoding="utf-8")
    )
    lo, hi = ctx.spec.boundary_min_m, ctx.spec.boundary_max_m
    if lo is None or hi is None:
        return None
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def _thin(rows: list[dict[str, Any]], t0: float) -> list[dict[str, Any]]:
    """1 Hz before the fix outage, 0.5 s during it: a tracked trace small enough for the audit."""
    out: list[dict[str, Any]] = []
    last = -1e9
    for r in rows:
        step = 0.5 if r["t_s"] >= t0 else 1.0
        if r["t_s"] - last + 1e-9 < step:
            continue
        last = r["t_s"]
        out.append(
            {
                "t_s": r["t_s"],
                "err_m": round(r["err_m"], 4),
                "sigma_m": round(r["sigma_m"], 4),
                "estimator_health": r["estimator_health"],
                "estimator_reasons": r["estimator_reasons"],
                "safety_state": r["safety_state"],
                "link_status": r["link_status"],
            }
        )
    return out


def localization(bundle: Path, trace: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Estimator error and sigma through the fix outage, the safety timeline and the safe-state check."""
    from conrad.sim.mission.unity_faults import check_safe_state

    loc = cfg["stressors"]["localization_degradation"]
    t0 = float(loc["fix_outage_start_s"])
    rows = trace["steps"]
    t = np.array([r["t_s"] for r in rows])
    err = np.array([r["err_m"] for r in rows])
    sig = np.array([r["sigma_m"] for r in rows])
    during = t >= t0
    all_events = [dict(e) for e in trace["report"]["runtime"]["safety_events"]]
    first = {}
    for name in ("DEGRADED", "HOLD"):
        hit = next((e for e in all_events if e["to"] == name and e["t_s"] >= t0), None)
        first[name] = None if hit is None else hit["t_s"]
    events = list(all_events)
    lost = next(
        (r["t_s"] for r in rows if r["t_s"] >= t0 and "LOCALIZATION_LOST" in r["estimator_reasons"]), None
    )
    elevated = next(
        (r["t_s"] for r in rows if r["t_s"] >= t0 and "POSE_SIGMA_ELEVATED" in r["estimator_reasons"]), None
    )
    # conrad.sim.mission.unity_faults.check_safe_state, on a log rebuilt from the safety-event timeline
    # (the supervisor emits an event on every state change and on every newly seen reason code).
    log: list[list[Any]] = []
    reasons: list[str] = []
    for r in rows:
        while events and events[0]["t_s"] <= r["t_s"] + 1e-9:
            reasons = list(events.pop(0)["reasons"])
        log.append([round(r["t_s"] * 1e9), r["safety_state"], reasons, None, None])
    traj = {
        "t_s": [float(x) for x in t],
        "true_m": [r["true_m"] for r in rows],
        "estimated_m": [r["est_m"] for r in rows],
    }
    expect = {
        "state": str(loc["expect_state"]),
        "reason": str(loc["expect_reason"]),
        "deadline_s": float(loc["deadline_s"]),
        "motion": str(loc["motion"]),
        "hold_radius_m": float(loc["hold_radius_m"]),
    }
    collisions = int(trace["report"]["vehicle"]["collisions"])
    safe = check_safe_state(log, traj, expect, t0, None, collisions)
    after_lost = t >= (lost if lost is not None else math.inf)
    box = _boundary(bundle)
    outside: dict[str, Any] = {"boundary_min_m": None, "boundary_max_m": None}
    if box is not None:
        lo, hi = box
        true_xyz = np.asarray(traj["true_m"], dtype=float)
        est_xyz = np.asarray(traj["estimated_m"], dtype=float)
        outside = {
            "boundary_min_m": [float(v) for v in lo],
            "boundary_max_m": [float(v) for v in hi],
            "true_steps_outside": int(((true_xyz < lo) | (true_xyz > hi)).any(axis=1).sum()),
            "estimated_steps_outside": int(((est_xyz < lo) | (est_xyz > hi)).any(axis=1).sum()),
            "true_first_outside_t_s": next(
                (float(ti) for ti, p in zip(t, true_xyz, strict=True) if bool(((p < lo) | (p > hi)).any())),
                None,
            ),
        }
    return {
        "boundary": outside,
        "trace": _thin(rows, t0),
        "fix_outage_start_s": t0,
        "fix_outage_duration_s": float(loc["fix_outage_duration_s"]),
        "error_before_outage_max_m": float(err[~during].max()) if (~during).any() else None,
        "sigma_before_outage_max_m": float(sig[~during].max()) if (~during).any() else None,
        "error_during_outage_max_m": float(err[during].max()) if during.any() else None,
        "error_at_end_m": float(err[-1]),
        "sigma_at_end_m": float(sig[-1]),
        "max_error_over_sigma": float(np.max(err[during] / np.maximum(sig[during], 1e-9)))
        if during.any()
        else None,
        "first_pose_sigma_elevated_t_s": elevated,
        "first_localization_lost_t_s": lost,
        "time_to_localization_lost_s": None if lost is None else lost - t0,
        "first_safety_state_after_outage": first,
        "safety_events": all_events,
        "runtime_state_history": trace["report"]["runtime"]["runtime_state_history"],
        "true_speed_after_lost_mps": (
            None
            if not after_lost.any() or after_lost.sum() < 2
            else float(
                (
                    np.linalg.norm(np.diff(np.array(traj["true_m"])[after_lost], axis=0), axis=1)
                    / np.diff(t[after_lost])
                ).mean()
            )
        ),
        "safe_state_check": safe,
        "expect": expect,
    }


def execution(trace: dict[str, Any]) -> dict[str, Any]:
    rt = trace["report"]["runtime"]
    return {
        "commands_accepted": rt["commands_accepted"],
        "commands_rejected": rt["commands_rejected"],
        "commands_refused_by_safety_supervisor": rt["commands_refused_by_safety_supervisor"],
        "gateway_rejection_reasons": rt["gateway_rejection_reasons"],
        "collisions": trace["report"]["vehicle"]["collisions"],
        "min_clearance_m": trace["report"]["vehicle"]["min_clearance_m"],
        "uir": rt["uir"],
        "uir_report": rt["uir_report"],
        "decisions": rt["decisions"],
        "abstentions": rt["abstentions"],
        "failed_modules": rt["failed_modules"],
        "module_health": rt["module_health"],
        "beliefs_published": rt["beliefs_published"],
        "unity": trace["report"]["unity"],
    }


def analyse(bundle: Path, trace: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": cfg["experiment_id"],
        "kind": cfg["kind"],
        "scenario": SCENARIO,
        "backend": "unity",
        "seed": trace["seed"],
        "run_id": trace["run_id"],
        "run_dir": trace["run_dir"],
        "player": trace["player"],
        "identities": identities(bundle),
        "runtime": trace["runtime"],
        "structural": structural_and_contradiction(bundle, cfg),
        "communications": comms_timeline(bundle, trace, cfg),
        "localization": localization(bundle, trace, cfg),
        "execution": execution(trace),
    }


# ================================================================================================== falsification
def falsify(bundle: Path) -> dict[str, Any]:
    cmd = [sys.executable, str(ROOT / "scripts" / "falsify_truth_removal.py"), str(bundle)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    report = ROOT / "artifacts" / "falsification" / f"{bundle.parent.name}__{bundle.name}" / "report.json"
    out: dict[str, Any] = {"returncode": proc.returncode, "report_path": str(report)}
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        out["verdict"] = data.get("verdict")
        out["report"] = data
    else:
        out["stderr_tail"] = proc.stderr[-2000:]
    return out


# =========================================================================================== pinned identities
def identities(bundle: Path) -> dict[str, Any]:
    """Everything the bundle pins, so the run can be named exactly (bundle_manifest.json replay inputs)."""
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    inp = manifest["replay_inputs"]
    truth = json.loads((bundle / "truth" / "truth_record.json").read_text(encoding="utf-8"))["meta"]
    return {
        "scenario_id": inp["scenario_id"],
        "scenario_version": inp["scenario_version"],
        "scenario_uuid": truth["scenario_uuid"],
        "world_family": truth["world_family"],
        "seed": inp["scenario_seed"],
        "config_digest": inp["config_digest"],
        "sensor_configuration_digest": inp["sensor_configuration"],
        "robot_config_digest": inp["robot_config_digest"],
        "twin_versions": inp["twin_versions"],
        "model_versions": inp["model_versions"],
        "git_commit": inp["git_commit"],
        "architecture_id": inp["architecture_id"],
        "stack_id": inp["stack_id"],
        "unity": inp["unity"],
        "manifest_files": len(manifest["files"]),
        "manifest_objects": len(manifest.get("object_digests", {})),
    }


# ======================================================================================= full (CC-10) replay
def _sql(db: Path, query: str) -> list[tuple[Any, ...]]:
    import sqlite3

    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        return [tuple(r) for r in con.execute(query)]
    finally:
        con.close()


def _json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _jsonl_file(path: Path) -> list[Any]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _cmp(name: str, a: Any, b: Any) -> dict[str, Any]:
    idx = None
    if isinstance(a, list) and isinstance(b, list):
        idx = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
        if idx is None and len(a) != len(b):
            idx = min(len(a), len(b))
    return {
        "item": name,
        "original": len(a) if isinstance(a, list) else 1,
        "replayed": len(b) if isinstance(b, list) else 1,
        "equal": bool(a == b),
        "first_difference": idx,
    }


def _uncertainty(db: Path) -> list[tuple[Any, ...]]:
    from conrad.persistence.db import make_engine
    from conrad.persistence.repository import Repository

    engine = make_engine(db)
    try:
        revs = Repository(engine).all_revisions()
    finally:
        engine.dispose()
    out = []
    for r in revs:
        u = r.cell.uncertainty
        out.append((str(r.belief_id), r.revision, u.aleatoric, u.epistemic, u.contradiction, u.observational))
    return out


def _claims(path: Path) -> list[Any]:
    rows = []
    for rec in _jsonl_file(path):
        graph = rec.get("claim_graph") or {}
        rows.append(
            [
                rec.get("decision_id"),
                (rec.get("chosen") or {}).get("action_type"),
                bool(rec.get("abstained")),
                json.dumps(graph, sort_keys=True),
            ]
        )
    return rows


def replay_compare(original: Path, replayed: Path) -> list[dict[str, Any]]:
    """The itemised CC-10 comparison of the flagship run against its re-execution."""
    from conrad.runtime.event_log import EventType, event_signature, read_events

    da, db_ = original / "conrad.sqlite", replayed / "conrad.sqlite"
    ma, mb = original / "mission", replayed / "mission"
    ua, ub = _uncertainty(da), _uncertainty(db_)
    dsig = [
        max(abs(x - y) for x, y in zip(p[2:], q[2:], strict=True))
        for p, q in zip(ua, ub, strict=False)
        if p[:2] == q[:2]
    ]
    cmds_a = [
        json.dumps(e.payload, sort_keys=True, default=str)
        for e in read_events(original / "events.jsonl")
        if e.event_type is EventType.COMMAND_SENT
    ]
    cmds_b = [
        json.dumps(e.payload, sort_keys=True, default=str)
        for e in read_events(replayed / "events.jsonl")
        if e.event_type is EventType.COMMAND_SENT
    ]
    obs = "SELECT observation_id, sensor_id, modality, measurement_time_ns, payload_digest FROM observations"
    ev = (
        "SELECT evidence_id, source_observation_id, modality, measurement_time_ns, independence_group, "
        "payload_json FROM evidence"
    )
    rev = (
        "SELECT belief_id, revision, predecessor_revision, update_kind, measurement_time_ns, late, "
        "provenance_root, commit_sequence FROM belief_revisions ORDER BY id"
    )
    items = [
        _cmp(
            "event signature",
            event_signature(list(read_events(original / "events.jsonl"))),
            event_signature(list(read_events(replayed / "events.jsonl"))),
        ),
        _cmp("observation identities", _sql(da, obs), _sql(db_, obs)),
        _cmp("evidence digests", _sql(da, ev), _sql(db_, ev)),
        _cmp("belief revision order", _sql(da, rev), _sql(db_, rev)),
        _cmp("Model1 claims and decisions", _claims(ma / "decisions.jsonl"), _claims(mb / "decisions.jsonl")),
        _cmp("information needs", _json_file(ma / "requirements.json"), _json_file(mb / "requirements.json")),
        _cmp(
            "MCBR candidate ranking and selected plan",
            _json_file(ma / "mcbr_candidate_tables.json"),
            _json_file(mb / "mcbr_candidate_tables.json"),
        ),
        _cmp(
            "navigation goals",
            (_json_file(ma / "trajectories.json") or {}).get("goals"),
            (_json_file(mb / "trajectories.json") or {}).get("goals"),
        ),
        _cmp("command sequence", cmds_a, cmds_b),
        _cmp(
            "BAAC deltas",
            _jsonl_file(ma / "baac_transmissions.jsonl"),
            _jsonl_file(mb / "baac_transmissions.jsonl"),
        ),
        _cmp(
            "receiver state", _json_file(ma / "receiver_state.json"), _json_file(mb / "receiver_state.json")
        ),
        _cmp(
            "terminal mission status",
            (_json_file(ma / "runtime_metrics.json") or {}).get("runtime_state_history"),
            (_json_file(mb / "runtime_metrics.json") or {}).get("runtime_state_history"),
        ),
    ]
    items.append(
        {
            "item": "uncertainty (UA, UE, UC, UO) per revision",
            "original": len(ua),
            "replayed": len(ub),
            "equal": ua == ub,
            "max_abs_difference": max(dsig) if dsig else None,
            "tolerance": 0.0,
            "first_difference": next((i for i, d in enumerate(dsig) if d > 0.0), None),
        }
    )
    return items


def full_replay(bundle: Path, scratch: Path) -> dict[str, Any]:
    from conrad.sim.mission.unity_run import replay_unity_run

    report = replay_unity_run(bundle, scratch=scratch)
    report["items"] = replay_compare(bundle, Path(report["replay_dir"]))
    report["all_items_equal"] = all(bool(i["equal"]) for i in report["items"])
    return report


# ============================================================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", choices=("development", "final_test"), default="final_test")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-falsify", action="store_true")
    ap.add_argument("--analyse-only", default=None, help="re-analyse an existing bundle + trace")
    ap.add_argument("--replay", default=None, help="full CC-10 replay of a bundle (launches a fresh player)")
    ap.add_argument("--replay-scratch", default=None)
    args = ap.parse_args()
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.replay:
        bundle = Path(args.replay)
        scratch = Path(args.replay_scratch) if args.replay_scratch else bundle.parent / "replay"
        rep = full_replay(bundle, scratch)
        out = OUT_DIR / f"replay_{bundle.name}.json"
        out.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str), encoding="utf-8")
        print(json.dumps(rep, indent=1, default=str))
        print(f"\nwrote {out}")
        return 0

    if args.analyse_only:
        bundle = Path(args.analyse_only)
        trace = json.loads((OUT_DIR / f"trace_{bundle.name}.json").read_text(encoding="utf-8"))
        report = analyse(bundle, trace, cfg)
        previous = OUT_DIR / f"{SCENARIO}-s{report['seed']}.json"
        if previous.is_file():  # keep what re-analysis cannot recompute (the world check, the falsification)
            kept = json.loads(previous.read_text(encoding="utf-8"))
            for key in ("world", "falsification", "full_replay"):
                if key in kept:
                    report[key] = kept[key]
    else:
        seed = args.seed if args.seed is not None else int(cfg["final_world"])
        world = check_world(cfg, seed, args.partition)
        runs_root = (
            Path(args.runs_root) if args.runs_root else ROOT / "artifacts" / "runs_capture" / "flagship_unity"
        )
        run_id = args.run_id or f"{SCENARIO}-s{seed}"
        bundle, trace = fly(cfg, seed, runs_root, run_id)
        (OUT_DIR / f"trace_{bundle.name}.json").write_text(json.dumps(trace, default=str), encoding="utf-8")
        report = analyse(bundle, trace, cfg)
        report["world"] = world

    if not args.no_falsify and not args.analyse_only:
        report["falsification"] = falsify(Path(report["run_dir"]))
    path = OUT_DIR / f"{SCENARIO}-s{report['seed']}.json"
    path.write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "falsification"}, indent=1, default=str))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
