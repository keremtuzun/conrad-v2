"""I7-UNITY-E001: gate I7 on the FORMAL Unity path (ADR-0008), plus the same harness on the python kernel.

The communication harness is the surrogate's, unchanged: one mission drives a BAAC primary arm and four
shadow arms of the same ``ShoreLink`` (raw/send-all, FIFO, fixed priority, value-per-bit) that receive the
identical offer stream at identical times over a link with the identical profile and channel seed, and every
arm is scored with ``conrad.evaluation.decision_experiments.com_i7`` against the communication oracle. Only
the mission backend changes: ``prepare_unity`` (built player, lock-step TCP) instead of ``prepare``.

``backend="kernel"`` exists to VERIFY the harness without a player. It produces SURROGATE numbers and is
refused on final worlds; only ``backend="unity"`` may touch the declared final worlds.

Worlds, arms, bandwidth levels, the reduced-sweep declaration and the per-criterion decision rule live in
``configs/eval/i7_unity.yaml`` and are read from there, never hard-coded here. All link numbers are
SYNTHETIC_ONLY.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.com_i7 import (
    NS,
    POLICIES,
    _reconnection_trace,
    _score_arm,
)
from conrad.evaluation.oracle.comm_oracle import sync_error
from conrad.orchestration.mission_config import runtime_config
from conrad.settings import REPO_ROOT, ConradSettings, load_settings
from conrad.sim.mission.run import prepare
from conrad.sim.mission.scenarios import resolve
from conrad.sim.mission.unity_run import player_identity, prepare_unity, resolve_unity

CONFIG_PATH = REPO_ROOT / "configs" / "eval" / "i7_unity.yaml"
KERNEL = "kernel"
UNITY = "unity"
EVIDENCE_CLASS = {UNITY: "FORMAL (built Unity V2 player)", KERNEL: "SURROGATE (python L1 kernel mission)"}
BANDWIDTH = "bandwidth"
OUTAGE = "outage"


# ------------------------------------------------------------------------------------------- declaration
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def sweep_of(cfg: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    """One declared sweep (``full`` or ``reduced``) with its name attached."""
    key = name or str(cfg["default_sweep"])
    if key not in cfg["sweeps"]:
        raise KeyError(f"unknown I7 sweep {key!r}; declared: {sorted(cfg['sweeps'])}")
    return {"name": key, **dict(cfg["sweeps"][key])}


def final_worlds(cfg: dict[str, Any]) -> list[int]:
    """The declared held-out Unity worlds, checked against the digest-pinned unity_gate final_test split."""
    seeds = [int(s) for s in cfg["final_worlds"]]
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        allowed = set(P.split("unity_gate", "final_test", "final_evaluation").world_seeds)
    if not set(seeds) <= allowed:
        raise ValueError("I7 final worlds must be unity_gate final_test worlds")
    return seeds


def development_worlds(cfg: dict[str, Any], backend: str) -> list[int]:
    """Design-partition worlds for harness verification (unity_gate development, or mission development)."""
    if backend == UNITY:
        seeds = [int(s) for s in cfg["development_worlds"]]
        with P.purpose_scope(P.Purpose.DESIGN):
            allowed = set(P.split("unity_gate", "development", "design").world_seeds)
    else:
        seeds = [int(s) for s in cfg["kernel_check_seeds"]]
        with P.purpose_scope(P.Purpose.DESIGN):
            allowed = set(P.split("mission", "development", "design").world_seeds)
    if not set(seeds) <= allowed:
        raise ValueError(f"I7 {backend} development worlds are not development seeds")
    return seeds


def policy_names(cfg: dict[str, Any]) -> dict[str, str]:
    """Arm name -> scheduler policy, checked against the shared policy table of the surrogate."""
    arms = {str(k): str(v) for k, v in cfg["arms"].items()}
    if arms != POLICIES:
        raise ValueError(f"I7 arms {arms} differ from the surrogate's policy table {POLICIES}")
    return arms


# ------------------------------------------------------------------------------------------------ flights
def flight_settings(
    cfg: dict[str, Any], seed: int, scenario: str, factor: float, primary: str, shadows: list[str]
) -> ConradSettings:
    """The declared mission config with the bandwidth level, the primary policy and the shadow arms set."""
    arms = policy_names(cfg)
    base = load_settings(REPO_ROOT / str(cfg["mission_config"]))
    mission = dict(base.sim.get("mission", {}))
    if scenario in _unity_ids(cfg):
        _, rt = resolve_unity(scenario, mission)
    else:
        _, rt = resolve(scenario, mission)
    ref_bps = runtime_config(rt).link.bandwidth_bps  # the declared mission link
    runtime = dict(mission.get("runtime", {}))
    runtime["link"] = {**dict(runtime.get("link", {})), "bandwidth_bps": ref_bps * factor}
    runtime["baac"] = {
        **dict(runtime.get("baac", {})),
        "scheduler_policy": arms[primary],
        "shadow_arms": [{"name": n, "policy": arms[n]} for n in shadows],
    }
    sim = {**base.sim, "mission": {**mission, "runtime": runtime}}
    settings = base.model_copy(update={"sim": sim, "run": base.run.model_copy(update={"seed": int(seed)})})
    assert isinstance(settings, ConradSettings)
    return settings


def _unity_ids(cfg: dict[str, Any]) -> set[str]:
    return {str(v) for v in cfg["unity_scenarios"].values()}


def _kernel_scenario(cfg: dict[str, Any], role: str) -> str:
    """The kernel scenario a Unity I7 scenario is built from (verification backend)."""
    from conrad.sim.mission.unity_run import UNITY_SCENARIOS

    return str(UNITY_SCENARIOS[str(cfg["unity_scenarios"][role])]["base"])


def flight_jobs(cfg: dict[str, Any], worlds: list[int], sweep: dict[str, Any], backend: str) -> list[dict]:
    """Every flight of one sweep, in the order they must be flown (one Unity player at a time)."""
    arms = list(policy_names(cfg))
    primary = str(cfg["primary"])
    jobs: list[dict[str, Any]] = []
    for seed in worlds:
        for role, levels in ((BANDWIDTH, "bandwidth_levels"), (OUTAGE, "outage_levels")):
            scenario = str(cfg["unity_scenarios"][role]) if backend == UNITY else _kernel_scenario(cfg, role)
            for f in [float(x) for x in sweep[levels]]:
                jobs.append(
                    {
                        "seed": seed,
                        "role": role,
                        "scenario": scenario,
                        "factor": f,
                        "primary": primary,
                        "shadows": [a for a in arms if a != primary],
                        "run_id": f"I7-{backend}-{role}-s{seed}-bw{f:g}",
                    }
                )
        for f in [float(x) for x in sweep.get("closed_loop_levels", [])]:
            scenario = (
                str(cfg["unity_scenarios"][OUTAGE]) if backend == UNITY else _kernel_scenario(cfg, OUTAGE)
            )
            for p in [a for a in arms if a != primary]:
                jobs.append(
                    {
                        "seed": seed,
                        "role": OUTAGE,
                        "scenario": scenario,
                        "factor": f,
                        "primary": p,
                        "shadows": [],
                        "run_id": f"I7-{backend}-closed-{p}-s{seed}-bw{f:g}",
                    }
                )
    return jobs


def _extra(rep: dict[str, Any], arm: Any, shore: Any, outages: list[tuple[float, float]]) -> dict[str, Any]:
    """The measurements the ch26 Phase 11 criterion needs beyond ``com_i7._score_arm``."""
    latest = {b: m for b, (m, _) in shore.intent_latest.items()}
    trace = rep["queue_trace"]
    held = {}
    for a, b in outages:
        window = [q for q in trace if a <= q[0] < b]
        crit_offer = [t / NS for _, _, t in shore.critical_offers if a <= t / NS < b]
        after = [q for q in window if not crit_offer or q[0] >= min(crit_offer)]
        held[f"{a:g}-{b:g}"] = {
            "samples": len(after),
            "critical_units_min": min((q[3] for q in after), default=0),
            "critical_units_max": max((q[3] for q in after), default=0),
            "units_max": max((q[1] for q in window), default=0),
            "held_whole_outage": bool(after) and all(q[3] >= 1 for q in after),
        }
    return {
        "receiver_sync_error": sync_error(arm.receiver, latest),
        "receiver_alerts": len(rep["receiver_alerts"]),
        "queue_critical_max": max((q[3] for q in trace), default=0),
        "queue_critical_end": trace[-1][3] if trace else 0,
        "critical_held_during_outage": held,
        "resync_requests": rep["resync_requests"],
        "reevaluations": rep["reevaluations"],
        "drop_reasons": rep["drop_reasons"],
    }


def fly(cfg: dict[str, Any], job: dict[str, Any], backend: str, runs_root: Path) -> dict[str, Any]:
    """One integrated mission on one backend, scored exactly as the surrogate scores it."""
    seed, scenario, factor = int(job["seed"]), str(job["scenario"]), float(job["factor"])
    primary, shadows = str(job["primary"]), list(job["shadows"])
    settings = flight_settings(cfg, seed, scenario, factor, primary, shadows)
    started = time.perf_counter()
    if backend == UNITY:
        session: Any = prepare_unity(
            scenario, settings, run_id=str(job["run_id"]), runs_root=runs_root, seed=seed
        )
    else:
        session = prepare(scenario, settings, run_id=str(job["run_id"]), runs_root=runs_root)
    try:
        session.run()
        outcome = session.finish()
    except BaseException:
        if backend == UNITY:
            session.abort()
        raise
    wall_s = time.perf_counter() - started
    shore = session.runtime.shore
    harness = shore.harness_report()
    duration = float(session.rcfg.duration_s)
    steps = round(duration / float(session.rcfg.control_period_s))
    outages = [(float(a), float(b)) for a, b in shore.profile.outages_s]
    arms: dict[str, Any] = {}
    reconnect: dict[str, Any] = {}
    for arm_name, arm in shore.arms.items():
        key = primary if arm_name == "primary" else arm_name
        rep = harness["arms"][arm_name]
        arms[key] = {**_score_arm(rep, arm, shore, duration, cfg), **_extra(rep, arm, shore, outages)}
        if outages:
            reconnect[key] = _reconnection_trace(rep, shore)
    report = outcome["report"]
    rt = dict(report.get("runtime", {}))
    history = list(rt.get("runtime_state_history", []))
    return {
        "run_id": str(job["run_id"]),
        "backend": backend,
        "seed": seed,
        "role": str(job["role"]),
        "scenario": scenario,
        "bandwidth_factor": factor,
        "primary": primary,
        "run_dir": _rel(outcome["run_dir"]),
        "link": harness["link"],
        "outages_s": [list(o) for o in outages],
        # "full mission" means the world clock reached the declared duration AND the deployment stack was
        # still whole: a mission whose sensing module died at 8 s and ended in EMERGENCY_STOP is not a full
        # mission, however many control steps the driver kept executing afterwards.
        "mission": {
            "duration_s": duration,
            "control_period_s": float(session.rcfg.control_period_s),
            "declared_steps": steps,
            "world_time_s": round(float(session.world.t_s), 3),
            "reached_declared_duration": float(session.world.t_s)
            >= duration - float(session.rcfg.control_period_s),
            "failed_modules": sorted(str(m) for m in rt.get("failed_modules", [])),
            "terminal_state": history[-1] if history else None,
            "beliefs_published": rt.get("beliefs_published"),
            "ran_full_mission": bool(
                float(session.world.t_s) >= duration - float(session.rcfg.control_period_s)
                and not rt.get("failed_modules")
            ),
            "wall_clock_s": round(wall_s, 1),
        },
        "patch_first_visible_t_s": report.get("patch_first_visible_t_s"),
        "critical_offers": harness["critical_offers"],
        "offers": len(harness["offer_log"]),
        "beliefs_reported": len(harness["intent_latest"]),
        "arms": arms,
        "reconnection": reconnect,
    }


def _rel(path: str) -> str:
    p = Path(path)
    return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)


# ------------------------------------------------------------------------------------------- the criteria
C_BANDWIDTH = "full mission under constrained bandwidth"
C_OUTAGE = "full mission under outages"
C_RETAIN = "BAAC retains more mission-relevant information than raw/FIFO/fixed-priority"
C_LATENCY = "critical latency and sync error compared against baselines"
CRITERIA = (C_BANDWIDTH, C_OUTAGE, C_RETAIN, C_LATENCY)
RETAINED = "mission_information_retained"


def _rows(runs: list[dict[str, Any]], seed: int, role: str) -> list[dict[str, Any]]:
    """The shadow-arm flights of one world and role (closed-loop flights have another primary)."""
    return [r for r in runs if r["seed"] == seed and r["role"] == role and r["primary"] == "baac"]


def _delivered(arm: dict[str, Any]) -> bool:
    """At least one increment of any fidelity reached this arm's receiver."""
    return bool(
        int(arm.get("receiver_alerts", 0)) > 0
        or int(arm["receiver_beliefs"]) > 0
        or arm["critical_alert_latency_s"] is not None
    )


def bandwidth_measure(runs: list[dict[str, Any]], seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Criterion 1: every declared level flew a full mission and delivered under the constraint.

    The level check is DELIVERY, not score: whether BAAC's retained score is above zero at 1 % or 0.1 % is a
    property of the world (an F0 alert scores zero once the belief has been revised again, see
    ``comm_oracle.belief_score``), not of the mission running under the constraint. Superiority is criterion 3.
    """
    rows = _rows(runs, seed, BANDWIDTH)
    levels = {}
    for r in rows:
        baac = r["arms"]["baac"]
        zero = r["bandwidth_factor"] == 0.0
        levels[f"{r['bandwidth_factor'] * 100:g}pct"] = {
            "ran_full_mission": r["mission"]["ran_full_mission"],
            "world_time_s": r["mission"]["world_time_s"],
            "failed_modules": r["mission"]["failed_modules"],
            "terminal_state": r["mission"]["terminal_state"],
            "beliefs_published": r["mission"]["beliefs_published"],
            "arms_measured": sorted(r["arms"]),
            "baac_retained": baac[RETAINED],
            "baac_bits_sent": baac["bits_sent"],
            "bits_all_arms": {a: m["bits_sent"] for a, m in r["arms"].items()},
            "retained_per_mbit": {a: m["retained_per_mbit"] for a, m in r["arms"].items()},
            "queue_depth_max": {a: m["queue_depth_max"] for a, m in r["arms"].items()},
            "backlog_bits_max": {a: m["backlog_bits_max"] for a, m in r["arms"].items()},
            "duplicate_contributions": {a: m["duplicate_contributions"] for a, m in r["arms"].items()},
            "baac_delivered_something": _delivered(baac),
            "ok": bool(
                r["mission"]["ran_full_mission"]
                and sorted(r["arms"]) == sorted(POLICIES)
                and all(m["duplicate_contributions"] == 0 for m in r["arms"].values())
                and (
                    all(m["bits_sent"] == 0 for m in r["arms"].values())
                    and not any(_delivered(m) for m in r["arms"].values())
                    if zero
                    else _delivered(baac)
                )
            ),
        }
    return {"levels": levels, "ok": bool(levels) and all(v["ok"] for v in levels.values())}


def outage_measure(runs: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Criterion 2: the finding is made during the outage, is held in the queue, and arrives after reconnect.

    What must hold at mission end is that the receiver holds AT LEAST the critical revision that was found
    during the outage. Exact end-of-mission equality with the sender's newest revision is not part of "full
    mission under outages": on a world whose critical belief is revised hundreds of times, the last offer can
    be younger than the link latency. That end-of-mission sync error is measured and compared against the
    baselines under criterion 4, which is where ch26 Phase 11 puts it.
    """
    rows = _rows(runs, seed, OUTAGE)
    out = {}
    for r in rows:
        baac = r["arms"]["baac"]
        crit = baac["critical"]
        found = {c["belief_id"]: int(c["revision"]) for c in crit}
        at_receiver = [
            (s["belief_id"], found[s["belief_id"]], s["receiver_revision"])
            for s in baac["sync_critical"]
            if s["belief_id"] in found
        ]
        recon = r["reconnection"].get("baac", [])
        held = list(baac["critical_held_during_outage"].values())
        reconnect_s = max((float(o[1]) for o in r["outages_s"]), default=0.0)
        delta_arrival = [
            c["finding_t_s"] + c["delta_latency_s"] for c in crit if c["delta_latency_s"] is not None
        ]
        row = {
            "ran_full_mission": r["mission"]["ran_full_mission"],
            "failed_modules": r["mission"]["failed_modules"],
            "critical_findings": len(crit),
            "link_down_at_finding": [c["link_down_at_finding"] for c in crit],
            "finding_t_s": [c["finding_t_s"] for c in crit],
            "alert_latency_s": baac["critical_alert_latency_s"],
            "delta_latency_s": baac["critical_delta_latency_s"],
            "delta_arrival_s": delta_arrival,
            "reconnect_s": reconnect_s,
            "critical_delivered_first": [bool(x["critical_delivered_first"]) for x in recon],
            "critical_held_whole_outage": [bool(h["held_whole_outage"]) for h in held],
            "critical_revision_at_receiver": [[b, rev, got] for b, rev, got in at_receiver],
            "critical_revision_delivered": bool(at_receiver)
            and all(got is not None and int(got) >= rev for _, rev, got in at_receiver),
            "sync_critical_all_equal": baac["sync_critical_all_equal"],
            "receiver_sync_error": baac["receiver_sync_error"],
            "coalesced": baac["coalesced"],
            "duplicate_contributions": baac["duplicate_contributions"],
            "resync_requests": baac["resync_requests"],
        }
        row["ok"] = bool(
            r["mission"]["ran_full_mission"]
            and crit
            and all(row["link_down_at_finding"])
            and row["critical_held_whole_outage"]
            and all(row["critical_held_whole_outage"])
            and recon
            and all(row["critical_delivered_first"])
            and delta_arrival
            and all(t >= reconnect_s for t in delta_arrival)
            and row["critical_revision_delivered"]
            and int(row["coalesced"]) > 0
            and int(row["duplicate_contributions"]) == 0
            and int(row["resync_requests"]) == 0
        )
        out[f"{r['bandwidth_factor'] * 100:g}pct"] = row
    return {"levels": out, "ok": bool(out) and all(v["ok"] for v in out.values())}


def retention_measure(runs: list[dict[str, Any]], seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Criterion 3: BAAC strictly above raw / FIFO / fixed priority at every non-zero level and in the outage."""
    spec = [str(p) for p in cfg["spec_comparison"]]
    cells = {}
    ok = True
    for r in _rows(runs, seed, BANDWIDTH) + _rows(runs, seed, OUTAGE):
        if r["bandwidth_factor"] == 0.0:
            continue
        b = float(r["arms"]["baac"][RETAINED])
        diffs = {a: b - float(m[RETAINED]) for a, m in r["arms"].items() if a != "baac"}
        cell_ok = all(diffs[p] > 0 for p in spec)
        ok = ok and cell_ok
        cells[f"{r['role']}-{r['bandwidth_factor'] * 100:g}pct"] = {
            "baac_retained": b,
            "baac_minus_policy": diffs,
            "spec_set_beaten": cell_ok,
            "value_per_bit_reported": diffs.get("value_per_bit"),
        }
    return {"cells": cells, "spec_comparison": spec, "ok": bool(cells) and ok}


def latency_sync_measure(runs: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Criterion 4 (ch26 Phase 11): the comparison itself, with every case where BAAC is worse listed."""
    inf = float("inf")
    cells: dict[str, Any] = {}
    worse: list[str] = []
    complete = True
    for r in _rows(runs, seed, BANDWIDTH) + _rows(runs, seed, OUTAGE):
        if r["bandwidth_factor"] == 0.0:
            continue
        cond = f"{r['role']}-{r['bandwidth_factor'] * 100:g}pct"
        row = {}
        for a, m in r["arms"].items():
            keys = ("critical_alert_latency_s", "critical_delta_latency_s", "receiver_sync_error")
            complete = complete and all(k in m for k in keys) and "sync_critical_all_equal" in m
            row[a] = {
                **{k: m[k] for k in keys},
                "critical_in_sync": bool(m["sync_critical_all_equal"]),
                "deadline_success": m["deadline_success"],
                "retained_per_mbit": m["retained_per_mbit"],  # information per bit
            }
        base = row["baac"]
        for a, m in row.items():
            if a == "baac":
                continue
            lat = m["critical_alert_latency_s"]
            blat = base["critical_alert_latency_s"]
            if (inf if lat is None else lat) < (inf if blat is None else blat):
                worse.append(f"{cond} alert latency: {a} {lat} < baac {blat}")
            if float(m["receiver_sync_error"]) < float(base["receiver_sync_error"]):
                worse.append(
                    f"{cond} sync error: {a} {m['receiver_sync_error']} < baac {base['receiver_sync_error']}"
                )
            if m["critical_in_sync"] and not base["critical_in_sync"]:
                worse.append(f"{cond} critical sync: {a} in sync, baac not")
        cells[cond] = row
    return {"cells": cells, "baac_worse_than_baseline_in": worse, "ok": bool(cells) and complete}


def measure(runs: list[dict[str, Any]], seed: int, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        C_BANDWIDTH: bandwidth_measure(runs, seed, cfg),
        C_OUTAGE: outage_measure(runs, seed),
        C_RETAIN: retention_measure(runs, seed, cfg),
        C_LATENCY: latency_sync_measure(runs, seed),
    }


# ------------------------------------------------------------------------------------------------- driver
def run_gate(
    cfg: dict[str, Any],
    worlds: list[int],
    sweep: dict[str, Any],
    backend: str,
    runs_root: Path,
    results_path: Path | None = None,
) -> dict[str, Any]:
    """Fly every job of one sweep SEQUENTIALLY (one Unity player at a time) and score each world."""
    declared_final = {int(s) for s in cfg["final_worlds"]}
    if backend == KERNEL and set(worlds) & declared_final:
        raise ValueError("the kernel backend is SURROGATE: it may never run on the I7 final worlds")
    if backend == UNITY and not set(worlds) <= declared_final | set(development_worlds(cfg, UNITY)):
        raise ValueError("Unity flights must use the declared final worlds or the declared dev worlds")
    jobs = flight_jobs(cfg, worlds, sweep, backend)
    runs = [fly(cfg, j, backend, runs_root) for j in jobs]
    per_world = {str(s): measure(runs, s, cfg) for s in worlds}
    criteria = {c: {w: bool(m[c]["ok"]) for w, m in per_world.items()} for c in CRITERIA}
    result = {
        "experiment_id": cfg["experiment_id"],
        "gate": cfg["gate"],
        "backend": backend,
        "evidence_class": EVIDENCE_CLASS[backend],
        "data_status": "SYNTHETIC_ONLY",
        "config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "sweep": sweep,
        "reduced_sweep": bool(sweep.get("reduced", False)),
        "partition_file": "configs/eval/partitions_unity_gates.yaml",
        "partition_digest": P.load_unity_gates()["digest"],
        "worlds": worlds,
        "arms": policy_names(cfg),
        "decision_rule": cfg["decision_rule"],
        "player": player_identity() if backend == UNITY else None,
        "flights": len(runs),
        "wall_clock_s": round(sum(float(r["mission"]["wall_clock_s"]) for r in runs), 1),
        "criteria": criteria,
        "per_world": per_world,
        "runs": runs,
    }
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result


def wiring_check(cfg: dict[str, Any], sweep_name: str | None = None) -> dict[str, Any]:
    """Dry run: the Unity path resolves end to end without launching anything (no world is built)."""
    from conrad.evaluation.gates import GATE_BY_ID
    from conrad.sim.mission.unity_run import UNITY_SCENARIOS

    sweep = sweep_of(cfg, sweep_name)
    worlds = final_worlds(cfg)
    jobs = flight_jobs(cfg, worlds, sweep, UNITY)
    resolved = []
    for job in jobs:
        settings = flight_settings(
            cfg,
            int(job["seed"]),
            str(job["scenario"]),
            float(job["factor"]),
            str(job["primary"]),
            list(job["shadows"]),
        )
        _, rt = resolve_unity(str(job["scenario"]), dict(settings.sim.get("mission", {})))
        rcfg = runtime_config(rt)
        resolved.append(
            {
                "run_id": job["run_id"],
                "scenario": job["scenario"],
                "bandwidth_bps": rcfg.link.bandwidth_bps,
                "outages_s": [list(o) for o in rcfg.link.outages_s],
                "duration_s": rcfg.duration_s,
                "control_period_s": rcfg.control_period_s,
                "policy": rcfg.baac["scheduler_policy"],
                "shadow_arms": [a["name"] for a in rcfg.baac["shadow_arms"]],
            }
        )
    return {
        "sweep": sweep["name"],
        "reduced": bool(sweep.get("reduced", False)),
        "worlds": worlds,
        "flights": len(jobs),
        "unity_scenarios_registered": sorted(_unity_ids(cfg) & set(UNITY_SCENARIOS)),
        "unity_scenarios_missing": sorted(_unity_ids(cfg) - set(UNITY_SCENARIOS)),
        "criteria_match_gates_py": tuple(CRITERIA) == GATE_BY_ID["I7"].criteria,
        "player": player_identity(),
        "resolved": resolved,
    }


__all__ = [
    "CRITERIA",
    "C_BANDWIDTH",
    "C_LATENCY",
    "C_OUTAGE",
    "C_RETAIN",
    "KERNEL",
    "UNITY",
    "development_worlds",
    "final_worlds",
    "flight_jobs",
    "flight_settings",
    "fly",
    "load_config",
    "measure",
    "run_gate",
    "sweep_of",
    "wiring_check",
]
