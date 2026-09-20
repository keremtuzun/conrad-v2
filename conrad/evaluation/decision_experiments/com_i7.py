"""COM-I7-E001 / COM-I7-E002: gate I7 on the integrated mission (python-kernel SURROGATE evidence).

E001 (bandwidth sweep): the full mission (scenario I7-BANDWIDTH) at 100/50/10/1/0.1 % (+0 %) of the declared
mission link. E002 (outage + critical finding): scenario I7-OUTAGE-CRITICAL, where the link drops before the
robot's lane pass reaches a view of the (lane-side) defect, so the critical structural finding is made while
the link is down.

Every policy sees the SAME mission: the primary arm (BAAC, or the configured primary policy for closed-loop
checks) drives the mission; the other policies are shadow arms of ``ShoreLink`` that receive the identical
offer stream over the identical link profile and channel seed (conrad.orchestration.comms). Scoring uses the
communication oracle (conrad.evaluation.oracle.comm_oracle) against everything Model 1 asked to report.

Seeds come from the frozen partitions (conrad.evaluation.partitions): design runs use development seeds, the
reported runs use final_test seeds. All channel numbers are SYNTHETIC_ONLY.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from conrad.evaluation.oracle.comm_oracle import mission_information_retained
from conrad.evaluation.partitions import Partition, Purpose, check_access, partition_of
from conrad.orchestration.mission_config import runtime_config
from conrad.settings import REPO_ROOT, ConradSettings, load_settings
from conrad.sim.mission.run import prepare
from conrad.sim.mission.scenarios import resolve

DEFAULT_CONFIG = "configs/sim/mission_default.yaml"
EVIDENCE_CLASS = "SURROGATE (python L1 kernel mission, not Unity)"
POLICIES: dict[str, str] = {
    "baac": "C-B10_baac",
    "raw": "C-B0_send_all",
    "fifo": "C-B1_fifo",
    "fixed_priority": "C-B2_fixed_priority",
    "value_per_bit": "C-B4_value_per_bit",
}
SPEC_COMPARISON = ("raw", "fifo", "fixed_priority")  # ch25 I7 comparison set
NS = 1_000_000_000


# ---------------------------------------------------------------------------------------------- one mission
def _settings(seed: int, scenario: str, factor: float, primary: str, shadows: list[str]) -> ConradSettings:
    base = load_settings(REPO_ROOT / DEFAULT_CONFIG)
    mission = dict(base.sim.get("mission", {}))
    _, rt = resolve(scenario, mission)
    ref_bps = runtime_config(rt).link.bandwidth_bps  # the declared mission link
    runtime = dict(mission.get("runtime", {}))
    runtime["link"] = {**dict(runtime.get("link", {})), "bandwidth_bps": ref_bps * factor}
    runtime["baac"] = {
        **dict(runtime.get("baac", {})),
        "scheduler_policy": POLICIES[primary],
        "shadow_arms": [{"name": n, "policy": POLICIES[n]} for n in shadows],
    }
    sim = {**base.sim, "mission": {**mission, "runtime": runtime}}
    settings = base.model_copy(update={"sim": sim, "run": base.run.model_copy(update={"seed": int(seed)})})
    assert isinstance(settings, ConradSettings)
    return settings


def _linkup_s(t0: float, t1: float, outages: list[tuple[float, float]], bandwidth: float) -> float:
    """Seconds of available link between t0 and t1."""
    if bandwidth <= 0 or t1 <= t0:
        return 0.0
    down = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in outages)
    return (t1 - t0) - down


def _score_arm(
    rep: dict[str, Any], arm: Any, shore: Any, duration_s: float, cfg: dict[str, Any]
) -> dict[str, Any]:
    latest = {b: m for b, (m, _) in shore.intent_latest.items()}
    values = {b: v for b, (_, v) in shore.intent_latest.items()}
    retained_levels = shore.baac_config.information_retained
    retained = mission_information_retained(arm.receiver, latest, values, retained_levels)
    crit_ids = {b for b, _, _ in shore.critical_offers}
    crit_latest = {b: m for b, m in latest.items() if b in crit_ids}
    crit_retained = (
        mission_information_retained(arm.receiver, crit_latest, values, retained_levels)
        if crit_latest
        else None
    )
    bits = int(sum(t["bits"] for t in rep["transmissions"]))
    outages = [(float(a), float(b)) for a, b in shore.effective_outages_s]
    bw = arm.profile.bandwidth_bps

    # arrivals by belief: alert/delta receipts (t, revision)
    alerts: dict[str, list[tuple[int, int]]] = {}
    deltas: dict[str, list[tuple[int, int]]] = {}
    for r in rep["receipts"]:
        if r["kind"] == "alert":
            alerts.setdefault(r["belief_id"], []).append((r["t_ns"], r["revision"]))
        elif r["kind"] == "deltas":
            deltas.setdefault(r["belief_id"], []).append((r["t_ns"], r["revision"]))

    def first(d: dict[str, list[tuple[int, int]]], bid: str, rev: int) -> int | None:
        hits = [t for t, rr in d.get(bid, []) if rr >= rev]
        return min(hits) if hits else None

    crit = []
    for b, rev, t0 in shore.critical_offers:
        ta = [x for x in (first(alerts, str(b), rev), first(deltas, str(b), rev)) if x is not None]
        td = first(deltas, str(b), rev)
        crit.append(
            {
                "belief_id": str(b),
                "revision": rev,
                "finding_t_s": t0 / NS,
                "link_down_at_finding": shore.channel.bandwidth(shore.profile.name, t0 / NS) <= 0,
                "alert_latency_s": (min(ta) - t0) / NS if ta else None,
                "delta_latency_s": (td - t0) / NS if td is not None else None,
            }
        )

    # deadline success (deadlines in seconds of AVAILABLE link, ENGINEERING_ESTIMATE from config)
    dl = {"critical": [0, 0], "routine": [0, 0]}  # [met, evaluable]
    for b, rev, t0n, value in shore.offer_log:
        t0 = t0n / NS
        critical = value >= shore.baac_config.critical_value
        key = "critical" if critical else "routine"
        limit = float(cfg["deadline_linkup_s"][key])
        hits = [
            x
            for x in (first(deltas, str(b), rev), first(alerts, str(b), rev) if critical else None)
            if x is not None
        ]
        arrive = min(hits) / NS if hits else None
        if arrive is not None and _linkup_s(t0, arrive, outages, bw) <= limit:
            dl[key][0] += 1
            dl[key][1] += 1
        elif _linkup_s(t0, duration_s, outages, bw) >= limit:
            dl[key][1] += 1  # the deadline passed inside the mission and was missed

    # receiver synchronisation for critical beliefs
    sync = []
    for b in sorted(crit_ids, key=str):
        s_rev = rep["sender_latest_revisions"].get(str(b))
        r_rev = rep["receiver_revisions"].get(str(b))
        sync.append(
            {
                "belief_id": str(b),
                "sender_revision": s_rev,
                "intent_revision": latest[b].revision if b in latest else None,
                "receiver_revision": r_rev,
                "equal": s_rev is not None and r_rev == s_rev,
                "duplicates": len(rep["receiver_applied"].get(str(b), []))
                - len(set(rep["receiver_applied"].get(str(b), []))),
            }
        )
    qt = rep["queue_trace"]
    step = max(1, int(cfg.get("queue_trace_every_s", 10)))
    return {
        "policy": rep["policy"],
        "bandwidth_bps": bw,
        "mission_information_retained": retained,
        "critical_information_retained": crit_retained,
        "bits_sent": bits,
        "energy_j": rep["energy_j"],
        "retained_per_mbit": retained / (bits / 1e6) if bits else None,
        "receiver_beliefs": len(rep["receiver_revisions"]),
        "critical": crit,
        "critical_alert_latency_s": crit[0]["alert_latency_s"] if crit else None,
        "critical_delta_latency_s": crit[0]["delta_latency_s"] if crit else None,
        "deadline_success": {k: (v[0] / v[1] if v[1] else None) for k, v in dl.items()},
        "deadline_counts": {k: {"met": v[0], "evaluable": v[1]} for k, v in dl.items()},
        "queue_depth_max": max((q[1] for q in qt), default=0),
        "backlog_bits_max": max((q[2] for q in qt), default=0),
        "queue_depth_end": qt[-1][1] if qt else 0,
        "backlog_bits_end": qt[-1][2] if qt else 0,
        "queue_trace": [q for i, q in enumerate(qt) if i % step == 0],
        "sync_critical": sync,
        "sync_critical_all_equal": bool(sync) and all(s["equal"] for s in sync),
        "duplicate_contributions": rep["duplicate_contributions"],
        "stale_ignored": rep["stale_ignored"],
        "coalesced": rep["coalesced"],
        "drop_reasons": rep["drop_reasons"],
    }


def _reconnection_trace(rep: dict[str, Any], shore: Any, n: int = 12) -> list[dict[str, Any]]:
    """What happened at each reconnection: first sends/receipts and whether the critical belief went first."""
    crit = {str(b): r for b, r, _ in shore.critical_offers}
    out = []
    for a, b in shore.effective_outages_s:
        a_ns, b_ns = int(a * NS), int(b * NS)
        found = [c for c in shore.critical_offers if a_ns <= c[2] < b_ns]
        txs = [t for t in rep["transmissions"] if t["sent_ns"] >= b_ns][:n]
        recs = [r for r in rep["receipts"] if r["t_ns"] >= b_ns and r["kind"] in ("alert", "deltas")]
        first_crit_delta = next(
            (
                i
                for i, r in enumerate(recs)
                if r["kind"] == "deltas" and r["belief_id"] in crit and r["revision"] >= crit[r["belief_id"]]
            ),
            None,
        )
        before = recs if first_crit_delta is None else recs[: first_crit_delta + 1]
        out.append(
            {
                "outage_s": [a, b],
                "findings_during_outage": [[str(x), r, t / NS] for x, r, t in found],
                "first_sent_after_reconnect": txs,
                "first_receipts_after_reconnect": recs[:n],
                "critical_delta_index": first_crit_delta,
                "critical_delivered_first": bool(found)
                and first_crit_delta is not None
                and all(r["belief_id"] in crit for r in before),
                "queue_at_reconnect": next((q for q in rep["queue_trace"] if q[0] >= b), None),
            }
        )
    return out


def mission_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one integrated mission with a primary arm and shadow arms; return scored, JSON-safe results."""
    seed, scenario, factor = int(job["seed"]), str(job["scenario"]), float(job["factor"])
    primary, shadows = str(job["primary"]), list(job["shadows"])
    settings = _settings(seed, scenario, factor, primary, shadows)
    runs_root = Path(job["runs_root"])
    session = prepare(scenario, settings, run_id=job["run_id"], runs_root=runs_root)
    session.run()
    outcome = session.finish()
    shore = session.runtime.shore
    harness = shore.harness_report()
    duration = float(session.rcfg.duration_s)
    arms = {}
    reconnect = {}
    for arm_name, arm in shore.arms.items():
        policy_key = primary if arm_name == "primary" else arm_name
        rep = harness["arms"][arm_name]
        arms[policy_key] = _score_arm(rep, arm, shore, duration, job["cfg"])
        if shore.effective_outages_s:
            reconnect[policy_key] = _reconnection_trace(rep, shore)
    report = outcome["report"]
    offer_digest = hashlib.sha256(json.dumps(harness["offer_log"]).encode()).hexdigest()
    result = {
        "run_id": job["run_id"],
        "seed": seed,
        "scenario": scenario,
        "bandwidth_factor": factor,
        "primary": primary,
        "link": harness["link"],
        "patch_first_visible_t_s": report.get("patch_first_visible_t_s"),
        "critical_offers": harness["critical_offers"],
        "offers": len(harness["offer_log"]),
        "offer_stream_sha256": offer_digest,
        "beliefs_reported": len(harness["intent_latest"]),
        "arms": arms,
        "reconnection": reconnect,
    }
    trace_dir = Path(job["trace_dir"])
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"{job['run_id']}.json").write_text(json.dumps(harness, default=str), encoding="utf-8")
    if not job.get("keep_bundle", False):
        shutil.rmtree(Path(outcome["run_dir"]), ignore_errors=True)
    print(f"finished {job['run_id']}", flush=True)  # progress for long background runs
    return result


# ---------------------------------------------------------------------------------------------- experiment
def _check_seeds(seeds: list[int], partition: str, domain: str = "mission") -> None:
    """Every seed belongs to the declared partition of the declared domain (``partition_domain``).

    The reported I7 runs moved to the digest-pinned ``i7_mission_v2`` domain (configs/eval/partitions_i7_v2.yaml)
    after the 2026-09-20 BAAC repair made the COM-I7-E001/E002 evidence on 5300000-5300004 stale. Development
    runs keep using the ``mission`` development split.
    """
    part = Partition(partition)
    purpose = (
        Purpose.FINAL_EVALUATION if part in (Partition.FINAL_TEST, Partition.OOD_TEST) else Purpose.DESIGN
    )
    check_access(part, purpose)
    for s in seeds:
        got = partition_of(domain, s)
        if got is not part:
            raise ValueError(f"seed {s} is in partition {got} of domain {domain!r}, not {part.value}")


def _jobs(config: dict[str, Any], seeds: list[int], out: Path) -> list[dict[str, Any]]:
    eid = str(config["experiment_id"])
    scenario = str(config["scenario"])
    levels = [float(x) for x in config["bandwidth_levels"]]
    all_arms = list(POLICIES)
    runs_root = REPO_ROOT / "artifacts" / "runs" / eid
    jobs = []
    for seed in seeds:
        for f in levels:
            jobs.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "factor": f,
                    "primary": "baac",
                    "shadows": [a for a in all_arms if a != "baac"],
                    "run_id": f"{eid}-s{seed}-bw{f:g}-shadow",
                    "runs_root": str(runs_root),
                    "trace_dir": str(out / "traces"),
                    "cfg": config,
                    "keep_bundle": bool(config.get("keep_bundles", False)),
                }
            )
        for f in [float(x) for x in config.get("closed_loop_levels", [])]:
            for p in all_arms:
                if p == "baac":
                    continue
                jobs.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "factor": f,
                        "primary": p,
                        "shadows": [],
                        "run_id": f"{eid}-s{seed}-bw{f:g}-closed-{p}",
                        "runs_root": str(runs_root),
                        "trace_dir": str(out / "traces"),
                        "cfg": config,
                        "keep_bundle": False,
                    }
                )
    return jobs


def _mean(xs: list[Any]) -> float | None:
    v = [float(x) for x in xs if x is not None]
    return float(np.mean(v)) if v else None


SCALARS = (
    "mission_information_retained",
    "critical_information_retained",
    "bits_sent",
    "retained_per_mbit",
    "critical_alert_latency_s",
    "critical_delta_latency_s",
    "queue_depth_max",
    "backlog_bits_max",
    "queue_depth_end",
    "backlog_bits_end",
    "duplicate_contributions",
    "stale_ignored",
    "coalesced",
    "energy_j",
)


def summarize(rows: list[dict[str, Any]], levels: list[float]) -> dict[str, Any]:
    shadow = [r for r in rows if r["run_id"].endswith("-shadow")]
    summary: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for f in levels:
        cond = f"bw_{f * 100:g}pct"
        at = [r for r in shadow if r["bandwidth_factor"] == f]
        summary[cond] = {}
        for p in POLICIES:
            arms = [r["arms"][p] for r in at if p in r["arms"]]
            s: dict[str, Any] = {k: _mean([a[k] for a in arms]) for k in SCALARS}
            s["deadline_success_critical"] = _mean([a["deadline_success"]["critical"] for a in arms])
            s["deadline_success_routine"] = _mean([a["deadline_success"]["routine"] for a in arms])
            s["sync_critical_all_equal_fraction"] = _mean([float(a["sync_critical_all_equal"]) for a in arms])
            s["critical_delivered_fraction"] = _mean(
                [float(a["critical_alert_latency_s"] is not None) for a in arms if a["critical"]]
            )
            s["n_seeds"] = len(arms)
            summary[cond][p] = s
        comparisons[cond] = {}
        for p in [x for x in POLICIES if x != "baac"]:
            diffs = [
                r["arms"]["baac"]["mission_information_retained"]
                - r["arms"][p]["mission_information_retained"]
                for r in at
            ]
            comparisons[cond][p] = {
                "baac_minus_policy_mean": _mean(diffs),
                "baac_minus_policy_min": min(diffs) if diffs else None,
                "baac_strictly_more_every_seed": bool(diffs) and all(d > 0 for d in diffs),
                "baac_strictly_more_mean": bool(diffs) and float(np.mean(diffs)) > 0,
            }
    return {"summary": summary, "comparisons": comparisons}


def run(config: dict[str, Any], seeds: list[int], out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    partition = str(config.get("partition", "final_test"))
    domain = str(config.get("partition_domain", "mission"))
    _check_seeds(seeds, partition, domain)
    jobs = _jobs(config, seeds, out)
    workers = int(config.get("workers", 1))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(mission_job, jobs))
    else:
        rows = [mission_job(j) for j in jobs]
    rows.sort(key=lambda r: r["run_id"])
    levels = [float(x) for x in config["bandwidth_levels"]]
    result: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "evidence_class": EVIDENCE_CLASS,
        "data_status": "SYNTHETIC_ONLY",
        "partition": partition,
        "partition_domain": domain,
        "seeds": seeds,
        "scenario": config["scenario"],
        "policies": POLICIES,
        "spec_comparison_set": list(SPEC_COMPARISON),
        "config": config,
        **summarize(rows, levels),
        "per_run": rows,
    }
    closed = [r for r in rows if "-closed-" in r["run_id"]]
    if closed:
        shadow_by = {(r["seed"], r["bandwidth_factor"]): r for r in rows if r["run_id"].endswith("-shadow")}
        result["closed_loop"] = [
            {
                "seed": r["seed"],
                "bandwidth_factor": r["bandwidth_factor"],
                "primary": r["primary"],
                "arm": r["arms"][r["primary"]],
                "same_offer_stream_as_baac_primary": r["offer_stream_sha256"]
                == shadow_by[(r["seed"], r["bandwidth_factor"])]["offer_stream_sha256"],
                "critical_offers": r["critical_offers"],
            }
            for r in closed
        ]
    name = str(config["experiment_id"]).lower().replace("-", "_") + ".json"
    (out / name).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result
