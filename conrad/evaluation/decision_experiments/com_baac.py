"""COM-BAAC-E001: bandwidth collapse (100/50/10/1/0.1/0 %) and outage/reconnection, BAAC vs baselines.

SYNTHETIC_ONLY belief stream and channel. Metric: mission-relevant information retained at the
receiver at mission end, critical-alert latency, receiver sync error, bits and energy used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from conrad.communication import (
    BAAC_POLICY,
    BASELINE_POLICIES,
    BAACConfig,
    BAACSender,
    ChannelSim,
    LinkProfile,
    ReceiverStore,
    SchedulingPolicy,
)
from conrad.evaluation.decision_experiments.fixtures import CLOCK, make_belief, unc
from conrad.evaluation.oracle.comm_oracle import mission_information_retained, sync_error
from conrad.schemas.belief import BeliefMessage
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import NS_PER_S, stamp
from conrad.schemas.world import Domain

EXPERIMENT_ID = "COM-BAAC-E001"
LEVELS = (1.0, 0.5, 0.1, 0.01, 0.001, 0.0)


def make_stream(
    seed: int, duration_s: float, n_routine_tech: int, n_eco: int
) -> list[tuple[float, BeliefMessage, float, dict[UUID, int]]]:
    """(time, message, mission_value, evidence byte sizes). Critical target changes at fixed times."""
    rng = np.random.default_rng(seed)
    ids = IdFactory(seed + 7)
    specs = [("critical", Domain.TECHNICAL, 0.95)]
    specs += [("tech", Domain.TECHNICAL, 0.5)] * n_routine_tech
    specs += [("eco", Domain.ECOLOGICAL, 0.2)] * n_eco
    bids = [ids.new() for _ in specs]
    rev = [0] * len(specs)
    out = []
    critical_times = {round(duration_s * f) for f in (0.1, 0.4, 0.7)}
    for t in np.arange(0.0, duration_s, 10.0):
        for i, (kind, domain, value) in enumerate(specs):
            first = t == 0.0
            if kind == "critical":
                fire = first or round(t) in critical_times
            else:
                fire = first or rng.random() < 0.5
            if not fire:
                continue
            u = (
                unc(uc=0.7, uo=0.2)
                if kind == "critical" and not first
                else unc(ua=float(rng.uniform(0.05, 0.6)))
            )
            m = make_belief(
                ids,
                domain=domain,
                belief_id=bids[i],
                revision=rev[i],
                time_s=float(t),
                uncertainty=u,
                properties={
                    "condition": "CHANGED" if kind == "critical" and not first else "NOMINAL",
                    "level": float(rng.uniform()),
                },
                severity=0.9 if kind == "critical" and not first else None,
                n_evidence=2,
            )
            rev[i] += 1
            sizes = {e: int(rng.integers(100_000, 300_000)) for e in m.evidence_support}
            out.append((float(t), m, value, sizes))
    return out


def run_arm(
    stream: list[Any],
    policy: SchedulingPolicy,
    factor: float,
    outage: tuple[float, float] | None,
    duration_s: float,
    cfg: BAACConfig,
    link: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    profile = LinkProfile(
        name="acoustic",
        bandwidth_schedule=((0.0, factor),),
        outages_s=() if outage is None else (outage,),
        **link,
    )
    channel = ChannelSim([profile], seed=seed)
    sender = BAACSender(IdFactory(seed + 11), channel, cfg, policy)
    store = ReceiverStore()
    arrivals: dict[tuple[UUID, int], int] = {}

    def sink(incr: dict[str, Any]) -> Any:
        if incr.get("kind") == "alert":
            key = (UUID(str(incr["belief_id"])), int(incr["revision"]))
            arrivals.setdefault(key, sender.current_arrival_ns)
        elif incr.get("kind") == "deltas" and incr["deltas"]:
            d = incr["deltas"][0]
            arrivals.setdefault(
                (UUID(str(d["belief_id"])), int(d["new_revision"])), sender.current_arrival_ns
            )
        return store.receive(incr)

    latest: dict[UUID, BeliefMessage] = {}
    values: dict[UUID, float] = {}
    critical_events: list[tuple[UUID, int, float]] = []
    i = 0
    for t in np.arange(0.0, duration_s, 1.0):
        while i < len(stream) and stream[i][0] <= t:
            _, m, v, sizes = stream[i]
            latest[m.belief_id], values[m.belief_id] = m, v
            if v >= cfg.critical_value and m.revision > 0:
                critical_events.append((m.belief_id, m.revision, float(t)))
            sender.offer(m, v, stamp(float(t), CLOCK), sizes)
            i += 1
        sender.step(float(t), 1.0, sink)
    lat = []
    for bid, r, t0 in critical_events:
        hits = [ns for (b, rr), ns in arrivals.items() if b == bid and rr >= r]
        if hits:
            lat.append(min(hits) / NS_PER_S - t0)
    bits = sum(tx.bits for tx in sender.transmissions)
    retained = mission_information_retained(store, latest, values, cfg.information_retained)
    return {
        "mission_information_retained": retained,
        "critical_alerts_delivered_fraction": len(lat) / len(critical_events) if critical_events else None,
        "critical_alert_latency_mean_s": float(np.mean(lat)) if lat else None,
        "critical_alert_latency_max_s": float(np.max(lat)) if lat else None,
        "sync_error": sync_error(store, latest),
        "bits_sent": bits,
        "energy_j": sender.energy_used_j,
        "retained_per_mbit": retained / (bits / 1e6) if bits else None,
        "dropped_units": len(sender.queue.dropped),
        "dropped_critical": sum(d.critical for d in sender.queue.dropped),
        "queue_depth_end": len(sender.queue),
        "resync_requests": len(store.resync_requests),
    }


def run(config: dict[str, Any], seeds: list[int], out_dir: str | Path) -> dict[str, Any]:
    duration = float(config.get("duration_s", 300.0))
    link = dict(
        config.get(
            "link",
            {
                "bandwidth_bps": 20000.0,
                "latency_s": 1.0,
                "packet_loss": 0.05,
                "bit_error_rate": 1e-6,
                "energy_per_bit_j": 1e-4,
            },
        )
    )
    cfg = BAACConfig(**config.get("baac", {}))
    outage = tuple(config.get("outage_s", (60.0, 180.0)))
    policies = {BAAC_POLICY.name: BAAC_POLICY, **BASELINE_POLICIES}
    conditions: dict[str, tuple[float, tuple[float, float] | None]] = {
        f"bw_{lv * 100:g}pct": (lv, None) for lv in LEVELS
    }
    conditions["outage_reconnect_100pct"] = (1.0, (float(outage[0]), float(outage[1])))
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        stream = make_stream(
            seed, duration, int(config.get("n_routine_tech", 4)), int(config.get("n_eco", 15))
        )
        per_seed[str(seed)] = {
            cond: {
                name: run_arm(stream, pol, lv, out, duration, cfg, link, seed)
                for name, pol in policies.items()
            }
            for cond, (lv, out) in conditions.items()
        }
    summary: dict[str, Any] = {}
    for cond in conditions:
        summary[cond] = {}
        for name in policies:
            rows = [per_seed[s][cond][name] for s in per_seed]
            summary[cond][name] = {
                k: (
                    float(np.mean([r[k] for r in rows if r[k] is not None]))
                    if any(r[k] is not None for r in rows)
                    else None
                )
                for k in rows[0]
            }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "data_status": "SYNTHETIC_ONLY",
        "config": config,
        "seeds": seeds,
        "summary": summary,
        "per_seed": per_seed,
    }
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "com_baac_e001.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
