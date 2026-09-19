"""NAV-CAL-E001: estimator covariance calibration through position-fix outages (evaluation side).

For each (variant, outage duration, IMU noise multiplier, seed) a closed-loop NavigationStack transit runs
on the L1 kernel. Fixes stop at ``outage_start_s`` for the outage duration and the IMU noise fault is
injected at the same instant. Truth is read ONLY here (TruthAccess) to score the estimator.

Metrics over the outage window (sampled every ``sample_period_s``):
NEES (3-D position, full covariance), per-axis 1/2/3-sigma containment vs 68.27/95.45/99.73 %,
max ||e|| / sqrt(trace P), time to LOCALIZATION_LOST vs time the true error first exceeded the danger
threshold, the "missed danger" duration (true error above the threshold while the estimator did not
report LOCALIZATION_LOST), and point error.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from conrad.evaluation.nav_benchmarks.runner import ROBOT_CONFIG, _goal
from conrad.evaluation.nav_benchmarks.scenarios import SyntheticPositionFixRenderer
from conrad.robotics.estimation import EkfConfig
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.navigation import NavigationStack, NavigationStackConfig
from conrad.runtime.command_gateway import CommandGateway
from conrad.schemas.frames import WORLD, Pose
from conrad.schemas.ids import IdFactory
from conrad.settings import CommandMode, ExecutionLane, RuntimeSettings
from conrad.sim.kernel import FaultType, SimKernelConfig, build_sim_hardware

EXPERIMENT_ID = "NAV-CAL-E001"
NOMINAL_CONTAINMENT = {"1sigma": 0.6827, "2sigma": 0.9545, "3sigma": 0.9973}
VARIANTS = {"before_fix": EkfConfig.legacy_baseline, "after_fix": EkfConfig}


def _environment(seed: int, scn: dict[str, Any], outage_s: float) -> dict[str, Any]:
    """Per-seed synthetic current and gust (evaluation-side; never given to the estimator)."""
    rng = np.random.default_rng([seed, 7031])
    ang, mag = rng.uniform(0, 2 * math.pi), rng.uniform(0, float(scn["current_speed_max_mps"]))
    g_ang, g_mag = rng.uniform(0, 2 * math.pi), rng.uniform(0, float(scn["gust_speed_max_mps"]))
    g_t = float(scn["outage_start_s"]) + rng.uniform(0, max(outage_s, 1e-6))
    return {
        "current_mps": [mag * math.cos(ang), mag * math.sin(ang), 0.0],
        "gust": {"t_s": g_t, "vector": [g_mag * math.cos(g_ang), g_mag * math.sin(g_ang), 0.0]},
    }


def run_case(
    variant: str, outage_s: float, noise: float, seed: int, scn: dict[str, Any], sample_period_s: float
) -> dict[str, Any]:
    cfg = load_robot_config(ROBOT_CONFIG)
    env = _environment(seed, scn, outage_s)
    ids = IdFactory(seed=seed)
    mission_id, run_id = ids.new(), ids.new()
    t0 = float(scn["outage_start_s"])
    renderer = SyntheticPositionFixRenderer(
        mission_id,
        run_id,
        ids,
        np.random.default_rng([seed, 17]),
        float(scn["fixes"]["period_s"]),
        float(scn["fixes"]["sigma_m"]),
        [[t0, t0 + outage_s]],
    )
    current = np.asarray(env["current_mps"], dtype=np.float64)
    hw = build_sim_hardware(
        cfg, seed, SimKernelConfig(), current_field=lambda p, t: current, renderer=renderer
    )
    start = np.asarray(scn["start"], dtype=np.float64)
    hw.kernel.reset(start)
    truth = hw.truth_access()
    dt = float(scn["control_period_s"])
    stack_cfg = NavigationStackConfig(control_period_s=dt, estimator=VARIANTS[variant]())
    stack = NavigationStack(
        hw,
        cfg,
        ids,
        mission_id,
        run_id,
        Pose(frame_id=WORLD, position_m=tuple(float(x) for x in start)),
        config=stack_cfg,
    )
    hw.set_estimated_pose_provider(lambda: stack.estimator.get_state().pose)
    gateway = CommandGateway(
        hw,
        cfg,
        RuntimeSettings(command_mode=CommandMode.SIMULATED),
        ExecutionLane.SIMULATION,
        mission_id,
        run_id,
        state_age_s=stack.state_age_s,
    )
    goal_scn = {"goal": {"primitive": "GO_TO", "target": list(scn["goal_target"]), "tolerance_m": 0.3}}
    goal, _ = _goal(goal_scn, ids)
    hw.advance(0.1)
    stack.set_goal(goal)
    t_end = t0 + outage_s + float(scn["post_outage_s"])
    noise_done = gust_done = False
    every = max(1, round(sample_period_s / dt))
    rows: list[tuple[float, np.ndarray, np.ndarray, bool, str]] = []
    k = 0
    while hw.kernel.t_s < t_end:
        t = hw.kernel.t_s
        if not noise_done and t >= t0:
            hw.inject_fault(FaultType.SENSOR_NOISE, "imu", float(noise))
            noise_done = True
        if not gust_done and t >= env["gust"]["t_s"]:
            hw.inject_fault(
                FaultType.CURRENT_GUST, None, 0.0, float(scn["gust_duration_s"]), tuple(env["gust"]["vector"])
            )
            gust_done = True
        res = stack.step()
        if res.decision.authorized:
            gateway.submit(res.command)
        hw.advance(dt)
        k += 1
        if k % every == 0:
            est = stack.estimator
            e = np.asarray(est.get_state().pose.position_m) - truth.true_state().position_world_m
            rows.append(
                (hw.kernel.t_s, e, est.position_covariance, est.localization_lost, res.assessment.state.value)
            )
    return {
        "variant": variant,
        "outage_s": outage_s,
        "noise": noise,
        "seed": seed,
        "env": env,
        **_case_metrics(rows, t0, t0 + outage_s, cfg, stack),
    }


def _case_metrics(
    rows: list[tuple[float, np.ndarray, np.ndarray, bool, str]],
    t_start: float,
    t_stop: float,
    cfg: Any,
    stack: NavigationStack,
) -> dict[str, Any]:
    danger = stack.config.safety.sigma_k * float(
        cfg.safety.max_pose_sigma_m.require("safety.max_pose_sigma_m")
    )
    win = [r for r in rows if t_start <= r[0] <= t_stop]
    nees, nees_h, z, ratio, err = [], [], [], [], []
    t_lost: float | None = None
    t_danger: float | None = None
    t_hold: float | None = None
    missed = 0.0
    period = (win[1][0] - win[0][0]) if len(win) > 1 else 0.0
    for t, e, p, lost, sstate in win:
        nees.append(float(e @ np.linalg.solve(p, e)))
        nees_h.append(float(e[:2] @ np.linalg.solve(p[:2, :2], e[:2])))
        z.extend(np.abs(e[:2]) / np.sqrt(np.diag(p)[:2]))  # horizontal axes: the part fixes alone observe
        n = float(np.linalg.norm(e))
        err.append(n)
        ratio.append(n / math.sqrt(float(np.trace(p))))
        if lost and t_lost is None:
            t_lost = t - t_start
        if sstate in ("HOLD", "RETURN") and t_hold is None:
            t_hold = t - t_start
        if n > danger:
            t_danger = t - t_start if t_danger is None else t_danger
            if not lost:
                missed += period
    za = np.asarray(z)
    return {
        "danger_threshold_m": danger,
        "samples": len(win),
        "nees_mean": float(np.mean(nees)),
        "nees_per_dof_mean": float(np.mean(nees) / 3.0),
        "nees_horizontal_per_dof_mean": float(np.mean(nees_h) / 2.0),
        "within_1sigma": float(np.mean(za <= 1.0)),
        "within_2sigma": float(np.mean(za <= 2.0)),
        "within_3sigma": float(np.mean(za <= 3.0)),
        "max_error_to_sigma": float(np.max(ratio)),
        "time_to_localization_lost_s": t_lost,
        "time_to_hold_s": t_hold,
        "time_error_exceeds_danger_s": t_danger,
        "missed_danger_s": round(missed, 3),
        "mean_error_m": float(np.mean(err)),
        "max_error_m": float(np.max(err)),
        "final_outage_error_m": float(err[-1]),
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return float(np.mean([c[key] for c in cases]))

    lost = [c["time_to_localization_lost_s"] for c in cases]
    dang = [c["time_error_exceeds_danger_s"] for c in cases]
    return {
        "n_seeds": len(cases),
        "nees_per_dof_mean": mean("nees_per_dof_mean"),
        "nees_horizontal_per_dof_mean": mean("nees_horizontal_per_dof_mean"),
        "within_1sigma": mean("within_1sigma"),
        "within_2sigma": mean("within_2sigma"),
        "within_3sigma": mean("within_3sigma"),
        "max_error_to_sigma": float(np.max([c["max_error_to_sigma"] for c in cases])),
        "mean_error_m": mean("mean_error_m"),
        "max_error_m": float(np.max([c["max_error_m"] for c in cases])),
        "lost_declared_fraction": float(np.mean([x is not None for x in lost])),
        "time_to_lost_mean_s": float(np.mean([x for x in lost if x is not None]))
        if any(x is not None for x in lost)
        else None,
        "danger_reached_fraction": float(np.mean([x is not None for x in dang])),
        "missed_danger_mean_s": mean("missed_danger_s"),
        "missed_danger_max_s": float(np.max([c["missed_danger_s"] for c in cases])),
    }


def run(config: dict[str, Any], seeds: list[int], out_dir: str | Path) -> dict[str, Any]:
    dev = [int(s) for s in config.get("dev_seeds", [])]
    final = [int(s) for s in seeds]
    if set(dev) & set(final):
        raise ValueError(f"dev and held-out seed partitions overlap: {sorted(set(dev) & set(final))}")
    scn = config["scenario"]
    sample = float(config["metrics"]["sample_period_s"])
    partitions = {"final": final, "dev": dev}
    jobs = [
        (v, float(d), float(n), s, part)
        for part in config.get("partitions", ["final"])
        for v in config.get("variants", list(VARIANTS))
        for d in config["outage_durations_s"]
        for n in config["imu_noise_multipliers"]
        for s in partitions[part]
    ]
    workers = int(config.get("workers", 1))
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(run_case, v, d, n, s, scn, sample) for v, d, n, s, _ in jobs]
            results = [f.result() for f in futs]
    else:
        results = [run_case(v, d, n, s, scn, sample) for v, d, n, s, _ in jobs]
    for r, job in zip(results, jobs, strict=True):
        r["partition"] = job[4]
    summary: dict[str, Any] = {}
    for part in config.get("partitions", ["final"]):
        for v in config.get("variants", list(VARIANTS)):
            for d in config["outage_durations_s"]:
                for n in config["imu_noise_multipliers"]:
                    cs = [
                        r
                        for r in results
                        if (r["partition"], r["variant"], r["outage_s"], r["noise"])
                        == (part, v, float(d), float(n))
                    ]
                    summary[f"{part}/{v}/outage{int(d)}s/imu{n:g}x"] = _aggregate(cs)
            allv = [r for r in results if r["partition"] == part and r["variant"] == v]
            summary[f"{part}/{v}/ALL"] = _aggregate(allv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "experiment_id": EXPERIMENT_ID,
        "label": "SYNTHETIC_ONLY (L1 Python kernel)",
        "seed_partitions": {"dev": dev, "final_held_out": final},
        "nominal_containment": NOMINAL_CONTAINMENT,
        "summary": summary,
    }
    (out / "summary.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    (out / "cases.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return doc
