"""Deterministic NAV-001..008 runner: SimRobotHardware + NavigationStack + CommandGateway.

Truth is read ONLY here, via TruthAccess, for metrics. The stack sees RHI readings only.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from conrad.evaluation.nav_benchmarks.scenarios import (
    SyntheticPositionFixRenderer,
    _segment_distance,
    geometry_of,
    load_scenario,
)
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.navigation import GoalStatus, NavigationStack, NavigationStackConfig
from conrad.runtime.command_gateway import CommandGateway
from conrad.schemas.decision import NavigationGoal
from conrad.schemas.frames import WORLD, Pose, SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import RobotConfig
from conrad.settings import CommandMode, ExecutionLane, RuntimeSettings
from conrad.sim.kernel import FaultType, SimKernelConfig, build_sim_hardware

ROBOT_CONFIG = "configs/robot/sim_reference.yaml"


def _goal(scn: dict[str, Any], ids: IdFactory) -> tuple[NavigationGoal, np.ndarray]:
    g = dict(scn["goal"])
    prim = g.pop("primitive")
    tol = float(g.pop("tolerance_m"))
    constraints: dict[str, Any] = {"primitive": prim}
    target_pose = None
    target_region = None
    if prim == "PIPELINE_FOLLOW":
        constraints.update(g)
        final = np.asarray(g["polyline"][-1], dtype=np.float64) + np.array([0.0, 0.0, g["standoff_m"]])
        poly = np.asarray(g["polyline"], dtype=np.float64)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        c, h = (lo + hi) / 2, (hi - lo) / 2
        target_region = SpatialSupport(
            frame_id=WORLD,
            center_m=(float(c[0]), float(c[1]), float(c[2])),
            half_extent_m=(float(h[0]), float(h[1]), float(h[2])),
        )
    else:
        final = np.asarray(g.pop("target"), dtype=np.float64)
        target_pose = Pose(frame_id=WORLD, position_m=(float(final[0]), float(final[1]), float(final[2])))
        constraints.update(g)
    goal = NavigationGoal(
        goal_id=ids.new(),
        trace_id=ids.new(),
        target_pose=target_pose,
        target_region=target_region,
        position_tolerance_m=tol,
        orientation_tolerance_rad=0.2,
        observation_constraints=constraints,
        risk_limit=0.1,
    )
    return goal, final


def run_benchmark(
    benchmark_id: str,
    seed: int,
    out_dir: str | Path | None = None,
    robot_config: RobotConfig | None = None,
    stack_config: NavigationStackConfig | None = None,
) -> dict[str, Any]:
    scn = load_scenario(benchmark_id)
    cfg = robot_config or load_robot_config(ROBOT_CONFIG)
    ids = IdFactory(seed=seed)
    mission_id, run_id = ids.new(), ids.new()
    geo = geometry_of(scn)
    current = np.asarray(scn["current_mps"], dtype=np.float64)
    fixes = scn["fixes"]
    renderer = SyntheticPositionFixRenderer(
        mission_id,
        run_id,
        ids,
        np.random.default_rng([seed, 17]),
        fixes["period_s"],
        fixes["sigma_m"],
        fixes["outages"],
    )
    hw = build_sim_hardware(
        cfg,
        seed,
        SimKernelConfig(),
        current_field=lambda p, t: current,
        sdf=None if geo.empty else geo.sdf,
        renderer=renderer,
    )
    start = np.asarray(scn["start"], dtype=np.float64)
    hw.kernel.reset(start)
    truth = hw.truth_access()
    dt = float(scn["control_period_s"])
    inflation = float(scn.get("planner_inflation_m", 0.5))
    stack = NavigationStack(
        hw,
        cfg,
        ids,
        mission_id,
        run_id,
        Pose(frame_id=WORLD, position_m=(float(start[0]), float(start[1]), float(start[2]))),  # launch point
        config=stack_config or NavigationStackConfig(control_period_s=dt),
        is_free=None if geo.empty else (lambda p: geo.sdf(p) > inflation),
        local_distance=None if geo.empty else geo.sdf,
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
    goal, final = _goal(scn, ids)
    hw.advance(0.1)  # let the first IMU/depth samples arrive
    traj = stack.set_goal(goal)
    faults = sorted(scn["faults"], key=lambda f: f["t_s"])
    route = np.array([p.pose.position_m for p in traj.points]) if traj else final[None, :]
    log_t, log_true, log_est, states = [], [], [], Counter[str]()
    refused, gw_reasons = 0, Counter[str]()
    arrived_s: float | None = None
    steps = round(float(scn["duration_s"]) / dt)
    for _ in range(steps):
        t = hw.kernel.t_s
        while faults and faults[0]["t_s"] <= t:
            f = faults.pop(0)
            vec = tuple(f["vector"]) if "vector" in f else None
            hw.inject_fault(
                FaultType(f["type"]),
                f.get("target"),
                float(f.get("magnitude", 0.0)),
                f.get("duration_s"),
                vec,
            )
        res = stack.step()
        states[res.assessment.state.value] += 1
        if res.decision.authorized:
            ack = gateway.submit(res.command)
            gw_reasons.update(ack.reason_codes)
        else:
            refused += 1
        hw.advance(dt)
        if arrived_s is None and stack.status in (GoalStatus.ARRIVED, GoalStatus.COMPLETE):
            arrived_s = hw.kernel.t_s
        log_t.append(hw.kernel.t_s)
        log_true.append(truth.true_state().position_world_m)
        log_est.append(np.asarray(stack.estimator.get_state().pose.position_m))
    tt, est = np.array(log_t), np.array(log_est)
    tr = np.array(log_true)
    metrics = _metrics(scn, tt, tr, est, route, final, geo)
    success_cfg = scn["success"]
    checks = {
        "goal_accepted": traj is not None,
        "final_position": metrics["final_position_error_m"] <= success_cfg["final_tolerance_m"],
        "collisions": truth.collision_count <= success_cfg["max_collisions"],
    }
    if "standoff_rms_max_m" in success_cfg:
        checks["standoff"] = metrics["standoff_rms_error_m"] <= success_cfg["standoff_rms_max_m"]
    if "station_rms_max_m" in success_cfg:
        checks["station_keeping"] = metrics["station_rms_error_m"] <= success_cfg["station_rms_max_m"]
    safety_events = [
        {"t_s": round(ns / 1e9, 3), "state": st.value, "reasons": list(r)}
        for ns, st, r in stack.supervisor.history
    ]
    result = {
        "benchmark_id": benchmark_id,
        "description": scn["description"],
        "seed": seed,
        "simulation_validity_level": truth.validity_level.value,
        "success": all(checks.values()),
        "checks": checks,
        **metrics,
        "time_to_goal_s": arrived_s,
        "goal_status": stack.status.value,
        "energy_j": round(truth.energy_used_j, 1),
        "collisions": truth.collision_count,
        "min_clearance_m": None if geo.empty else round(truth.min_clearance_m, 3),
        "safety_state_time_s": {k: round(v * dt, 2) for k, v in states.items()},
        "safety_events": safety_events[:50],
        "safety_event_count": len(safety_events),
        "commands_refused_by_supervisor": refused,
        "gateway_accepted": gateway.accepted,
        "gateway_rejected": gateway.rejected,
        "gateway_rejection_reasons": dict(gw_reasons),
        "watchdog_trips": hw.watchdog_trips,
        "faults_injected": [f.fault_type.value for f in hw.fault_log],
        "position_fixes_delivered": renderer.delivered,
        "trace_records": len(stack.records),
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{benchmark_id}_seed{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _metrics(
    scn: dict[str, Any],
    t: np.ndarray,
    tr: np.ndarray,
    est: np.ndarray,
    route: np.ndarray,
    final: np.ndarray,
    geo: Any,
) -> dict[str, Any]:
    path_err = _segment_distance(tr, route) if len(route) > 1 else np.linalg.norm(tr - route[0], axis=1)
    est_err = np.linalg.norm(est - tr, axis=1)
    out: dict[str, Any] = {
        "final_position_error_m": round(float(np.linalg.norm(tr[-1] - final)), 4),
        "path_rms_error_m": round(float(np.sqrt(np.mean(path_err**2))), 4),
        "path_max_error_m": round(float(path_err.max()), 4),
        "path_length_m": round(float(np.sum(np.linalg.norm(np.diff(tr, axis=0), axis=1))), 3),
        "estimation_rms_error_m": round(float(np.sqrt(np.mean(est_err**2))), 4),
        "estimation_max_error_m": round(float(est_err.max()), 4),
    }
    tol = float(scn["goal"]["tolerance_m"])
    inside = np.linalg.norm(tr - final, axis=1) <= tol
    outside_idx = np.nonzero(~inside)[0]
    last_out = outside_idx[-1] if len(outside_idx) else -1
    out["truth_settled_time_s"] = round(float(t[last_out + 1]), 2) if last_out + 1 < len(t) else None
    goal = scn["goal"]
    if goal["primitive"] == "PIPELINE_FOLLOW" and geo.pipeline is not None:
        d = _segment_distance(tr, geo.pipeline)
        offset = float(goal["standoff_m"]) * np.asarray(goal.get("standoff_direction", [0.0, 0.0, 1.0]))
        offset_path = geo.pipeline + offset[None, :]
        reached = np.nonzero(np.linalg.norm(tr - offset_path[0], axis=1) < 0.5)[0]
        begin = int(reached[0]) if len(reached) else 0
        on = np.arange(len(tr)) >= begin
        on &= (tr[:, 0] > geo.pipeline[0, 0] + 0.5) & (tr[:, 0] < geo.pipeline[-1, 0] - 0.5)
        err = np.abs(d[on] - float(goal["standoff_m"]))
        out["standoff_rms_error_m"] = round(float(np.sqrt(np.mean(err**2))), 4) if err.size else None
        out["standoff_max_error_m"] = round(float(err.max()), 4) if err.size else None
        xt = _segment_distance(tr[on], offset_path)
        out["pipeline_cross_track_rms_m"] = round(float(np.sqrt(np.mean(xt**2))), 4) if xt.size else None
        out["pipeline_cross_track_max_m"] = round(float(xt.max()), 4) if xt.size else None
    if goal["primitive"] == "STATION_KEEP":
        window = t >= 5.0
        err = np.linalg.norm(tr[window] - final, axis=1)
        out["station_rms_error_m"] = round(float(np.sqrt(np.mean(err**2))), 4)
        out["station_max_error_m"] = round(float(err.max()), 4)
    return out
