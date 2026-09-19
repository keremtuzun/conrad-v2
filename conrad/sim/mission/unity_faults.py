"""Gate I2 "faults reach defined safe states" on Unity (ch26 Phase 6 pass criterion). TRUTH / EVALUATION SIDE.

Each case of ``configs/sim/nav_fault_cases.yaml`` runs the NAV-001 transit through ``run_unity_nav`` with ONE fault
injected on the hardware-adapter side (a Unity bridge fault, or an outage of the synthetic USBL-like fix), then
checks the SafetySupervisor's recorded response against the case's declared expectation:

* the expected safety state and reason code are reached within ``deadline_s`` of the fault and held to the end;
* nothing worse than DEGRADED happened before the fault;
* the motion rule of the state holds after it is reached (all-zero thrust, no motion authorized, a fixed hold
  point, or no command to the faulted thruster);
* no collision.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from conrad.robotics.safety.monitors import SEVERITY, SafetyState
from conrad.settings import REPO_ROOT
from conrad.sim.mission.unity_run import NavFault, run_unity_nav

FAULT_CASES_PATH = REPO_ROOT / "configs" / "sim" / "nav_fault_cases.yaml"


def fault_cases(path: Path = FAULT_CASES_PATH) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def run_fault_case(case_id: str, seed: int, bundle_dir: str | Path) -> dict[str, Any]:
    """Run one fault case on Unity and return the run result plus ``safe_state`` (checks and measurements)."""
    cfg = fault_cases()
    case = cfg["cases"][case_id]
    t_fault = float(cfg["fault_t_s"])
    f = case["fault"]
    fault = NavFault(str(f["kind"]), t_fault, f.get("target"), float(f.get("magnitude", 1.0)))
    result = run_unity_nav(
        str(cfg["base_benchmark"]),
        seed,
        bundle_dir,
        faults=(fault,),
        duration_s=float(cfg["duration_s"]),
        battery_capacity_j=case.get("battery_capacity_j"),
    )
    out = Path(bundle_dir)
    log = json.loads((out / "safety.json").read_text(encoding="utf-8"))
    traj = json.loads((out / "trajectory.json").read_text(encoding="utf-8"))
    safe = check_safe_state(log, traj, case["expect"], t_fault, fault.target, int(result["collisions"]))
    result["safe_state"] = {"case": case_id, "fault": fault.as_json(), "expect": case["expect"], **safe}
    (out / "safe_state.json").write_text(json.dumps(result["safe_state"], indent=1), encoding="utf-8")
    return result


STOPPED_MPS = 0.1  # evaluation threshold: mean true speed over the last 10 s of a "stopped"/holding vehicle


def _speed_last(traj: dict[str, Any], window_s: float) -> float:
    tt, tru = np.asarray(traj["t_s"]), np.asarray(traj["true_m"])
    last = tt >= tt[-1] - window_s
    if last.sum() < 2:
        return 0.0
    return float((np.linalg.norm(np.diff(tru[last], axis=0), axis=1) / np.diff(tt[last])).mean())


def check_safe_state(
    log: list[list[Any]],
    traj: dict[str, Any],
    expect: dict[str, Any],
    t_fault_s: float,
    target: str | None,
    collisions: int,
) -> dict[str, Any]:
    """``log`` rows: [time_ns, state, reason_codes, authorized thruster commands or None, RHI accepted or None]."""
    t = np.array([row[0] for row in log], dtype=np.float64) / 1e9
    want, reason = SafetyState(expect["state"]), str(expect["reason"])
    worst_before = max(
        (SafetyState(row[1]) for row, ti in zip(log, t, strict=True) if ti < t_fault_s),
        key=SEVERITY.__getitem__,
        default=SafetyState.NORMAL,
    )
    hit = [i for i, row in enumerate(log) if t[i] >= t_fault_s and row[1] == want.value and reason in row[2]]
    first = hit[0] if hit else None
    reached_s = None if first is None else float(t[first] - t_fault_s)
    after = log[first:] if first is not None else []
    held = bool(after) and all(row[1] == want.value and reason in row[2] for row in after)
    sent = [row[3] for row in after if row[3] is not None]
    executed = [bool(row[4]) for row in after if len(row) > 4 and row[4] is not None]
    motion = str(expect["motion"])
    m: dict[str, Any] = {
        "authorized_after": len(sent),
        "executed_after": sum(executed),
        "true_speed_last_10s_mean_mps": _speed_last(traj, 10.0),
    }
    stopped = m["true_speed_last_10s_mean_mps"] <= STOPPED_MPS
    if motion == "zero":
        # all-zero commands; the vehicle really stops (RHI stop commands or the command-timeout watchdog)
        m["max_abs_command_after"] = max((abs(v) for c in sent for v in c.values()), default=0.0)
        motion_ok = m["max_abs_command_after"] == 0.0 and stopped
    elif motion == "refused":
        m["nonzero_authorized_after"] = sum(any(v != 0.0 for v in c.values()) for c in sent)
        motion_ok = m["nonzero_authorized_after"] == 0 and stopped
    elif motion == "avoid_target":
        # graceful degradation: the remaining thrusters keep being commanded AND executed, the faulted one is not
        m["max_abs_command_to_target_after"] = max((abs(c.get(str(target), 0.0)) for c in sent), default=0.0)
        m["nonzero_authorized_after"] = sum(any(v != 0.0 for v in c.values()) for c in sent)
        m["executed_fraction_after"] = sum(executed) / len(executed) if executed else 0.0
        motion_ok = (
            m["max_abs_command_to_target_after"] == 0.0
            and m["nonzero_authorized_after"] > 0
            and m["executed_fraction_after"] >= 0.9
        )
    elif motion == "hold":
        tt = np.asarray(traj["t_s"])
        est = np.asarray(traj["estimated_m"])
        if first is None:
            motion_ok = False
        else:
            k0 = int(np.searchsorted(tt, t[first]))
            drift = np.linalg.norm(est[k0:] - est[k0], axis=1)
            m["estimated_drift_from_hold_point_max_m"] = float(drift.max())
            motion_ok = drift.max() <= float(expect["hold_radius_m"]) and stopped
    else:
        raise ValueError(f"unknown motion rule {motion!r}")
    checks = {
        "nominal_before_fault": SEVERITY[worst_before] <= SEVERITY[SafetyState.DEGRADED],
        "reached_in_time": reached_s is not None and reached_s <= float(expect["deadline_s"]),
        "held_to_end": held,
        "motion_rule": bool(motion_ok),
        "no_collision": collisions == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "worst_state_before_fault": worst_before.value,
        "time_to_safe_state_s": reached_s,
        "final_state": log[-1][1] if log else None,
        "final_reasons": log[-1][2] if log else [],
        **m,
    }


__all__ = ["FAULT_CASES_PATH", "check_safe_state", "fault_cases", "run_fault_case"]
