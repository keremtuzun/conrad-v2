"""ID-REHEARSAL-E001: end-to-end SYNTHETIC_TOOLING_CHECK of the identification protocol on the Python kernel.

runner (allocator -> safety -> command gateway -> SimRobotHardware) -> delivery (CSV + manifest)
-> intake -> identification on the identification split -> held-out validation -> recovered vs true.

The "true" values are the kernel's own SYNTHETIC_ONLY simulator inputs (read here, on the evaluation
side, where truth is allowed). This validates the TOOLING only; it never produces IDENTIFIED RobotConfig
values and says nothing about the physical vehicle.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from conrad.robotics.estimation.rotations import quat_to_rot
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.hardware.identification.dataset import ExperimentKind
from conrad.robotics.hardware.identification.pipeline import PipelineConfig, PipelineResult, run_pipeline
from conrad.robotics.hardware.identification.protocol import (
    HoldProfile,
    MotionReferenceKind,
    StepProfile,
    Variant,
    WrenchPulseProfile,
)
from conrad.robotics.hardware.identification.runner import (
    EkfStateSource,
    ManoeuvreRunner,
    MotionSample,
    RunnerConfig,
)
from conrad.runtime.command_gateway import CommandGateway
from conrad.schemas.frames import WORLD, Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import RobotState
from conrad.settings import CommandMode, ExecutionLane, RuntimeSettings
from conrad.sim.kernel import SimKernelConfig, SimRobotHardware, build_sim_hardware
from conrad.sim.kernel.truth import TruthAccess

LIMITATIONS = (
    "SYNTHETIC_TOOLING_CHECK: validates the identification tooling only",
    "never produces IDENTIFIED RobotConfig values (report status SYNTHETIC_TOOLING_CHECK is refused by "
    "to_characterization_records)",
    "true values are Python-kernel SYNTHETIC_ONLY simulator inputs, not vehicle measurements",
    "motion/thrust/current reference is simulator truth plus configured synthetic noise",
    "real identification stays BLOCKED_EXTERNAL (EXT-HW-04) until the hardware owner delivers logs",
)


def euler_zyx(q: np.ndarray) -> tuple[float, float, float]:
    r = quat_to_rot(q)
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    pitch = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return roll, pitch, yaw


class KernelReference:
    """SIMULATOR_TRUTH reference with synthetic measurement noise (stands in for DVL/tracking/load cell)."""

    kind = MotionReferenceKind.SIMULATOR_TRUTH

    def __init__(self, hw: SimRobotHardware, rng: np.random.Generator, noise: dict[str, float]) -> None:
        self.name = "SYNTHETIC_ONLY:conrad.sim.kernel TruthAccess + noise"
        self._truth = TruthAccess(hw.kernel)
        self._index = hw.kernel.thrusters.index
        self._rng = rng
        self._n = noise
        self._surface = hw.kernel.config.water_surface_z_m

    def _noisy(self, x: np.ndarray, key: str) -> np.ndarray:
        return x + self._n.get(key, 0.0) * self._rng.standard_normal(x.shape)

    def motion(self) -> MotionSample:
        s = self._truth.true_state()
        p = self._noisy(s.position_world_m, "position_m")
        e = self._noisy(np.asarray(euler_zyx(s.orientation_wxyz)), "angle_rad")
        v = self._noisy(s.linear_velocity_body_mps, "velocity_mps")
        w = self._noisy(s.angular_velocity_body_rps, "rate_rps")
        f = lambda a: (float(a[0]), float(a[1]), float(a[2]))  # noqa: E731
        return MotionSample(f(p), f(e), f(v), f(w), self._surface)

    def thrust_n(self, thruster_id: str) -> float:
        t = float(self._truth.true_state().thrust_n[self._index[thruster_id]])
        return t + self._n.get("thrust_n", 0.0) * float(self._rng.standard_normal())

    def water_current_world_mps(self) -> tuple[float, float, float]:
        c = self._noisy(self._truth.true_state().current_world_mps, "current_mps")
        return float(c[0]), float(c[1]), float(c[2])


def onboard_fix(hw: SimRobotHardware, rng: np.random.Generator, cfg: dict[str, float]) -> Any:
    """The simulated robot's OWN positioning aid for its EKF (separate noise stream from the reference).

    Without it the dead-reckoned pose sigma grows until the safety supervisor holds the vehicle."""
    truth, sigma, period = TruthAccess(hw.kernel), float(cfg["sigma_m"]), float(cfg["period_s"])
    last = [-math.inf]

    def fix() -> tuple[np.ndarray, float] | None:
        t = hw.kernel.t_s
        if t - last[0] < period:
            return None
        last[0] = t
        return truth.true_state().position_world_m + sigma * rng.standard_normal(3), sigma

    return fix


def hold_policy(gains: dict[str, float], hold_xyz: np.ndarray, hold_yaw: float) -> Any:
    """Rehearsal-only PD station keeper on the reference (gains are controller settings, not vehicle data)."""
    kp, kd, kpy, kdy = gains["kp"], gains["kd"], gains["kp_yaw"], gains["kd_yaw"]

    def policy(m: MotionSample | None, _state: RobotState) -> np.ndarray:
        if m is None:
            return np.zeros(6)
        _roll, _pitch, yaw = m.euler_rpy_rad
        cy, sy = math.cos(yaw), math.sin(yaw)
        r = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        f_world = -kp * (np.asarray(m.position_world_m) - hold_xyz) - kd * (
            r @ np.asarray(m.velocity_body_mps)
        )
        f_body = r.T @ f_world
        err = (yaw - hold_yaw + math.pi) % (2 * math.pi) - math.pi
        return np.array([*f_body, 0.0, 0.0, -kpy * err - kdy * m.rate_body_rps[2]])

    return policy


def _step(p: dict[str, Any]) -> StepProfile:
    return StepProfile(
        levels=tuple(float(x) for x in p["levels"]), on_s=float(p["on_s"]), off_s=float(p["off_s"])
    )


def _pulse(p: dict[str, Any]) -> WrenchPulseProfile:
    return WrenchPulseProfile(
        fractions=tuple(float(x) for x in p["fractions"]), on_s=float(p["on_s"]), off_s=float(p["off_s"])
    )


def record_delivery(
    config: dict[str, Any], seed: int, out_dir: Path
) -> tuple[Path, SimRobotHardware, list[str]]:
    """Run the protocol on the kernel and write the delivery. Returns (manifest, hardware, aborted)."""
    robot = load_robot_config(config["robot_config"])
    kcfg = SimKernelConfig.model_validate(config.get("kernel", {}))
    hw = build_sim_hardware(robot, seed, kcfg)
    start = np.asarray(config["start_position_world_m"], dtype=np.float64)
    hw.kernel.reset(start)
    ids = IdFactory(seed=seed, namespace="id-rehearsal")
    mission, run = ids.new(), ids.new()
    gateway = CommandGateway(
        hw, robot, RuntimeSettings(command_mode=CommandMode.SIMULATED), ExecutionLane.SIMULATION, mission, run
    )
    initial = Pose(frame_id=WORLD, position_m=(float(start[0]), float(start[1]), float(start[2])))
    ref_seq, fix_seq = np.random.SeedSequence(seed).spawn(4)[2:]
    reference = KernelReference(hw, np.random.default_rng(ref_seq), config.get("reference_noise", {}))
    fix = onboard_fix(hw, np.random.default_rng(fix_seq), config["onboard_fix"])
    runner = ManoeuvreRunner(
        hw,
        gateway,
        robot,
        ids,
        hw.advance,
        reference,
        EkfStateSource(hw, robot, initial, fix),
        mission,
        run,
        RunnerConfig(
            control_period_s=float(config["control_period_s"]),
            settle_s=float(config.get("settle_s", 0.0)),
            attitude_hold=config.get("attitude_hold"),
        ),
    )
    period = float(config["control_period_s"])
    b_period = float(config.get("thruster_step_period_s", period))
    reps = int(config["repetitions"])
    prof = config["profiles"]
    validation: list[str] = []

    def rep_ids(prefix: str) -> list[str]:
        names = [f"{prefix}-r{i + 1}" for i in range(reps)]
        validation.append(names[-1])  # rule declared up front: the last repetition of each group is held out
        return names

    for tid in config.get("thrusters_under_test") or runner.thruster_ids:
        for rid in rep_ids(f"B-{tid}"):
            runner.run_thruster_step(rid, rid, tid, _step(prof["B_THRUSTER_STEP"]), b_period)
    for kind in (
        ExperimentKind.SURGE_ACCELERATION,
        ExperimentKind.SWAY,
        ExperimentKind.HEAVE,
        ExperimentKind.YAW_ROTATION,
    ):
        if kind.value not in prof:
            continue
        for rid in rep_ids(kind.value):
            runner.run_wrench_pulses(rid, rid, kind, _pulse(prof[kind.value]), period)
    if "A_TILT_THRUSTERS" in prof:
        for rid in rep_ids("A-tilt"):
            runner.run_wrench_pulses(
                rid,
                rid,
                ExperimentKind.STATIC_TRIM,
                _pulse(prof["A_TILT_THRUSTERS"]),
                period,
                Variant.TILT_THRUSTERS,
            )
    if "F_STATION_KEEPING" in prof:
        f = prof["F_STATION_KEEPING"]
        hold = HoldProfile(duration_s=float(f["duration_s"]), settled_fraction=float(f["settled_fraction"]))
        current = np.asarray(f["current_world_mps"], dtype=np.float64)
        for rid in rep_ids("F-hold"):
            m = reference.motion()
            # protocol F: head so the current lies along body surge
            policy = hold_policy(
                f["gains"], np.asarray(m.position_world_m), math.atan2(current[1], current[0])
            )
            # scenario input (SYNTHETIC_ONLY); it must cover the unrecorded settle phase too
            end_s = hw.kernel.t_s + float(config.get("settle_s", 0.0)) + hold.duration_s + 1.0
            hw.kernel.set_gust(current, end_s)
            runner.run_station_keeping(rid, rid, hold, policy, period)
            hw.kernel.set_gust(np.zeros(3), -1.0)
    aborted = [f"{s.trajectory_id}: {s.aborted}" for s in runner.segments if s.aborted]
    manifest = runner.write_delivery(
        out_dir / "delivery",
        split_id=f"id-rehearsal-seed{seed}",
        validation_repetitions=validation,
        segment_metadata={"water_density_kgm3": kcfg.water_density_kgm3},
        configuration={
            "ballast": "none (simulator)",
            "payload": "none (simulator)",
            "tether": "none (simulator)",
        },
        not_modelled={"water_temperature_c": "the Python kernel has no water temperature model"},
    )
    return manifest, hw, aborted


def true_values(hw: SimRobotHardware, config: dict[str, Any]) -> dict[str, float]:
    """Kernel SYNTHETIC_ONLY inputs, keyed like the report parameters."""
    p, kc = hw.kernel.params, hw.kernel.config
    out: dict[str, float] = {}
    for t in p.thrusters:
        out[f"B_thruster_{t.thruster_id}.k_fwd"] = t.k
        out[f"B_thruster_{t.thruster_id}.k_rev"] = t.k * t.max_reverse_n / t.max_forward_n
        out[f"B_thruster_{t.thruster_id}.time_constant_s"] = t.tau_s
        out[f"B_thruster_{t.thruster_id}.latency_s"] = t.latency_s
        out[f"B_thruster_{t.thruster_id}.deadzone"] = t.deadzone
    for kind, i in (
        (ExperimentKind.SURGE_ACCELERATION, 0),
        (ExperimentKind.SWAY, 1),
        (ExperimentKind.HEAVE, 2),
        (ExperimentKind.YAW_ROTATION, 5),
    ):
        out[f"{kind.value}.added_inertia"] = float(p.added_mass[i])
        out[f"{kind.value}.linear_damping"] = float(p.d_lin[i])
        out[f"{kind.value}.quadratic_damping"] = float(p.d_quad[i])
    g, rho = kc.gravity_mps2, kc.water_density_kgm3
    out[f"{ExperimentKind.HEAVE.value}.bias"] = rho * g * p.volume - p.mass * g
    out["D_HEAVE_displaced_volume.displaced_volume_m3"] = p.volume
    r_bg = p.r_b - p.r_g
    out["A_trim.z_bg_m"] = float(r_bg[2])
    out["A_trim.x_bg_m"] = float(r_bg[0])
    cur = np.asarray(
        config["profiles"].get("F_STATION_KEEPING", {}).get("current_world_mps", [0.0, 0.0, 0.0])
    )
    out["F_station_keeping.current_mps"] = float(cur[0])
    return out


def compare(result: PipelineResult, truth: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for fit_key, fit in sorted(result.report.fits.items()):
        for prm in fit.parameters:
            key = f"{fit_key}.{prm.name}"
            true = truth.get(key)
            z = None
            if true is not None and prm.sigma:
                z = (prm.value - true) / prm.sigma
            rows.append(
                {
                    "parameter": key,
                    "units": prm.units,
                    "recovered": prm.value,
                    "sigma": prm.sigma,
                    "true": true,
                    "abs_error": None if true is None else abs(prm.value - true),
                    "rel_error": None if not true else abs(prm.value - true) / abs(true),
                    "error_over_sigma": z,
                }
            )
    return rows


def run(config: dict[str, Any], seeds: list[int], out_dir: Path) -> dict[str, Any]:
    seed = int(seeds[0])
    manifest, hw, aborted = record_delivery(config, seed, out_dir)
    robot = load_robot_config(config["robot_config"])
    pcfg = PipelineConfig.model_validate(config.get("pipeline", {}))
    result = run_pipeline(manifest, robot, pcfg)
    (out_dir / "intake.json").write_text(
        json.dumps(result.intake.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    result.report.write(out_dir / "identification_report.json")
    rows = compare(result, true_values(hw, config))
    held = {k: v.pooled_rmse for k, v in result.report.validations.items()}
    summary = {
        "experiment_id": "ID-REHEARSAL-E001",
        "status": result.report.status.value,
        "seed": seed,
        "limitations": list(LIMITATIONS),
        "intake_accepted": result.intake.accepted,
        "intake_warnings": [f.model_dump() for f in result.intake.findings if f.severity != "FAIL"],
        "repeatability": [r.model_dump() for r in result.intake.repeatability],
        "aborted_segments": aborted,
        "parameters": rows,
        "held_out_pooled_rmse": held,
        "held_out_units": {
            k: (v.per_trajectory[0].units if v.per_trajectory else "")
            for k, v in result.report.validations.items()
        },
        "notes": result.notes,
        "not_recoverable": [
            "x_bg_m is fitted but the kernel places CoB and CoG on the same x (true 0); only z_bg is informative",
            "cross-coupling (Coriolis, Munk moment, off-diagonal added mass) is not identified by single-axis runs",
            "the kernel has no water temperature, tether or thruster advance-ratio model",
            "residual bias is expected from the kernel's discretisation (thrust updated at the start of each "
            "physics step) and from attitude-hold coupling; fit sigmas assume white residuals and understate it",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    flat: dict[str, Any] = {
        "intake_accepted": float(result.intake.accepted),
        "n_aborted": float(len(aborted)),
        "n_parameters": float(len(rows)),
    }
    for r in rows:
        if r["rel_error"] is not None:
            flat[f"rel_error.{r['parameter']}"] = float(r["rel_error"])
    for k, v in held.items():
        flat[f"held_out_rmse.{k}"] = float(v)
    return {**flat, "status": result.report.status.value}
