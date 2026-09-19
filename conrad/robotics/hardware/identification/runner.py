"""Manoeuvre runner: executes the identification protocol through the normal actuation path and logs it.

Command path (identical for the simulator, Unity and the physical robot)::

    profile -> WrenchCommand -> ThrusterAllocator -> SafetySupervisor.authorize -> CommandGateway.submit

The runner never calls ``RobotHardwareInterface.send``. Thruster steps (B) use the same allocator with
every other thruster masked out and deadzone compensation off, so the command that reaches the thruster
is exactly the protocol level.

Against a physical interface the runner refuses to start unless hardware control is explicitly enabled,
a safety envelope is supplied and the operator has acknowledged the run. The default is OFF. The command
gateway applies its own, independent hardware gating on top of this.

Time advances through an injected ``advance(seconds)`` callable: ``SimRobotHardware.advance`` for the
Python kernel, a lockstep call for a Unity adapter, a real-time wait for hardware. Logs from a
non-physical interface are written ``synthetic: true`` with the adapter as source.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import numpy as np
import yaml
from pydantic import Field

from conrad.robotics.allocation.allocator import AllocationConfig, ThrusterAllocator
from conrad.robotics.estimation.ekf import EkfStateEstimator
from conrad.robotics.estimation.rotations import quat_to_rot
from conrad.robotics.hardware.identification.dataset import ExperimentKind
from conrad.robotics.hardware.identification.protocol import (
    PROTOCOL_VERSION,
    WRENCH_AXIS,
    WRENCH_COLUMNS,
    HoldProfile,
    MotionReferenceKind,
    StepProfile,
    Variant,
    WrenchPulseProfile,
)
from conrad.robotics.hardware.interface import RobotHardwareInterface
from conrad.robotics.safety.monitors import SafetyInputs, SafetyState
from conrad.robotics.safety.supervisor import SafetySupervisor
from conrad.runtime.command_gateway import CommandGateway
from conrad.schemas.base import ConradModel
from conrad.schemas.frames import ROBOT, Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import AllocatedCommand, RobotConfig, RobotState, WrenchCommand
from conrad.schemas.timebase import NS_PER_S, TimeStamp

RUN_STATES = frozenset({SafetyState.NORMAL, SafetyState.DEGRADED})


class HardwareControlRefusedError(RuntimeError):
    """The runner will not command a physical vehicle without explicit enable, envelope and acknowledgement."""


class ManoeuvreAbortedError(RuntimeError):
    pass


class IdentificationSafetyEnvelope(ConradModel):
    """Limits for one identification session. Every field is required: there are no default limits."""

    max_abs_command: float = Field(gt=0, le=1, description="normalised thruster command cap")
    max_segment_duration_s: float = Field(gt=0)
    max_speed_mps: float = Field(gt=0, description="independent-reference speed that aborts a run")
    max_depth_m: float = Field(gt=0, description="independent-reference depth that aborts a run")
    max_abs_roll_pitch_rad: float = Field(gt=0)
    set_by: str = Field(min_length=1, description="who declared these limits (never a default)")


class AttitudeHoldGains(ConradModel):
    """PD attitude hold on the robot's OWN attitude estimate, for the rotational axes a manoeuvre does not
    excite. Controller tuning, not a vehicle property; OPEN for the physical vehicle until tuned."""

    kp_nm_per_rad: float = Field(gt=0)
    kd_nms_per_rad: float = Field(ge=0)


class RunnerConfig(ConradModel):
    hardware_control_enabled: bool = False
    operator_acknowledged: bool = False
    safety_envelope: IdentificationSafetyEnvelope | None = None
    control_period_s: float = Field(default=0.02, gt=0)
    warmup_s: float = Field(default=0.2, ge=0, description="let first IMU/depth samples arrive")
    settle_s: float = Field(
        default=0.0, ge=0, description="unrecorded zero-force attitude-hold period before each wrench segment"
    )
    attitude_hold: AttitudeHoldGains | None = Field(
        default=None, description="None = open loop on every axis (single-axis runs then drift and couple)"
    )


@dataclass(frozen=True)
class MotionSample:
    """Independent pose/velocity: WORLD position, ZYX Euler (rad), BODY linear and angular velocity."""

    position_world_m: tuple[float, float, float]
    euler_rpy_rad: tuple[float, float, float]
    velocity_body_mps: tuple[float, float, float]
    rate_body_rps: tuple[float, float, float]
    surface_z_m: float = 0.0

    @property
    def depth_m(self) -> float:
        return self.surface_z_m - self.position_world_m[2]


class ReferenceSource(Protocol):
    """Independent measurements that are NOT the robot's estimator (DVL/USBL/tracking/load cell/meter)."""

    name: str
    kind: MotionReferenceKind

    def motion(self) -> MotionSample | None: ...

    def thrust_n(self, thruster_id: str) -> float | None: ...

    def water_current_world_mps(self) -> tuple[float, float, float] | None: ...


class StateSource(Protocol):
    """The robot's OWN state estimate, used only for safety supervision (never logged as reference)."""

    def update(self, dt_s: float) -> None: ...

    def get_state(self) -> RobotState: ...

    def reasons(self) -> tuple[str, ...]: ...


HoldPolicy = Callable[[MotionSample | None, RobotState], np.ndarray]
"""F: returns the 6-vector body wrench that holds station (the hold controller is the caller's)."""


class EkfStateSource:
    """Feeds the robot's EKF from the RHI getters, exactly as the navigation stack does."""

    def __init__(
        self,
        hardware: RobotHardwareInterface,
        robot_config: RobotConfig,
        initial_pose: Pose,
        position_fix: Callable[[], tuple[np.ndarray, float] | None] | None = None,
    ) -> None:
        """``position_fix`` is the robot's OWN positioning aid (e.g. its onboard USBL/DVL-derived fix), if any.
        It must not be the independent reference that is logged for identification."""
        self._hw = hardware
        self._fix = position_fix
        stamp = TimeStamp(time_ns=hardware.now_ns(), clock_domain=hardware.clock_domain())
        self.estimator = EkfStateEstimator(robot_config, initial_pose, stamp)
        self._last_depth_ns = -1

    def update(self, dt_s: float) -> None:
        imu = self._hw.get_imu()
        if imu is not None:
            self.estimator.predict(imu, self._hw.get_thruster_state(), dt_s)
        depth = self._hw.get_depth()
        if depth is not None and depth.timestamp.time_ns > self._last_depth_ns:
            self._last_depth_ns = depth.timestamp.time_ns
            self.estimator.update_depth(depth)
        fix = self._fix() if self._fix is not None else None
        if fix is not None:
            self.estimator.update_position_fix(np.asarray(fix[0], dtype=np.float64), float(fix[1]))

    def get_state(self) -> RobotState:
        return self.estimator.get_state()

    def reasons(self) -> tuple[str, ...]:
        return self.estimator.health_reasons()


@dataclass
class RecordedSegment:
    trajectory_id: str
    kind: ExperimentKind
    variant: Variant
    repetition_id: str
    thruster_id: str | None
    columns: dict[str, list[float]]
    units: dict[str, str]
    aborted: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.columns.get("time_s", ()))


def _euler_zyx(q: np.ndarray) -> tuple[float, float, float]:
    r = quat_to_rot(q)
    return (
        math.atan2(float(r[2, 1]), float(r[2, 2])),
        math.asin(max(-1.0, min(1.0, -float(r[2, 0])))),
        math.atan2(float(r[1, 0]), float(r[0, 0])),
    )


def check_hardware_permission(hardware: RobotHardwareInterface, config: RunnerConfig) -> None:
    """Fail closed before anything is constructed or commanded."""
    if not hardware.is_physical:
        return
    missing = []
    if not config.hardware_control_enabled:
        missing.append("hardware_control_enabled is false (default OFF)")
    if config.safety_envelope is None:
        missing.append("no safety envelope")
    if not config.operator_acknowledged:
        missing.append("operator acknowledgement missing")
    if missing:
        raise HardwareControlRefusedError(
            f"refusing to command physical adapter {hardware.adapter_name!r}: " + "; ".join(missing)
        )


class ManoeuvreRunner:
    def __init__(
        self,
        hardware: RobotHardwareInterface,
        gateway: CommandGateway,
        robot_config: RobotConfig,
        ids: IdFactory,
        advance: Callable[[float], None],
        reference: ReferenceSource,
        state_source: StateSource,
        mission_id: UUID,
        run_id: UUID,
        config: RunnerConfig | None = None,
    ) -> None:
        self.config = cfg = config or RunnerConfig()
        check_hardware_permission(hardware, cfg)  # before touching anything else
        if hardware.is_physical and reference.kind is MotionReferenceKind.SIMULATOR_TRUTH:
            raise HardwareControlRefusedError("a physical run cannot use simulator truth as its reference")
        if not isinstance(gateway, CommandGateway):
            raise TypeError("commands must go through conrad.runtime.command_gateway.CommandGateway")
        if gateway.robot_config_digest != robot_config.content_digest():
            raise ValueError("gateway and runner were built from different RobotConfigs")
        self._hw = hardware
        self._gw = gateway
        self._cfg = robot_config
        self._ids = ids
        self._advance = advance
        self.reference = reference
        self._state = state_source
        self._mission, self._run = mission_id, run_id
        self._clock = hardware.clock_domain()
        self.supervisor = SafetySupervisor(robot_config, ids)
        self.allocator = ThrusterAllocator(robot_config, ids)
        self.thruster_ids = self.allocator.thruster_ids
        self._k = {
            t.thruster_id: (
                float(t.thrust_coefficient.require("thrust_coefficient")),
                float(t.max_forward_thrust_n.require("max_forward_thrust_n")),
                float(t.max_reverse_thrust_n.require("max_reverse_thrust_n")),
            )
            for t in robot_config.thrusters
        }
        self.synthetic = not hardware.is_physical
        self.source = (
            f"SYNTHETIC_ONLY:{hardware.adapter_name}"
            if self.synthetic
            else f"PHYSICAL:{hardware.adapter_name}"
        )
        self.segments: list[RecordedSegment] = []
        self._trace = ids.new()
        self._warm = False

    # -- command path -------------------------------------------------------------------------------------
    def _wrench(self, tau: np.ndarray) -> WrenchCommand:
        return WrenchCommand(
            command_id=self._ids.new(),
            trace_id=self._trace,
            timestamp=TimeStamp(time_ns=self._hw.now_ns(), clock_domain=self._clock),
            frame_id=ROBOT,
            force_n=(float(tau[0]), float(tau[1]), float(tau[2])),
            torque_nm=(float(tau[3]), float(tau[4]), float(tau[5])),
        )

    def _submit(
        self, allocator: ThrusterAllocator, tau: np.ndarray
    ) -> tuple[AllocatedCommand | None, str | None]:
        """Allocate, supervise and submit one command. Returns (sent command, abort reason)."""
        now = self._hw.now_ns()
        cmd, _ = allocator.allocate(
            self._wrench(tau),
            mission_id=self._mission,
            run_id=self._run,
            now_ns=now,
            clock_domain=self._clock,
        )
        env = self.config.safety_envelope
        if env is not None and any(
            abs(v) > env.max_abs_command + 1e-12 for v in cmd.thruster_commands.values()
        ):
            return None, "ENVELOPE_COMMAND_LIMIT"
        assessment = self.supervisor.assess(
            SafetyInputs(
                now_ns=now,
                clock_domain=self._clock,
                state=self._state.get_state(),
                estimator_reasons=self._state.reasons(),
                health=self._hw.get_health(),
                battery=self._hw.get_power_state(),
                thrusters=self._hw.get_thruster_state(),
            )
        )
        if assessment.state not in RUN_STATES:
            return None, f"SAFETY_STATE_{assessment.state.value}:{','.join(assessment.reason_codes)}"
        decision = self.supervisor.authorize(cmd, assessment, now, self._clock)
        if not decision.authorized:
            return None, "SAFETY_REFUSED:" + ",".join(decision.reason_codes)
        ack = self._gw.submit(decision.command)
        if not ack.accepted:
            return None, "GATEWAY_REJECTED:" + ",".join(ack.reason_codes)
        return decision.command, None

    def stop(self) -> None:
        """Explicit all-zero command through the same path (allowed in every safety state)."""
        now = self._hw.now_ns()
        cmd = self.allocator.zero_command(
            trace_id=self._trace,
            mission_id=self._mission,
            run_id=self._run,
            now_ns=now,
            clock_domain=self._clock,
            source_wrench_id=self._trace,
        )
        assessment = self.supervisor.last
        if assessment is None:
            return
        decision = self.supervisor.authorize(cmd, assessment, now, self._clock)
        if decision.authorized:
            self._gw.submit(decision.command)

    def _hold_moment(self, held: tuple[int, ...], target_yaw: float) -> np.ndarray:
        """Body moments (mx, my, mz) holding roll = pitch = 0 and yaw = target on the ``held`` axes (3..5)."""
        gains = self.config.attitude_hold
        out = np.zeros(3)
        if gains is None or not held:
            return out
        st = self._state.get_state()
        roll, pitch, yaw = _euler_zyx(np.asarray(st.pose.orientation_wxyz, dtype=np.float64))
        rates = np.asarray(st.angular_velocity_body_rps or (0.0, 0.0, 0.0), dtype=np.float64)
        err = np.array([roll, pitch, (yaw - target_yaw + math.pi) % (2 * math.pi) - math.pi])
        for axis in held:
            j = axis - 3
            out[j] = -gains.kp_nm_per_rad * err[j] - gains.kd_nms_per_rad * rates[j]
        return out

    # -- recording ------------------------------------------------------------------------------------------
    def _envelope_violation(self, m: MotionSample | None, elapsed_s: float) -> str | None:
        env = self.config.safety_envelope
        if env is None:
            return None
        if elapsed_s > env.max_segment_duration_s:
            return "ENVELOPE_DURATION"
        if m is None:
            return "ENVELOPE_REFERENCE_LOST" if self._hw.is_physical else None
        if math.hypot(*m.velocity_body_mps) > env.max_speed_mps:
            return "ENVELOPE_SPEED"
        if m.depth_m > env.max_depth_m:
            return "ENVELOPE_DEPTH"
        if max(abs(m.euler_rpy_rad[0]), abs(m.euler_rpy_rad[1])) > env.max_abs_roll_pitch_rad:
            return "ENVELOPE_ATTITUDE"
        return None

    def _row(self, seg: RecordedSegment, cmd: AllocatedCommand, tau: np.ndarray | None) -> None:
        nan = float("nan")
        row: dict[str, tuple[float, str]] = {"time_s": (self._hw.now_ns() / NS_PER_S, "s")}
        for tid in self.thruster_ids:
            row[f"cmd_{tid}"] = (float(cmd.thruster_commands[tid]), "unitless")
        if tau is not None:
            for (name, units), v in zip(WRENCH_COLUMNS, tau, strict=True):
                row[name] = (float(v), units)
        for ts in self._hw.get_thruster_state():
            if ts.rpm is not None:
                row[f"rpm_{ts.thruster_id}"] = (float(ts.rpm), "rpm")
            if ts.current_a is not None:
                row[f"thr_current_{ts.thruster_id}_a"] = (float(ts.current_a), "A")
        imu = self._hw.get_imu()
        acc = imu.linear_acceleration_mps2 if imu is not None else (nan, nan, nan)
        gyr = imu.angular_velocity_rps if imu is not None else (nan, nan, nan)
        for ax, a, g in zip("xyz", acc, gyr, strict=True):
            row[f"imu_a{ax}_mps2"] = (float(a), "m/s^2")
            row[f"imu_g{ax}_rps"] = (float(g), "rad/s")
        depth = self._hw.get_depth()
        row["pressure_depth_m"] = (depth.depth_m if depth is not None else nan, "m")
        power = self._hw.get_power_state()
        row["battery_voltage_v"] = (
            float(power.voltage_v) if power is not None and power.voltage_v is not None else nan,
            "V",
        )
        row["battery_current_a"] = (
            float(power.current_a) if power is not None and power.current_a is not None else nan,
            "A",
        )
        m = self.reference.motion()
        if m is not None:
            vals = (*m.position_world_m, *m.euler_rpy_rad, *m.velocity_body_mps, *m.rate_body_rps)
            names = (
                ("ref_x_m", "m"),
                ("ref_y_m", "m"),
                ("ref_z_m", "m"),
                ("ref_roll_rad", "rad"),
                ("ref_pitch_rad", "rad"),
                ("ref_yaw_rad", "rad"),
                ("ref_u_mps", "m/s"),
                ("ref_v_mps", "m/s"),
                ("ref_w_mps", "m/s"),
                ("ref_p_rps", "rad/s"),
                ("ref_q_rps", "rad/s"),
                ("ref_r_rps", "rad/s"),
            )
            for (n, u), v in zip(names, vals, strict=True):
                row[n] = (float(v), u)
        if seg.thruster_id is not None:
            thrust = self.reference.thrust_n(seg.thruster_id)
            row["thrust_n"] = (float(thrust) if thrust is not None else nan, "N")
        if seg.kind is ExperimentKind.STATION_KEEPING:
            cur = self.reference.water_current_world_mps()
            if cur is not None:
                row["current_x_mps"] = (float(cur[0]), "m/s")
                row["current_y_mps"] = (float(cur[1]), "m/s")
                row["current_z_mps"] = (float(cur[2]), "m/s")
        n_prev = seg.n_samples
        for name, (value, units) in row.items():
            col = seg.columns.setdefault(name, [nan] * n_prev)
            seg.units.setdefault(name, units)
            col.append(value)
        for name, col in seg.columns.items():
            if name not in row:
                col.append(nan)

    def _warmup(self) -> None:
        if not self._warm and self.config.warmup_s > 0:
            self._advance(self.config.warmup_s)
            self._state.update(self.config.warmup_s)
        self._warm = True

    def _execute(
        self,
        seg: RecordedSegment,
        duration_s: float,
        wrench_at: Callable[[float, MotionSample | None], np.ndarray],
        allocator: ThrusterAllocator,
        period_s: float,
        log_wrench: bool,
        excited: tuple[int, ...] = (),
    ) -> RecordedSegment:
        self._warmup()
        env = self.config.safety_envelope
        if env is not None and duration_s > env.max_segment_duration_s:
            raise ManoeuvreAbortedError(
                f"{seg.trajectory_id}: profile lasts {duration_s:.1f} s, envelope allows "
                f"{env.max_segment_duration_s:.1f} s"
            )
        steps = round(duration_s / period_s)
        held = tuple(a for a in (3, 4, 5) if a not in excited)
        target_yaw = _euler_zyx(np.asarray(self._state.get_state().pose.orientation_wxyz, dtype=np.float64))[
            2
        ]
        if self.config.attitude_hold is not None and held:
            seg.notes.append(f"attitude hold on body axes {held} (robot's own estimate)")
        if log_wrench and self.config.settle_s > 0 and self.config.attitude_hold is not None:
            for _ in range(round(self.config.settle_s / period_s)):
                tau0 = np.zeros(6)
                tau0[3:] = self._hold_moment((3, 4, 5), target_yaw)
                _, reason = self._submit(allocator, tau0)
                if reason is not None:
                    seg.aborted = f"settle {reason}"
                    self.stop()
                    self.segments.append(seg)
                    return seg
                self._advance(period_s)
                self._state.update(period_s)
        try:
            for k in range(steps):
                elapsed = k * period_s
                m = self.reference.motion()
                reason = self._envelope_violation(m, elapsed)
                if reason is None:
                    tau = np.array(wrench_at(elapsed, m), dtype=np.float64)
                    tau[3:] += self._hold_moment(held, target_yaw)
                    cmd, reason = self._submit(allocator, tau)
                if reason is not None:
                    seg.aborted = f"t={elapsed:.3f}s {reason}"
                    break
                assert cmd is not None
                self._row(seg, cmd, tau if log_wrench else None)
                self._advance(period_s)
                self._state.update(period_s)
        finally:
            self.stop()
        self.segments.append(seg)
        return seg

    # -- manoeuvres -------------------------------------------------------------------------------------------
    def run_thruster_step(
        self, trajectory_id: str, repetition_id: str, thruster_id: str, profile: StepProfile, period_s: float
    ) -> RecordedSegment:
        """B: one thruster, levels alternating with zero plateaus. The load-cell thrust is the reference."""
        if thruster_id not in self.thruster_ids:
            raise KeyError(f"unknown thruster {thruster_id!r}")
        single = ThrusterAllocator(
            self._cfg, self._ids, AllocationConfig(deadzone_compensation=False, fault_aware=False)
        )
        single.set_effectiveness({t: float(t == thruster_id) for t in self.thruster_ids})
        column = single.matrix[:, self.thruster_ids.index(thruster_id)]
        k, fmax, rmax = self._k[thruster_id]
        k_rev = k * rmax / fmax if fmax > 0 else k
        cycle = profile.on_s + profile.off_s

        def wrench_at(t: float, _m: MotionSample | None) -> np.ndarray:
            i, phase = int(t // cycle), t % cycle
            u = profile.levels[i] if i < len(profile.levels) and phase >= profile.off_s else 0.0
            thrust = k * u * u if u >= 0 else -k_rev * u * u  # inverted exactly by the allocator
            return np.asarray(column * thrust, dtype=np.float64)

        seg = RecordedSegment(
            trajectory_id, ExperimentKind.THRUSTER_STEP, Variant.STANDARD, repetition_id, thruster_id, {}, {}
        )
        return self._execute(
            seg, profile.duration_s, wrench_at, single, period_s, log_wrench=False, excited=(3, 4, 5)
        )

    def run_wrench_pulses(
        self,
        trajectory_id: str,
        repetition_id: str,
        kind: ExperimentKind,
        profile: WrenchPulseProfile,
        period_s: float,
        variant: Variant = Variant.STANDARD,
    ) -> RecordedSegment:
        """A-tilt (pitch) / C / D / E: signed plateaus as fractions of the axis capability, then coast."""
        axis = WRENCH_AXIS[kind]
        if kind is ExperimentKind.STATIC_TRIM and variant is not Variant.TILT_THRUSTERS:
            raise ValueError("only the TILT_THRUSTERS static-trim variant is executed by thrusters")
        capability = self.allocator.capability()[("fx", "fy", "fz", "mx", "my", "mz")[axis]]
        if capability <= 0:
            raise ManoeuvreAbortedError(f"{kind.value}: the thruster layout cannot actuate axis {axis}")
        cycle = profile.on_s + profile.off_s

        def wrench_at(t: float, _m: MotionSample | None) -> np.ndarray:
            i, phase = int(t // cycle), t % cycle
            tau = np.zeros(6)
            if i < len(profile.fractions) and phase >= profile.off_s:
                tau[axis] = profile.fractions[i] * capability
            return tau

        seg = RecordedSegment(trajectory_id, kind, variant, repetition_id, None, {}, {})
        seg.notes.append(f"axis capability {capability:.6g} (allocator, from RobotConfig)")
        return self._execute(
            seg, profile.duration_s, wrench_at, self.allocator, period_s, log_wrench=True, excited=(axis,)
        )

    def run_station_keeping(
        self,
        trajectory_id: str,
        repetition_id: str,
        profile: HoldProfile,
        policy: HoldPolicy,
        period_s: float,
    ) -> RecordedSegment:
        """F: hold station with the caller's policy while the current meter and reference record."""

        def wrench_at(_t: float, m: MotionSample | None) -> np.ndarray:
            return np.asarray(policy(m, self._state.get_state()), dtype=np.float64)

        seg = RecordedSegment(
            trajectory_id, ExperimentKind.STATION_KEEPING, Variant.STANDARD, repetition_id, None, {}, {}
        )
        seg.notes.append(f"settled_fraction={profile.settled_fraction}")
        return self._execute(
            seg, profile.duration_s, wrench_at, self.allocator, period_s, log_wrench=True, excited=(5,)
        )

    # -- delivery ---------------------------------------------------------------------------------------------
    def write_delivery(
        self,
        out_dir: str | Path,
        split_id: str,
        validation_repetitions: Sequence[str],
        segment_metadata: dict[str, float],
        configuration: dict[str, str],
        not_modelled: dict[str, str] | None = None,
        reference_device: str = "",
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write CSVs + the YAML manifest. The split is fixed here, before any fitting happens."""
        out = Path(out_dir)
        (out / "logs").mkdir(parents=True, exist_ok=True)
        val = set(validation_repetitions)
        reps = {s.repetition_id for s in self.segments if s.aborted is None}
        entries = []
        for s in self.segments:
            if s.aborted is not None:
                continue  # an aborted repetition is not delivered as data
            names = ["time_s", *sorted(n for n in s.columns if n != "time_s")]
            rel = f"logs/{s.trajectory_id}.csv"
            with (out / rel).open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(names)
                for i in range(s.n_samples):
                    w.writerow([repr(float(s.columns[n][i])) for n in names])
            entry: dict[str, Any] = {
                "trajectory_id": s.trajectory_id,
                "kind": s.kind.value,
                "variant": s.variant.value,
                "repetition_id": s.repetition_id,
                "split": "validation" if s.repetition_id in val else "identification",
                "file": rel,
                "units": {n: s.units[n] for n in names if n != "time_s"},
                "metadata": dict(segment_metadata),
                "synthetic": self.synthetic,
                "source": self.source,
                "clock_domain": self._clock,
            }
            if s.thruster_id is not None:
                entry["thruster_id"] = s.thruster_id
            if s.notes:
                entry["notes"] = list(s.notes)
            entries.append(entry)
        manifest: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "split_id": split_id,
            "synthetic": self.synthetic,
            "source": self.source,
            "thruster_ids": list(self.thruster_ids),
            "clock": {
                "domain": self._clock,
                "synchronization": "single adapter clock: every column sampled on the same tick",
                "max_offset_s": 0.0 if self.synthetic else None,
            },
            "motion_reference": {
                "kind": self.reference.kind.value,
                "device": reference_device or self.reference.name,
                "independent_of_robot_estimator": True,
            },
            "thrust_measurement": {"device": self.reference.name},
            "current_measurement": {"device": self.reference.name},
            "configuration": dict(configuration),
            "split_plan": {
                "assigned_before_fitting": True,
                "validation_repetitions": sorted(val & reps),
                "identification_repetitions": sorted(reps - val),
            },
            "aborted": [
                {"trajectory_id": s.trajectory_id, "reason": s.aborted} for s in self.segments if s.aborted
            ],
            "segments": entries,
        }
        if not_modelled:
            manifest["not_modelled"] = dict(not_modelled)
        if extra:
            manifest.update(extra)
        path = out / "manifest.yaml"
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        return path
