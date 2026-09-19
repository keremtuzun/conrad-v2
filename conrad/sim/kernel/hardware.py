"""SimRobotHardware: the simulated RobotHardwareInterface adapter over :class:`SimKernel`.

Everything returned by the RHI getters is a corrupted measurement. Truth leaves only through
``TruthAccess`` (evaluation) and the injected ``SensorRenderer`` (twin-side sensor rendering).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from conrad.robotics.estimation.rotations import quat_exp, quat_mul, quat_to_rot
from conrad.robotics.hardware.interface import RobotHardwareInterface
from conrad.schemas.frames import Pose
from conrad.schemas.observation import Observation
from conrad.schemas.robot import (
    AllocatedCommand,
    BatteryState,
    CommandAck,
    DepthSample,
    HealthLevel,
    ImuSample,
    RobotCapabilities,
    RobotConfig,
    SensorConfig,
    SystemHealth,
    ThrusterState,
)
from conrad.schemas.timebase import NS_PER_S, TimeStamp
from conrad.sim.kernel.dynamics import SimKernel
from conrad.sim.kernel.fault_injection import FaultInjectionMixin
from conrad.sim.kernel.faults import FaultRecord, SensorFaultState
from conrad.sim.kernel.sensors import SensorRenderer, _Channel
from conrad.sim.kernel.thrusters import static_thrust
from conrad.sim.kernel.truth import TruthAccess

SIM_CLOCK = "SIM"


class SimRobotHardware(FaultInjectionMixin, RobotHardwareInterface):
    adapter_name = "sim_kernel"
    is_physical = False

    def __init__(
        self,
        kernel: SimKernel,
        robot_config: RobotConfig,
        rng: np.random.Generator,
        renderer: SensorRenderer | None = None,
    ) -> None:
        self._kernel = kernel
        self._config = robot_config
        self._digest = robot_config.content_digest()
        self._rng = rng
        self._renderer = renderer
        self._estimated_pose: Callable[[], Pose | None] = lambda: None
        self._by_modality: dict[str, SensorConfig] = {}
        for s in robot_config.sensors:
            self._by_modality.setdefault(s.modality, s)
        self._imu = _Channel(self._by_modality["IMU"]) if "IMU" in self._by_modality else None
        self._depth = (
            _Channel(self._by_modality["PRESSURE_DEPTH"]) if "PRESSURE_DEPTH" in self._by_modality else None
        )
        self._exteroceptive = {
            m: _Channel(self._by_modality[m]) for m in ("RGB", "SONAR") if m in self._by_modality
        }
        self._gyro_drift = SensorFaultState()
        self._deadline_ns: int | None = None
        self._leak = False
        self._extra_faults: list[str] = []
        self.fault_log: list[FaultRecord] = []
        self.watchdog_trips = 0

    # -- wiring (integrator / evaluation side) -----------------------------------------------
    def truth_access(self) -> TruthAccess:
        """TRUTH. For evaluation/logging/twin wiring only; never pass to the robot stack."""
        return TruthAccess(self._kernel)

    @property
    def kernel(self) -> SimKernel:
        """TRUTH-side kernel handle for scenario setup (initial pose, currents); not for the stack."""
        return self._kernel

    def set_estimated_pose_provider(self, provider: Callable[[], Pose | None]) -> None:
        self._estimated_pose = provider

    def set_renderer(self, renderer: SensorRenderer | None) -> None:
        self._renderer = renderer

    # -- time ---------------------------------------------------------------------------------
    def clock_domain(self) -> str:
        return SIM_CLOCK

    def now_ns(self) -> int:
        return round(self._kernel.t_s * NS_PER_S)

    def _stamp(self, t_s: float, seq: int = 0) -> TimeStamp:
        return TimeStamp(time_ns=round(t_s * NS_PER_S), clock_domain=SIM_CLOCK, sequence_index=seq)

    def advance(self, duration_s: float) -> None:
        """Advance physics by ``duration_s`` (whole physics steps), sampling sensors on the way."""
        steps = max(1, round(duration_s / self._kernel.config.physics_dt_s))
        for _ in range(steps):
            if self._deadline_ns is not None and self.now_ns() >= self._deadline_ns:
                self._kernel.thrusters.zero(self._kernel.t_s)  # command-timeout watchdog
                self._deadline_ns = None
                self.watchdog_trips += 1
            self._kernel.step()
            self._sample_sensors()

    def _sample_sensors(self) -> None:
        t = self._kernel.t_s
        p, q, _v, w, accel = self._kernel._snapshot()
        cfg = self._kernel.config
        if self._imu is not None:
            ch = self._imu
            if ch.due(t):
                rot = quat_to_rot(q)
                specific = accel + rot.T @ np.array([0.0, 0.0, cfg.gravity_mps2])
                ns = ch.fault.noise_scale
                a_m = specific + ch.total_bias(t) + ns * ch.noise_std * self._rng.standard_normal(3)
                gyro_bias = np.array([0.0, 0.0, self._gyro_drift.bias_at(t)])
                w_m = w + gyro_bias + ns * cfg.imu_gyro_noise_rps * self._rng.standard_normal(3)
                dq = quat_exp(ns * cfg.imu_orientation_noise_rad * self._rng.standard_normal(3))
                q_m = quat_mul(q, dq)
                if not ch.fault.dropped(t):
                    ch.queue.append((t + ch.latency_s, t, np.concatenate([a_m, w_m, q_m])))
            ch.deliver(t)
        if self._depth is not None:
            ch = self._depth
            if ch.due(t):
                noise = ch.fault.noise_scale * ch.noise_std * float(self._rng.standard_normal())
                if not ch.fault.dropped(t):
                    ch.queue.append(
                        (
                            t + ch.latency_s,
                            t,
                            np.array([cfg.water_surface_z_m - p[2] + ch.total_bias(t) + noise]),
                        )
                    )
            ch.deliver(t)

    # -- RHI getters ----------------------------------------------------------------------------
    def capabilities(self) -> RobotCapabilities:
        sensors = self._config.sensors
        return RobotCapabilities(
            capability_version="sim_kernel-1",
            cameras=tuple(s.sensor_name for s in sensors if s.modality == "RGB"),
            sonars=tuple(s.sensor_name for s in sensors if s.modality == "SONAR"),
            imu=self._imu is not None,
            depth=self._depth is not None,
            thruster_count=len(self._config.thrusters),
            controllable_dof=self._controllable_dof(),
            communication_links=tuple(c.link_name for c in self._config.communications),
            onboard_compute=self._config.compute.description,
        )

    def _controllable_dof(self) -> tuple[str, ...]:
        b = self._kernel.params.allocation_matrix
        names = ("surge", "sway", "heave", "roll", "pitch", "yaw")
        return tuple(n for i, n in enumerate(names) if float(np.linalg.norm(b[i])) > 1e-9)

    def robot_config_digest(self) -> str:
        return self._digest

    def get_imu(self) -> ImuSample | None:
        if self._imu is None or self._imu.latest is None:
            return None
        measured, x = self._imu.latest
        orientation = None
        if self._kernel.config.imu_provides_orientation:
            orientation = (float(x[6]), float(x[7]), float(x[8]), float(x[9]))
        return ImuSample(
            timestamp=self._stamp(measured, self._imu.sequence),
            frame_id=self._imu.cfg.frame_id,
            linear_acceleration_mps2=(float(x[0]), float(x[1]), float(x[2])),
            angular_velocity_rps=(float(x[3]), float(x[4]), float(x[5])),
            orientation_wxyz=orientation,
        )

    def get_depth(self) -> DepthSample | None:
        if self._depth is None or self._depth.latest is None:
            return None
        measured, x = self._depth.latest
        degraded = self._depth.fault.noise_scale > 1.0
        return DepthSample(
            timestamp=self._stamp(measured, self._depth.sequence),
            frame_id=self._depth.cfg.frame_id,
            depth_m=float(x[0]),
            health=HealthLevel.DEGRADED if degraded else HealthLevel.OK,
        )

    def _render(self, modality: str) -> Observation | None:
        ch = self._exteroceptive.get(modality)
        t = self._kernel.t_s
        if ch is None or self._renderer is None or ch.fault.dropped(t) or not ch.due(t):
            return None
        truth = TruthAccess(self._kernel).true_state()
        ch.sequence += 1
        return self._renderer(
            ch.cfg.sensor_name, truth.pose(), self._estimated_pose(), self._stamp(t, ch.sequence)
        )

    def get_camera(self) -> Observation | None:
        return self._render("RGB")

    def get_sonar(self) -> Observation | None:
        return self._render("SONAR")

    def get_thruster_state(self) -> tuple[ThrusterState, ...]:
        bank, cfg = self._kernel.thrusters, self._kernel.config
        volts = self._kernel.params.battery_voltage_v
        out = []
        for i, p in enumerate(bank.params):
            fault = bank.faults[i]
            reported = cfg.report_thruster_faults
            thrust = float(bank.thrust[i]) if reported else static_thrust(p, float(bank.applied_command[i]))
            health = HealthLevel.OK
            if reported and (fault.effectiveness <= 0.0 or fault.stuck_command is not None):
                health = HealthLevel.FAULT
            elif reported and fault.effectiveness < 1.0:
                health = HealthLevel.DEGRADED
            power = cfg.thruster_power_w_per_n15 * abs(float(bank.thrust[i])) ** 1.5
            out.append(
                ThrusterState(
                    thruster_id=p.thruster_id,
                    command=float(np.clip(bank.applied_command[i], -1.0, 1.0)),
                    estimated_thrust_n=thrust,
                    current_a=power / volts if volts > 0 else None,
                    health=health,
                )
            )
        return tuple(out)

    def get_power_state(self) -> BatteryState | None:
        k = self._kernel
        volts = k.params.battery_voltage_v
        return BatteryState(
            timestamp=self._stamp(k.t_s),
            remaining_fraction=k.battery_remaining_fraction,
            voltage_v=volts,
            current_a=k.power_w / volts if volts > 0 else None,
            energy_used_j=k.energy_used_j,
        )

    def get_health(self) -> SystemHealth:
        k, t = self._kernel, self._kernel.t_s
        devices: dict[str, HealthLevel] = {s.thruster_id: s.health for s in self.get_thruster_state()}
        channels = [c for c in (self._imu, self._depth, *self._exteroceptive.values()) if c is not None]
        for ch in channels:
            if ch.fault.dropped(t):
                devices[ch.cfg.sensor_name] = HealthLevel.FAULT
            elif ch.fault.noise_scale > 1.0:
                devices[ch.cfg.sensor_name] = HealthLevel.DEGRADED
            else:
                devices[ch.cfg.sensor_name] = HealthLevel.OK
        faults = list(self._extra_faults)
        volts = k.params.battery_voltage_v
        if volts > 0 and k.power_w / volts > k.params.battery_max_current_a:
            faults.append("OVERCURRENT")
        if self._leak:
            faults.append("LEAK")
        overall = HealthLevel.OK
        if any(h is not HealthLevel.OK for h in devices.values()) or faults:
            overall = HealthLevel.DEGRADED
        if self._leak:
            overall = HealthLevel.FAULT
        return SystemHealth(
            timestamp=self._stamp(t),
            overall=overall,
            leak_detected=self._leak,
            devices=devices,
            faults=tuple(faults),
        )

    # -- actuation (Command Gateway only) ---------------------------------------------------------
    def send(self, command: AllocatedCommand) -> CommandAck:
        known = set(self._kernel.thrusters.index)
        reasons: list[str] = []
        if set(command.thruster_commands) - known:
            reasons.append("UNKNOWN_THRUSTER_ID")
        if any(not math.isfinite(v) or abs(v) > 1.0 for v in command.thruster_commands.values()):
            reasons.append("COMMAND_NOT_FINITE_OR_OUT_OF_RANGE")
        if not reasons:
            self._kernel.thrusters.command(self._kernel.t_s, dict(command.thruster_commands))
            self._deadline_ns = command.deadline_ns
        return CommandAck(
            command_id=command.command_id,
            accepted=not reasons,
            reason_codes=tuple(reasons),
            ack_time_ns=self.now_ns(),
            adapter=self.adapter_name,
        )
