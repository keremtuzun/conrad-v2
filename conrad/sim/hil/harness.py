"""HIL harness: run a deployment-equivalent stack loop against ANY RobotHardwareInterface.

    hardware (kernel / Unity / physical)  --get_*-->  stack_step (injected)  --command-->  submit (gateway)

* The harness never calls ``RobotHardwareInterface.send``; commands go through the injected ``submit``
  (the Command Gateway), exactly as in deployment.
* ``advance`` optionally drives simulated time (kernel ``advance`` / Unity lock-step ``step``); its cost
  is reported separately and excluded from the stack's observation->command latency.
* TARGET mode refuses to run anywhere but the configured onboard computer and reports BLOCKED_EXTERNAL.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import Field

from conrad.robotics.hardware.interface import HardwareUnavailableError, RobotHardwareInterface
from conrad.schemas.base import ConradModel
from conrad.schemas.observation import Observation
from conrad.schemas.robot import (
    AllocatedCommand,
    BatteryState,
    CommandAck,
    DepthSample,
    ImuSample,
    SystemHealth,
    ThrusterState,
)
from conrad.sim.hil.metrics import (
    Clock,
    FixedRateScheduler,
    LatencyStats,
    ResourceSampler,
    Sleep,
    default_gpu_probe,
)
from conrad.sim.hil.report import (
    FaultTrialResult,
    GateStatus,
    HilGateCriteria,
    HilGateReport,
    HilMode,
    SensorRateResult,
    evaluate_gate,
)


@dataclass(frozen=True)
class SensorFrame:
    """Everything the stack may read in one cycle, exactly as the RHI returned it."""

    cycle: int
    hw_time_ns: int
    imu: ImuSample | None
    depth: DepthSample | None
    camera: Observation | None
    sonar: Observation | None
    thrusters: tuple[ThrusterState, ...]
    power: BatteryState | None
    health: SystemHealth


@dataclass(frozen=True)
class StackOutput:
    command: AllocatedCommand | None
    stage_ns: dict[str, int] = field(default_factory=dict)
    queue_backlog: int = 0


StackStep = Callable[[SensorFrame], StackOutput]
Submit = Callable[[AllocatedCommand], CommandAck]
Advance = Callable[[int], None]


@dataclass(frozen=True)
class FaultTrial:
    name: str
    at_cycle: int
    inject: Callable[[], None]
    recovered: Callable[[SensorFrame], bool]
    max_recovery_cycles: int


class HilConfig(ConradModel):
    mode: HilMode = HilMode.HOST
    rate_hz: float = Field(gt=0)
    cycles: int = Field(gt=0)
    warmup_cycles: int = Field(default=0, ge=0)
    sensor_rates_hz: dict[str, float] = Field(default_factory=dict, description="imu/depth/camera/sonar")
    stale_tolerance: float = Field(default=1.5, gt=1.0, description="sample is late after tol/rate s")
    target_platform_id: str | None = None
    criteria: HilGateCriteria


def default_platform_id() -> str:
    return f"{platform.system()}-{platform.machine()}-{platform.node()}"


class _SensorTracker:
    def __init__(self, name: str, rate_hz: float, tolerance: float) -> None:
        self.name, self.rate_hz, self.tolerance = name, rate_hz, tolerance
        self.last_acq: int | None = None
        self.first_hw: int | None = None
        self.new = 0
        self.drops = 0
        self.ages_ns: list[int] = []

    def update(self, acq_ns: int | None, hw_now: int) -> None:
        if self.first_hw is None:
            self.first_hw = hw_now
        if acq_ns is not None and acq_ns != self.last_acq:
            self.new += 1
            self.last_acq = acq_ns
        reference = self.last_acq if self.last_acq is not None else self.first_hw
        if (hw_now - reference) / 1e9 > self.tolerance / self.rate_hz:
            self.drops += 1
        if acq_ns is not None:
            self.ages_ns.append(max(0, hw_now - acq_ns))

    def result(self, hw_end: int) -> SensorRateResult:
        span = None if self.first_hw is None else (hw_end - self.first_hw) / 1e9
        return SensorRateResult(
            sensor=self.name,
            expected_hz=self.rate_hz,
            new_samples=self.new,
            frame_drops=self.drops,
            achieved_hz=None if not span else self.new / span,
        )


def _acq(sample: ImuSample | DepthSample | Observation | None) -> int | None:
    return None if sample is None else sample.timestamp.time_ns


class HilHarness:
    def __init__(
        self,
        hardware: RobotHardwareInterface,
        stack_step: StackStep,
        submit: Submit,
        config: HilConfig,
        *,
        advance: Advance | None = None,
        fault_trials: tuple[FaultTrial, ...] = (),
        clock: Clock = time.perf_counter_ns,
        sleep: Sleep = time.sleep,
        gpu_probe: Callable[[], tuple[str, int | None]] = default_gpu_probe,
        platform_id: Callable[[], str] = default_platform_id,
    ) -> None:
        self._hw, self._step, self._submit, self._cfg = hardware, stack_step, submit, config
        self._advance, self._trials = advance, fault_trials
        self._clock, self._sleep = clock, sleep
        self._gpu_probe, self._platform_id = gpu_probe, platform_id

    def _base_report(self, status: GateStatus, reason: str | None) -> HilGateReport:
        digest = clock = None
        if status is not GateStatus.BLOCKED_EXTERNAL:
            digest, clock = self._hw.robot_config_digest(), self._hw.clock_domain()
        return HilGateReport(
            mode=self._cfg.mode,
            status=status,
            blocked_reason=reason,
            adapter_name=self._hw.adapter_name,
            adapter_is_physical=self._hw.is_physical,
            robot_config_digest=digest,
            clock_domain=clock,
            host={"platform_id": self._platform_id(), "python": platform.python_version()},
            rate_hz=self._cfg.rate_hz,
            cycles_planned=self._cfg.cycles,
            cycles_run=0,
            missed_deadlines=0,
            skipped_ticks=0,
        )

    def _read(self, cycle: int) -> SensorFrame:
        hw = self._hw
        return SensorFrame(
            cycle=cycle,
            hw_time_ns=hw.now_ns(),
            imu=hw.get_imu(),
            depth=hw.get_depth(),
            camera=hw.get_camera(),
            sonar=hw.get_sonar(),
            thrusters=hw.get_thruster_state(),
            power=hw.get_power_state(),
            health=hw.get_health(),
        )

    def run(self) -> HilGateReport:
        cfg = self._cfg
        if cfg.mode is HilMode.TARGET and (
            cfg.target_platform_id is None or self._platform_id() != cfg.target_platform_id
        ):
            reason = (
                "target-HIL must execute on the onboard computer "
                f"(configured {cfg.target_platform_id!r}, this host {self._platform_id()!r}); EXT-HW-HIL"
            )
            return self._base_report(GateStatus.BLOCKED_EXTERNAL, reason)
        report = self._base_report(GateStatus.FAIL, None)
        period_ns = round(1e9 / cfg.rate_hz)
        trackers = {n: _SensorTracker(n, r, cfg.stale_tolerance) for n, r in cfg.sensor_rates_hz.items()}
        e2e: list[int] = []
        cycle_ns: list[int] = []
        stages: dict[str, list[int]] = {}
        errors: list[str] = []
        pending: dict[int, int] = {}
        trial_results: list[FaultTrialResult] = []
        missed = overruns = backlog = submitted = accepted = cycles_run = 0
        sampler = ResourceSampler(self._gpu_probe)
        sampler.start()
        sched = FixedRateScheduler(period_ns, self._clock, self._sleep)
        sched.start()
        hw_now = 0
        for cycle in range(cfg.cycles):
            if cycle == cfg.warmup_cycles:
                sampler.mark_baseline()
            measured = cycle >= cfg.warmup_cycles
            if self._advance is not None:
                t = self._clock()
                self._advance(period_ns)
                stages.setdefault("sim_advance", []).append(self._clock() - t)
            t0 = self._clock()
            try:
                frame = self._read(cycle)
            except HardwareUnavailableError as exc:
                errors.append(f"cycle {cycle}: hardware unavailable: {exc}")
                break
            t_read = self._clock()
            hw_now = frame.hw_time_ns
            acq = {
                "imu": _acq(frame.imu),
                "depth": _acq(frame.depth),
                "camera": _acq(frame.camera),
                "sonar": _acq(frame.sonar),
            }
            if measured:
                for name, tracker in trackers.items():
                    tracker.update(acq.get(name), hw_now)
            for i, trial in enumerate(self._trials):
                if cycle == trial.at_cycle:
                    trial.inject()
                    pending[i] = cycle
                elif i in pending and (
                    trial.recovered(frame) or cycle - pending[i] > trial.max_recovery_cycles
                ):
                    started = pending.pop(i)
                    ok = cycle - started <= trial.max_recovery_cycles
                    trial_results.append(
                        FaultTrialResult(
                            name=trial.name,
                            injected_at_cycle=started,
                            recovered=ok,
                            recovery_cycles=cycle - started if ok else None,
                            recovery_time_ms=(cycle - started) * period_ns / 1e6 if ok else None,
                        )
                    )
            try:
                out = self._step(frame)
            except Exception as exc:  # recorded as a gate failure, never swallowed silently
                errors.append(f"cycle {cycle}: stack raised {type(exc).__name__}: {exc}")
                out = StackOutput(command=None)
            t_stack = self._clock()
            if out.command is not None:
                ack = self._submit(out.command)
                submitted += 1
                accepted += int(ack.accepted)
            t_done = self._clock()
            cycles_run += 1
            if measured:
                stages.setdefault("acquire", []).append(t_read - t0)
                stages.setdefault("stack", []).append(t_stack - t_read)
                stages.setdefault("submit", []).append(t_done - t_stack)
                for name, ns in out.stage_ns.items():
                    stages.setdefault(f"stack.{name}", []).append(ns)
                cycle_ns.append(t_done - t0)
                if out.command is not None:
                    e2e.append(t_done - t0)
                backlog = max(backlog, out.queue_backlog)
                if t_done - t0 > period_ns:
                    missed += 1
            if not sched.wait_next() and measured:
                overruns += 1
        for i in pending:
            trial = self._trials[i]
            trial_results.append(
                FaultTrialResult(
                    name=trial.name,
                    injected_at_cycle=pending[i],
                    recovered=False,
                    recovery_cycles=None,
                    recovery_time_ms=None,
                    detail="run ended before recovery",
                )
            )
        report = report.model_copy(
            update={
                "cycles_run": cycles_run,
                "missed_deadlines": missed,
                "scheduler_overruns": overruns,
                "skipped_ticks": sched.skipped_ticks,
                "end_to_end_latency": LatencyStats.of(e2e),
                "cycle_latency": LatencyStats.of(cycle_ns),
                "stage_latency": {k: LatencyStats.of(v) for k, v in stages.items()},
                "sensor_age_ms": {k: LatencyStats.of(t.ages_ns) for k, t in trackers.items()},
                "sensor_rates": tuple(t.result(hw_now) for t in trackers.values()),
                "max_queue_backlog": backlog,
                "commands_submitted": submitted,
                "commands_accepted": accepted,
                "commands_rejected": submitted - accepted,
                "stack_errors": tuple(errors),
                "fault_trials": tuple(trial_results),
                "resources": sampler.finish(),
            }
        )
        return evaluate_gate(report, cfg.criteria)
