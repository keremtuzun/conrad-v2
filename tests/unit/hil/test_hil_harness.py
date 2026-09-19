"""HIL harness against three RHIs: a Python kernel, Unity-over-ZeroMQ (mock player) and a blocked target."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conrad_mock_unity_server import MockUnityServer
from conrad_unity_testkit import make_command, sim_robot_config

from conrad.adapters.unity import FaultInjectionRequest, FaultType, UnityBridgeConfig, UnityRobotHardware
from conrad.robotics.hardware.interface import PhysicalRobotHardware, RobotHardwareInterface
from conrad.runtime.command_gateway import CommandGateway
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import AllocatedCommand, CommandAck
from conrad.settings import CommandMode, ExecutionLane, RuntimeSettings
from conrad.sim.hil import (
    NOT_AVAILABLE,
    FaultTrial,
    FixedRateScheduler,
    GateStatus,
    HilConfig,
    HilGateCriteria,
    HilHarness,
    HilMode,
    SensorFrame,
    StackOutput,
    load_hil_config,
)

CONFIG = sim_robot_config()
LENIENT = HilGateCriteria(
    max_missed_deadline_fraction=0.5,
    max_loop_overrun_fraction=0.5,
    max_frame_drop_fraction=0.5,
    max_end_to_end_p99_ms=250.0,
    max_stage_p99_ms={"acquire": 200.0},
    max_python_heap_growth_bytes=64 * 1024 * 1024,
    max_queue_backlog=4,
    require_fault_recovery=True,
    min_cycles=20,
)


def _no_gpu() -> tuple[str, int | None]:
    return NOT_AVAILABLE + ":no_cuda_device", None


def _depth_hold(hw: RobotHardwareInterface, ids: IdFactory, mission, run, target_m: float = 2.5):
    def step(frame: SensorFrame) -> StackOutput:
        if frame.depth is None:
            return StackOutput(command=None, stage_ns={"control": 0})
        u = max(-0.5, min(0.5, 0.8 * (target_m - frame.depth.depth_m)))
        values = {t.thruster_id: (u if t.thruster_id.startswith("V") else 0.0) for t in CONFIG.thrusters}
        return StackOutput(
            command=make_command(ids, CONFIG, mission, run, hw.now_ns(), values), stage_ns={"control": 1000}
        )

    return step


def _never_submit(command: AllocatedCommand) -> CommandAck:
    raise AssertionError("the stack emits no commands, so nothing may reach the gateway")


def _gateway(hw: RobotHardwareInterface, mission, run) -> CommandGateway:
    return CommandGateway(
        hw, CONFIG, RuntimeSettings(command_mode=CommandMode.SIMULATED), ExecutionLane.HIL, mission, run
    )


def test_host_hil_against_python_kernel(ids: IdFactory, tmp_path: Path) -> None:
    kernel = pytest.importorskip("conrad.sim.kernel")
    hw = kernel.build_sim_hardware(CONFIG, seed=3)
    mission, run = ids.new(), ids.new()
    gateway = _gateway(hw, mission, run)
    trial = FaultTrial(
        name="depth_dropout",
        at_cycle=25,
        inject=lambda: hw.inject_fault(kernel.FaultType.SENSOR_DROPOUT, target="depth", duration_s=0.1),
        recovered=lambda f: f.depth is not None and f.hw_time_ns - f.depth.timestamp.time_ns < 60_000_000,
        max_recovery_cycles=30,
    )
    cfg = HilConfig(rate_hz=50.0, cycles=70, warmup_cycles=5, sensor_rates_hz={"imu": 50.0}, criteria=LENIENT)
    report = HilHarness(
        hw,
        _depth_hold(hw, ids, mission, run),
        gateway.submit,
        cfg,
        advance=lambda ns: hw.advance(ns / 1e9),
        fault_trials=(trial,),
        gpu_probe=_no_gpu,
    ).run()
    assert report.cycles_run == 70 and report.adapter_name == hw.adapter_name
    assert report.commands_submitted > 0 and report.commands_accepted == report.commands_submitted
    assert report.stage_latency["acquire"].count == 65 and "stack.control" in report.stage_latency
    assert report.end_to_end_latency is not None and report.end_to_end_latency.p99_ms is not None
    assert report.fault_trials[0].recovered and report.fault_trials[0].recovery_cycles is not None
    assert report.resources is not None and report.resources.gpu_status.startswith(NOT_AVAILABLE)
    assert report.status is GateStatus.PASS, [c for c in report.checks if not c.passed]
    doc = json.loads(report.write(tmp_path / "hil_gate.json").read_text(encoding="utf-8"))
    assert doc["status"] == "PASS" and doc["mode"] == "HOST"


def test_host_hil_against_unity_bridge_over_socket(ids: IdFactory) -> None:
    thrusters = tuple(t.thruster_id for t in CONFIG.thrusters)
    with MockUnityServer(CONFIG.content_digest(), thrusters) as srv:
        mission, run = ids.new(), ids.new()
        with UnityRobotHardware(
            UnityBridgeConfig(control_endpoint=srv.control_endpoint), CONFIG, ids, mission, run
        ) as hw:
            hw.connect()

            def inject_imu_dropout() -> None:
                hw.inject_fault(
                    FaultInjectionRequest(
                        fault_id="d1",
                        fault_type=FaultType.IMU_DROPOUT,
                        start_time_ns=hw.now_ns(),
                        duration_ns=50_000_000,
                    )
                )

            def advance(period_ns: int) -> None:
                hw.step(period_ns)

            trial = FaultTrial(
                name="imu_dropout",
                at_cycle=15,
                inject=inject_imu_dropout,
                recovered=lambda f: (
                    f.imu is not None and f.hw_time_ns - f.imu.timestamp.time_ns <= 20_000_000
                ),
                max_recovery_cycles=20,
            )
            cfg = HilConfig(
                rate_hz=100.0,
                cycles=50,
                warmup_cycles=5,
                sensor_rates_hz={"imu": 100.0, "depth": 100.0},
                criteria=LENIENT,
            )
            report = HilHarness(
                hw,
                _depth_hold(hw, ids, mission, run),
                _gateway(hw, mission, run).submit,
                cfg,
                advance=advance,
                fault_trials=(trial,),
                gpu_probe=_no_gpu,
            ).run()
        assert report.adapter_name == "unity_v2" and report.clock_domain == "SIM"
        assert report.commands_accepted == report.commands_submitted > 0
        assert len(srv.received_commands) == report.commands_submitted
        imu_rate = next(r for r in report.sensor_rates if r.sensor == "imu")
        assert imu_rate.frame_drops > 0  # the injected dropout is visible as dropped frames
        assert report.fault_trials[0].recovered
        assert report.sensor_age_ms["imu"].p50_ms == pytest.approx(10.0)  # the mock's 10 ms IMU latency


def test_target_mode_is_blocked_external_off_target() -> None:
    # PhysicalRobotHardware raises on every call, so a blocked run provably never touches hardware.
    cfg = HilConfig(
        mode=HilMode.TARGET,
        rate_hz=50.0,
        cycles=10,
        target_platform_id="Linux-aarch64-jetson",
        criteria=LENIENT,
    )
    report = HilHarness(
        PhysicalRobotHardware(),
        lambda f: StackOutput(None),
        _never_submit,
        cfg,
        platform_id=lambda: "Windows-AMD64-devbox",
    ).run()
    assert report.status is GateStatus.BLOCKED_EXTERNAL and report.cycles_run == 0
    assert report.adapter_is_physical and "onboard computer" in (report.blocked_reason or "")
    unset = HilConfig(mode=HilMode.TARGET, rate_hz=50.0, cycles=10, criteria=LENIENT)
    assert (
        HilHarness(PhysicalRobotHardware(), lambda f: StackOutput(None), _never_submit, unset).run().status
        is GateStatus.BLOCKED_EXTERNAL
    )


def test_gate_fails_on_stack_errors_missing_recovery_and_backlog(ids: IdFactory) -> None:
    kernel = pytest.importorskip("conrad.sim.kernel")
    hw = kernel.build_sim_hardware(CONFIG, seed=4)

    def broken(frame: SensorFrame) -> StackOutput:
        if frame.cycle == 7:
            raise RuntimeError("estimator diverged")
        return StackOutput(command=None, queue_backlog=frame.cycle)

    trial = FaultTrial(
        "never", at_cycle=5, inject=lambda: None, recovered=lambda f: False, max_recovery_cycles=3
    )
    cfg = HilConfig(rate_hz=200.0, cycles=25, criteria=LENIENT)
    report = HilHarness(
        hw,
        broken,
        _never_submit,
        cfg,
        advance=lambda ns: hw.advance(ns / 1e9),
        fault_trials=(trial,),
        gpu_probe=_no_gpu,
    ).run()
    failed = {c.name for c in report.checks if not c.passed}
    assert report.status is GateStatus.FAIL
    assert {"stack_errors", "fault_recovery", "max_queue_backlog", "end_to_end_p99_ms"} <= failed
    assert "estimator diverged" in report.stack_errors[0]


def test_fixed_rate_scheduler_counts_overruns_with_injected_clock() -> None:
    now = [0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += round(s * 1e9)

    sched = FixedRateScheduler(10_000_000, clock=lambda: now[0], sleep=sleep)
    sched.start()
    now[0] += 3_000_000
    assert sched.wait_next() and slept[-1] == pytest.approx(0.007)
    now[0] += 35_000_000  # overrun by 3.5 periods
    assert not sched.wait_next()
    assert sched.skipped_ticks == 3
    now[0] += 1_000_000
    assert sched.wait_next() and now[0] == 50_000_000


def test_hil_configs_load() -> None:
    settings, host = load_hil_config("configs/runtime/hil_host.yaml")
    assert host.mode is HilMode.HOST and settings.run.lane is ExecutionLane.HIL
    assert host.criteria.max_end_to_end_p99_ms == 20.0
    _, target = load_hil_config("configs/runtime/hil_target.yaml")
    assert target.mode is HilMode.TARGET and target.target_platform_id is None
