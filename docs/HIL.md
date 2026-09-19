# HIL harness

`conrad.sim.hil` runs a deployment-equivalent stack loop against **any** `RobotHardwareInterface` and produces a
HIL gate report (spec ch20 "HIL gate", ch22 ladder R1/R2).

```
hardware (SimRobotHardware | UnityRobotHardware | PhysicalRobotHardware)
   --get_*()-->  stack_step(SensorFrame) -> StackOutput(command, stage_ns, queue_backlog)
   --command-->  submit(command)   # the Command Gateway; the harness never calls send()
```

## Modes

| Mode | Where it runs | Status |
|---|---|---|
| `HOST` (ladder R1, software-in-the-loop) | development host, vehicle simulated | runs today against `conrad.sim.kernel` and against Unity over the bridge (tested with the mock player) |
| `TARGET` (ladder R2, HIL) | the real onboard computer, vehicle simulated | **BLOCKED_EXTERNAL**: the harness refuses to run unless `default_platform_id()` equals `sim.hil.target_platform_id`, and reports `BLOCKED_EXTERNAL` without touching the hardware |

## What is measured

| Metric | How |
|---|---|
| per-stage latency | `acquire` (all RHI reads), `stack`, `submit` (gateway + transport), `stack.<name>` from the stack's own `stage_ns`, `sim_advance` reported separately |
| end-to-end observation -> command | wall time from the start of sensor reads to the gateway ack, cycles with a command |
| missed deadlines | stack work longer than the period |
| loop overruns / skipped ticks | fixed-rate scheduler with absolute deadlines, no catch-up bursts |
| sensor rates / frame drops | per sensor, a cycle counts as a drop when its newest sample is older than `stale_tolerance / rate` |
| sensor age | `hw.now_ns() - acquisition time`, in the hardware clock |
| CPU | process CPU time / wall time (`time.process_time`) |
| RAM | Python heap growth after warm-up (`tracemalloc`); RSS only where `/proc/self/statm` exists, otherwise `NOT_AVAILABLE` |
| GPU | `NOT_AVAILABLE:no_cuda_device` unless CUDA is present |
| queue backlog | maximum `StackOutput.queue_backlog` |
| fault recovery | `FaultTrial(inject, recovered, max_recovery_cycles)`: cycles and ms until recovered |

Gate thresholds are configuration (`sim.hil.criteria` in `configs/runtime/hil_host.yaml`). They are
ENGINEERING_ESTIMATE values, not spec values. The report is JSON (`HilGateReport.write`); `status` is `PASS`,
`FAIL` or `BLOCKED_EXTERNAL`, and every check lists measured vs limit.

## Wiring the real stack (integrator)

```python
from conrad.sim.hil import FaultTrial, HilHarness, StackOutput, load_hil_config
from conrad.sim.kernel import build_sim_hardware, FaultType

settings, cfg = load_hil_config("configs/runtime/hil_host.yaml")
hw = build_sim_hardware(robot_config, seed=settings.run.seed)
gateway = CommandGateway(hw, robot_config, settings.runtime, settings.run.lane, mission_id, run_id)


def stack_step(frame):  # the real estimator -> navigation -> control -> allocation chain
    command, timings, backlog = conrad_stack.tick(frame)
    return StackOutput(command=command, stage_ns=timings, queue_backlog=backlog)


trials = (
    FaultTrial(
        "depth_dropout",
        at_cycle=500,
        inject=lambda: hw.inject_fault(FaultType.SENSOR_DROPOUT, target="depth", duration_s=0.5),
        recovered=lambda f: f.depth is not None,
        max_recovery_cycles=100,
    ),
)
report = HilHarness(
    hw, stack_step, gateway.submit, cfg, advance=lambda ns: hw.advance(ns / 1e9), fault_trials=trials
).run()
report.write("artifacts/hil/host_gate.json")
```

For Unity in lock-step mode pass `advance=hw.step` with a connected `UnityRobotHardware`. For target-HIL, run the
same script **on the onboard computer** with `configs/runtime/hil_target.yaml`, after setting
`target_platform_id` to the value printed on that machine by:

```
python -m uv run python -c "from conrad.sim.hil import default_platform_id; print(default_platform_id())"
```

Tests: `python -m uv run pytest tests/unit/hil -q`. Every socket test has a ZeroMQ receive timeout and a
`faulthandler` hard timeout, so a hang aborts with stack dumps instead of blocking.
