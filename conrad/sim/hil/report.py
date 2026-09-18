"""HIL gate criteria, report model and evaluation (ch20 HIL gate, ch22 sim-to-real ladder R2).

Gate (ch20): full mission software runs onboard, sensor rates sustained, control deadlines met,
no unacceptable memory growth, fault recovery works, Model 2 inference fits compute budget.
A host run can PASS only the HOST_HIL gate; the target gate stays BLOCKED_EXTERNAL until it runs
on the real onboard computer.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import Field

from conrad.schemas.base import ConradModel
from conrad.sim.hil.metrics import LatencyStats, ResourceReport


class HilMode(str, Enum):
    HOST = "HOST"  # stack on the development host against a simulated vehicle (software-in-the-loop, R1)
    TARGET = "TARGET"  # stack on the real onboard computer (R2)


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED_EXTERNAL = "BLOCKED_EXTERNAL"


class HilGateCriteria(ConradModel):
    """Thresholds come from config (configs/runtime/hil_*.yaml); none are hard-coded in the harness."""

    max_missed_deadline_fraction: float = Field(ge=0, le=1, description="stack work longer than the period")
    max_loop_overrun_fraction: float = Field(
        ge=0, le=1, description="loop tick missed, simulator cost included"
    )
    max_frame_drop_fraction: float = Field(ge=0, le=1)
    max_end_to_end_p99_ms: float = Field(gt=0)
    max_stage_p99_ms: dict[str, float] = Field(default_factory=dict)
    max_python_heap_growth_bytes: int = Field(ge=0)
    max_queue_backlog: int = Field(ge=0)
    require_fault_recovery: bool = True
    min_cycles: int = Field(gt=0)


class FaultTrialResult(ConradModel):
    name: str
    injected_at_cycle: int
    recovered: bool
    recovery_cycles: int | None
    recovery_time_ms: float | None
    detail: str = ""


class SensorRateResult(ConradModel):
    sensor: str
    expected_hz: float
    new_samples: int
    frame_drops: int
    achieved_hz: float | None


class GateCheck(ConradModel):
    name: str
    passed: bool
    measured: float | int | str | None
    limit: float | int | str | None


class HilGateReport(ConradModel):
    report_version: str = "hil_gate.v1"
    mode: HilMode
    status: GateStatus
    blocked_reason: str | None = None
    adapter_name: str
    adapter_is_physical: bool
    robot_config_digest: str | None
    clock_domain: str | None
    host: dict[str, str]
    rate_hz: float
    cycles_planned: int
    cycles_run: int
    missed_deadlines: int
    scheduler_overruns: int = 0
    skipped_ticks: int
    end_to_end_latency: LatencyStats | None = None
    cycle_latency: LatencyStats | None = None
    stage_latency: dict[str, LatencyStats] = Field(default_factory=dict)
    sensor_age_ms: dict[str, LatencyStats] = Field(default_factory=dict)
    sensor_rates: tuple[SensorRateResult, ...] = ()
    max_queue_backlog: int = 0
    commands_submitted: int = 0
    commands_accepted: int = 0
    commands_rejected: int = 0
    stack_errors: tuple[str, ...] = ()
    fault_trials: tuple[FaultTrialResult, ...] = ()
    resources: ResourceReport | None = None
    checks: tuple[GateCheck, ...] = ()

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        return p


def evaluate_gate(report: HilGateReport, criteria: HilGateCriteria) -> HilGateReport:
    """Fill ``checks`` and ``status``. A BLOCKED report stays blocked."""
    if report.status is GateStatus.BLOCKED_EXTERNAL:
        return report
    checks: list[GateCheck] = []

    def add(
        name: str, passed: bool, measured: float | int | str | None, limit: float | int | str | None
    ) -> None:
        checks.append(GateCheck(name=name, passed=passed, measured=measured, limit=limit))

    n = max(report.cycles_run, 1)
    add("min_cycles", report.cycles_run >= criteria.min_cycles, report.cycles_run, criteria.min_cycles)
    missed = report.missed_deadlines / n
    add(
        "missed_deadline_fraction",
        missed <= criteria.max_missed_deadline_fraction,
        missed,
        criteria.max_missed_deadline_fraction,
    )
    overrun = report.scheduler_overruns / n
    add(
        "loop_overrun_fraction",
        overrun <= criteria.max_loop_overrun_fraction,
        overrun,
        criteria.max_loop_overrun_fraction,
    )
    for rate in report.sensor_rates:
        expected = rate.new_samples + rate.frame_drops
        frac = rate.frame_drops / expected if expected else 1.0
        add(
            f"frame_drop_fraction[{rate.sensor}]",
            frac <= criteria.max_frame_drop_fraction,
            frac,
            criteria.max_frame_drop_fraction,
        )
    e2e = report.end_to_end_latency.p99_ms if report.end_to_end_latency else None
    add(
        "end_to_end_p99_ms",
        e2e is not None and e2e <= criteria.max_end_to_end_p99_ms,
        e2e,
        criteria.max_end_to_end_p99_ms,
    )
    for stage, limit in criteria.max_stage_p99_ms.items():
        stats = report.stage_latency.get(stage)
        p99 = stats.p99_ms if stats else None
        add(f"stage_p99_ms[{stage}]", p99 is not None and p99 <= limit, p99, limit)
    growth = report.resources.python_heap_growth_bytes if report.resources else None
    add(
        "python_heap_growth_bytes",
        growth is not None and growth <= criteria.max_python_heap_growth_bytes,
        growth,
        criteria.max_python_heap_growth_bytes,
    )
    add(
        "max_queue_backlog",
        report.max_queue_backlog <= criteria.max_queue_backlog,
        report.max_queue_backlog,
        criteria.max_queue_backlog,
    )
    add("stack_errors", not report.stack_errors, len(report.stack_errors), 0)
    if criteria.require_fault_recovery:
        ok = bool(report.fault_trials) and all(t.recovered for t in report.fault_trials)
        add(
            "fault_recovery", ok, sum(t.recovered for t in report.fault_trials), len(report.fault_trials) or 1
        )
    status = GateStatus.PASS if all(c.passed for c in checks) else GateStatus.FAIL
    return report.model_copy(update={"checks": tuple(checks), "status": status})
