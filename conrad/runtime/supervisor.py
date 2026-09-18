"""Single-process runtime lifecycle state machine (ch34 Runtime process model).

BOOT -> SELF_TEST -> CONFIGURED -> CONNECTING -> READY -> RUNNING
RUNNING <-> DEGRADED;  RUNNING | DEGRADED -> SAFE_HOLD -> STOPPING -> STOPPED;  any -> FAILED

READY allows simulation/replay. Hardware RUNNING additionally needs the hardware adapter, the
safety supervisor, the command gateway and the declared HIL evidence (SS-10).

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any
from uuid import UUID

from conrad.runtime.health import HealthRegistry
from conrad.schemas.events import EventType, Severity
from conrad.settings import CommandMode, ConradSettings, ExecutionLane


class RuntimeState(str, Enum):
    BOOT = "BOOT"
    SELF_TEST = "SELF_TEST"
    CONFIGURED = "CONFIGURED"
    CONNECTING = "CONNECTING"
    READY = "READY"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    SAFE_HOLD = "SAFE_HOLD"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


_ALLOWED: dict[RuntimeState, frozenset[RuntimeState]] = {
    RuntimeState.BOOT: frozenset({RuntimeState.SELF_TEST}),
    RuntimeState.SELF_TEST: frozenset({RuntimeState.CONFIGURED}),
    RuntimeState.CONFIGURED: frozenset({RuntimeState.CONNECTING}),
    RuntimeState.CONNECTING: frozenset({RuntimeState.READY}),
    RuntimeState.READY: frozenset({RuntimeState.RUNNING, RuntimeState.STOPPING}),
    RuntimeState.RUNNING: frozenset({RuntimeState.DEGRADED, RuntimeState.SAFE_HOLD}),
    RuntimeState.DEGRADED: frozenset({RuntimeState.RUNNING, RuntimeState.SAFE_HOLD}),
    RuntimeState.SAFE_HOLD: frozenset({RuntimeState.STOPPING, RuntimeState.RUNNING}),
    RuntimeState.STOPPING: frozenset({RuntimeState.STOPPED}),
    RuntimeState.STOPPED: frozenset(),
    RuntimeState.FAILED: frozenset(),
}

OPERATOR_ACTIONS = frozenset({"start", "pause", "safe_hold", "resume", "stop"})


class IllegalTransitionError(RuntimeError):
    pass


class HardwareGateError(RuntimeError):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__("hardware RUNNING refused: " + ", ".join(reasons))
        self.reasons = reasons


Emit = Callable[..., Any]


class RuntimeSupervisor:
    def __init__(
        self,
        settings: ConradSettings,
        health: HealthRegistry,
        emit: Emit,
        run_id: UUID,
        safe_hold_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.health = health
        self._emit = emit
        self._run_id = run_id
        self._safe_hold_hook = safe_hold_hook
        self.state = RuntimeState.BOOT
        self.history: list[tuple[RuntimeState, str]] = [(RuntimeState.BOOT, "boot")]
        self.hardware_ready = {"adapter": False, "safety_supervisor": False, "command_gateway": False}

    def _go(self, new: RuntimeState, reason: str) -> None:
        if new is not RuntimeState.FAILED and new not in _ALLOWED[self.state]:
            raise IllegalTransitionError(f"{self.state.value} -> {new.value} is not permitted")
        old, self.state = self.state, new
        self.history.append((new, reason))
        self._emit(
            EventType.STATE_CHANGED,
            "conrad.runtime.supervisor",
            self._run_id,
            {"from": old.value, "to": new.value, "reason": reason},
            severity=Severity.CRITICAL if new is RuntimeState.FAILED else Severity.INFO,
        )

    def self_test(self, checks: list[tuple[str, Callable[[], None]]]) -> list[str]:
        """Run the SELF_TEST checks. Any failure moves the runtime to FAILED and is returned."""
        self._go(RuntimeState.SELF_TEST, "self-test started")
        problems: list[str] = []
        for name, check in checks:
            try:
                check()
            except Exception as exc:  # reported, never swallowed: the run fails closed
                problems.append(f"{name}: {exc}")
        if problems:
            self.fail("self-test failed: " + "; ".join(problems))
            return problems
        self._go(RuntimeState.CONFIGURED, "self-test passed")
        return problems

    def connect(self, connect_adapters: Callable[[], None]) -> None:
        self._go(RuntimeState.CONNECTING, "connecting adapters")
        try:
            connect_adapters()
        except Exception as exc:
            self.fail(f"adapter connection failed: {exc}")
            raise
        self._go(RuntimeState.READY, "adapters connected")

    def hardware_gate_reasons(self) -> list[str]:
        rt = self.settings.runtime
        reasons = []
        if rt.command_mode is not CommandMode.HARDWARE:
            reasons.append("command_mode is not hardware")
        if not rt.hardware_enable:
            reasons.append("local hardware enable is false")
        if self.settings.run.lane not in (ExecutionLane.PHYSICAL, ExecutionLane.HIL):
            reasons.append("lane is not hil/physical")
        if not rt.hil_evidence_ref:
            reasons.append("HIL evidence reference missing")
        reasons += [f"{name} not healthy" for name, ok in self.hardware_ready.items() if not ok]
        return reasons

    def start(self, hardware: bool = False) -> None:
        if hardware:
            reasons = self.hardware_gate_reasons()
            if reasons:
                raise HardwareGateError(reasons)
        self._go(RuntimeState.RUNNING, "hardware run" if hardware else "simulation run")

    def tick(self) -> RuntimeState:
        """Evaluate health. A critical failure reaches SAFE_HOLD without any new command (SS-06)."""
        if self.state not in (RuntimeState.RUNNING, RuntimeState.DEGRADED):
            return self.state
        critical = self.health.critical_failures()
        if critical:
            self.safe_hold("critical module failure: " + ", ".join(critical))
        elif self.health.degraded() and self.state is RuntimeState.RUNNING:
            self._go(RuntimeState.DEGRADED, "degraded: " + ", ".join(self.health.degraded()))
        elif not self.health.degraded() and self.state is RuntimeState.DEGRADED:
            self._go(RuntimeState.RUNNING, "recovered")
        return self.state

    def safe_hold(self, reason: str) -> None:
        if self.state is RuntimeState.SAFE_HOLD:
            return
        self._go(RuntimeState.SAFE_HOLD, reason)
        self._emit(
            EventType.SAFE_HOLD_ENTERED,
            "conrad.runtime.supervisor",
            self._run_id,
            {"reason": reason},
            severity=Severity.ERROR,
        )
        if self._safe_hold_hook is not None:
            self._safe_hold_hook(reason)

    def operator(self, action: str, operator_id: str, reason: str) -> None:
        """Human actions are append-only events and never bypass the safety gateway (ch36)."""
        if action not in OPERATOR_ACTIONS:
            raise ValueError(f"unknown operator action {action!r}")
        self._emit(
            EventType.OPERATOR_ACTION,
            "conrad.runtime.supervisor",
            self._run_id,
            {
                "action": action,
                "operator": operator_id,
                "reason": reason,
                "state": self.state.value,
                "health": {k: v.value for k, v in self.health.snapshot().items()},
            },
        )
        if action == "start":
            self.start(hardware=False)
        elif action in ("pause", "safe_hold"):
            self.safe_hold(f"operator {action}: {reason}")
        elif action == "resume":
            if self.health.critical_failures():
                raise IllegalTransitionError("cannot resume while a critical module is unhealthy")
            self._go(RuntimeState.RUNNING, f"operator resume: {reason}")
        elif action == "stop":
            self.stop(f"operator stop: {reason}")

    def stop(self, reason: str) -> None:
        if self.state in (RuntimeState.RUNNING, RuntimeState.DEGRADED):
            self.safe_hold(reason)
        if self.state in (RuntimeState.SAFE_HOLD, RuntimeState.READY):
            self._go(RuntimeState.STOPPING, reason)
            self._go(RuntimeState.STOPPED, reason)
            self._emit(
                EventType.RUN_TERMINATED, "conrad.runtime.supervisor", self._run_id, {"reason": reason}
            )

    def fail(self, reason: str) -> None:
        self._go(RuntimeState.FAILED, reason)
        self._emit(
            EventType.FAULT_DETECTED,
            "conrad.runtime.supervisor",
            self._run_id,
            {"reason": reason},
            severity=Severity.CRITICAL,
        )
