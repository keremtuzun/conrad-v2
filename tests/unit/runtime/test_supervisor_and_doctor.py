from __future__ import annotations

import pytest

from conrad.runtime.doctor import run_doctor
from conrad.runtime.health import HealthRegistry
from conrad.runtime.supervisor import (
    HardwareGateError,
    IllegalTransitionError,
    RuntimeState,
    RuntimeSupervisor,
)
from conrad.schemas.belief import Availability
from conrad.schemas.events import EventType
from conrad.schemas.ids import IdFactory
from conrad.settings import ConradSettings


class Clock:
    t = 0

    def __call__(self) -> int:
        return self.t


def build(settings: ConradSettings | None = None):
    clock = Clock()
    health = HealthRegistry(clock)
    health.register("model2", "v", freshness_s=1.0, critical=True)
    health.register("model2e", "v", freshness_s=1.0, critical=False)
    events: list[tuple] = []
    holds: list[str] = []
    sup = RuntimeSupervisor(
        settings or ConradSettings(),
        health,
        lambda *a, **k: events.append((a, k)),
        IdFactory(1).new(),
        holds.append,
    )
    return sup, health, clock, events, holds


def ready(sup: RuntimeSupervisor) -> None:
    assert sup.self_test([("ok", lambda: None)]) == []
    sup.connect(lambda: None)
    assert sup.state is RuntimeState.READY


def test_lifecycle_order_is_enforced() -> None:
    sup, *_ = build()
    with pytest.raises(IllegalTransitionError):
        sup.start()
    ready(sup)
    sup.start()
    assert sup.state is RuntimeState.RUNNING
    sup.stop("done")
    assert [s for s, _ in sup.history][-3:] == [
        RuntimeState.SAFE_HOLD,
        RuntimeState.STOPPING,
        RuntimeState.STOPPED,
    ]


def test_failed_self_test_fails_closed() -> None:
    sup, *_ = build()

    def boom() -> None:
        raise RuntimeError("robot config checksum mismatch")

    problems = sup.self_test([("robot_config", boom)])
    assert problems and sup.state is RuntimeState.FAILED
    with pytest.raises(IllegalTransitionError):
        sup.connect(lambda: None)


def test_ss06_critical_failure_reaches_safe_hold_without_commands() -> None:
    sup, health, _clock, events, holds = build()
    ready(sup)
    sup.start()
    health.heartbeat("model2")
    health.heartbeat("model2e")
    assert sup.tick() is RuntimeState.RUNNING
    health.failure("model2e", "crashed")  # non-critical child: degrade, keep running (failure isolation)
    assert sup.tick() is RuntimeState.DEGRADED
    health.failure("model2", "database unavailable")
    assert sup.tick() is RuntimeState.SAFE_HOLD and holds
    types = [a[0] for a, _ in events]
    assert EventType.SAFE_HOLD_ENTERED in types and EventType.COMMAND_SENT not in types
    with pytest.raises(IllegalTransitionError):
        sup.operator("resume", "op-1", "try again")


def test_silence_becomes_stale_not_available() -> None:
    _sup, health, clock, *_ = build()
    health.heartbeat("model2")
    clock.t = int(5e9)
    assert health.availability("model2") is Availability.STALE
    assert "model2" in health.critical_failures()


def test_hardware_running_requires_every_gate() -> None:
    sup, *_ = build()
    ready(sup)
    with pytest.raises(HardwareGateError) as exc:
        sup.start(hardware=True)
    assert len(exc.value.reasons) >= 5 and sup.state is RuntimeState.READY


def test_operator_actions_are_logged() -> None:
    sup, _health, _, events, _ = build()
    ready(sup)
    sup.operator("start", "op-7", "begin mission")
    sup.operator("safe_hold", "op-7", "visual check")
    logged = [k or a for a, k in events if a[0] is EventType.OPERATOR_ACTION]
    assert len(logged) == 2 and sup.state is RuntimeState.SAFE_HOLD
    with pytest.raises(ValueError):
        sup.operator("override_thrusters", "op-7", "no")


def test_doctor_passes_default_and_fails_physical_example() -> None:
    assert run_doctor("configs/runtime/default.yaml").ok
    bad = run_doctor("configs/runtime/physical_example.yaml")
    failed = {c.name for c in bad.checks if c.status == "FAIL"}
    assert not bad.ok and {"robot_config", "command_mode", "adapters"} <= failed
    assert not run_doctor("configs/runtime/does_not_exist.yaml").ok
