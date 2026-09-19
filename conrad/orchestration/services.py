"""Shared runtime services and supervised module execution. DEPLOYMENT PLANE.

``ModuleRunner.call`` is the only place a deployment module is invoked: a raised exception is never
swallowed into AVAILABLE. It marks the module UNAVAILABLE in the HealthRegistry, emits FAULT_DETECTED, and
the module is not called again in this run (the supervisor decides whether that is critical).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar
from uuid import UUID

from conrad.persistence.object_store import ObjectStore
from conrad.persistence.repository import Repository
from conrad.runtime.event_log import EventLog
from conrad.runtime.health import HealthRegistry
from conrad.schemas.belief import Availability
from conrad.schemas.events import EventType, Severity
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import ProvenanceRecord

T = TypeVar("T")
MODULE_PREFIX = "conrad.orchestration"


class InjectedModuleCrash(RuntimeError):
    """Deterministic crash injected by configuration (tests / INT benchmarks)."""


@dataclass
class RuntimeServices:
    ids: IdFactory
    log: EventLog
    repo: Repository
    store: ObjectStore
    health: HealthRegistry
    run_id: UUID
    mission_id: UUID
    clock_ns: Callable[[], int]
    provenance_buffer: list[ProvenanceRecord] = field(default_factory=list)

    def emit(self, event: EventType, module: str, trace: UUID, payload: dict[str, Any], **kw: Any) -> None:
        self.log.emit(event, module, trace, payload, **kw)

    def flush_provenance(self) -> None:
        if self.provenance_buffer:
            self.repo.put_provenance(self.run_id, self.provenance_buffer)
            self.provenance_buffer = []


class ModuleRunner:
    def __init__(self, services: RuntimeServices, crash_at_s: dict[str, float | None]) -> None:
        self.s = services
        self._crash_at = {k: v for k, v in crash_at_s.items() if v is not None}
        self.failed: set[str] = set()

    def available(self, name: str) -> bool:
        return name not in self.failed

    def call(self, name: str, fn: Callable[[], T], trace: UUID | None = None) -> T | None:
        """Run ``fn`` as module ``name``; None if the module is (or just became) UNAVAILABLE."""
        if name in self.failed:
            return None
        trace = trace or self.s.run_id
        now_s = self.s.clock_ns() / 1e9
        try:
            crash = self._crash_at.get(name)
            if crash is not None and now_s >= crash:
                self._crash_at.pop(name)
                self.s.emit(
                    EventType.FAULT_INJECTED,
                    f"{MODULE_PREFIX}.faults",
                    trace,
                    {"module": name, "fault": "MODULE_CRASH", "t_s": round(now_s, 3)},
                )
                raise InjectedModuleCrash(f"{name}: injected crash at t={now_s:.2f}s")
            out = fn()
        except Exception as exc:  # recorded, module marked UNAVAILABLE; never swallowed into AVAILABLE
            self.failed.add(name)
            self.s.health.failure(name, f"{type(exc).__name__}: {exc}")
            self.s.emit(
                EventType.FAULT_DETECTED,
                f"{MODULE_PREFIX}.supervision",
                trace,
                {"module": name, "error_type": type(exc).__name__, "error": str(exc)[:300]},
                severity=Severity.ERROR,
                availability=Availability.UNAVAILABLE,
            )
            return None
        self.s.health.heartbeat(name)
        return out

    def heartbeat_idle(self, name: str) -> None:
        """A healthy module with nothing to do this cycle is still alive."""
        if name not in self.failed:
            self.s.health.heartbeat(name)
