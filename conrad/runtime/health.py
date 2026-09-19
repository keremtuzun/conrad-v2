"""Availability / freshness / deadline tracking for runtime modules (ch28 Runtime contract, ch34).

An exception inside a module is never swallowed into AVAILABLE: a failed heartbeat marks the
module UNAVAILABLE, and silence beyond its freshness window marks it STALE.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from conrad.schemas.belief import Availability


@dataclass
class ModuleHealth:
    name: str
    version: str
    freshness_s: float
    critical: bool
    last_ok_ns: int | None = None
    state: Availability = Availability.UNAVAILABLE
    last_error: str | None = None
    missed_deadlines: int = 0


class HealthRegistry:
    def __init__(self, clock_ns: Callable[[], int]) -> None:
        self._clock_ns = clock_ns
        self._modules: dict[str, ModuleHealth] = {}

    def register(self, name: str, version: str, freshness_s: float, critical: bool) -> None:
        self._modules[name] = ModuleHealth(
            name=name, version=version, freshness_s=freshness_s, critical=critical
        )

    def heartbeat(self, name: str, degraded: bool = False) -> None:
        m = self._modules[name]
        m.last_ok_ns = self._clock_ns()
        m.state = Availability.DEGRADED if degraded else Availability.AVAILABLE
        m.last_error = None

    def failure(self, name: str, error: str) -> None:
        m = self._modules[name]
        m.state = Availability.UNAVAILABLE
        m.last_error = error

    def missed_deadline(self, name: str) -> None:
        self._modules[name].missed_deadlines += 1

    def availability(self, name: str) -> Availability:
        m = self._modules[name]
        if m.state in (Availability.AVAILABLE, Availability.DEGRADED) and (
            m.last_ok_ns is None or (self._clock_ns() - m.last_ok_ns) / 1e9 > m.freshness_s
        ):
            return Availability.STALE
        return m.state

    def snapshot(self) -> dict[str, Availability]:
        return {name: self.availability(name) for name in self._modules}

    def critical_failures(self) -> list[str]:
        return [
            name
            for name, m in self._modules.items()
            if m.critical and self.availability(name) in (Availability.UNAVAILABLE, Availability.STALE)
        ]

    def degraded(self) -> list[str]:
        return [name for name in self._modules if self.availability(name) is not Availability.AVAILABLE]

    def module(self, name: str) -> ModuleHealth:
        return self._modules[name]
