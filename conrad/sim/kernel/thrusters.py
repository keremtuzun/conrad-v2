"""Thruster bank: T = k u|u|, deadzone, asymmetry, saturation, latency, first-order lag, noise, faults."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from conrad.sim.kernel.params import ThrusterParams


@dataclass
class ThrusterFault:
    effectiveness: float = 1.0  # 0 = complete failure
    stuck_command: float | None = None
    extra_latency_s: float = 0.0


def static_thrust(p: ThrusterParams, u: float) -> float:
    """Steady-state thrust for a normalized command (the model the allocator inverts)."""
    u = max(-1.0, min(1.0, u))
    if abs(u) < p.deadzone:
        return 0.0
    if u > 0:
        return min(p.k * u * u, p.max_forward_n)
    k_rev = p.k * p.max_reverse_n / p.max_forward_n if p.max_forward_n > 0 else p.k
    return -min(k_rev * u * u, p.max_reverse_n)


class ThrusterBank:
    def __init__(self, params: tuple[ThrusterParams, ...], noise_fraction: float) -> None:
        self.params = params
        self.noise_fraction = noise_fraction
        n = len(params)
        self.thrust = np.zeros(n)
        self.applied_command = np.zeros(n)
        self.requested = np.zeros(n)
        self.faults = [ThrusterFault() for _ in range(n)]
        self.global_extra_latency_s = 0.0
        self._queues: list[deque[tuple[float, float]]] = [deque() for _ in range(n)]
        self.index = {p.thruster_id: i for i, p in enumerate(params)}

    def command(self, t_s: float, commands: dict[str, float]) -> None:
        for tid, u in commands.items():
            i = self.index[tid]
            self.requested[i] = u
            delay = self.params[i].latency_s + self.faults[i].extra_latency_s + self.global_extra_latency_s
            self._queues[i].append((t_s + delay, float(u)))

    def zero(self, t_s: float) -> None:
        self.command(t_s, {p.thruster_id: 0.0 for p in self.params})

    def step(self, t_s: float, dt: float, rng: np.random.Generator) -> np.ndarray:
        for i, p in enumerate(self.params):
            q = self._queues[i]
            while q and q[0][0] <= t_s + 1e-12:
                self.applied_command[i] = q.popleft()[1]
            fault = self.faults[i]
            u = fault.stuck_command if fault.stuck_command is not None else self.applied_command[i]
            target = static_thrust(p, float(u)) * fault.effectiveness
            alpha = 1.0 - math.exp(-dt / p.tau_s) if p.tau_s > 0 else 1.0
            self.thrust[i] += alpha * (target - self.thrust[i])
        noise = 1.0 + self.noise_fraction * rng.standard_normal(len(self.params))
        return self.thrust * noise
