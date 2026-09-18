"""Sensor-rendering protocol and the rate/latency-gated measurement channel of the sim adapter."""

from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np

from conrad.schemas.frames import Pose
from conrad.schemas.observation import Observation
from conrad.schemas.robot import SensorConfig
from conrad.schemas.timebase import TimeStamp
from conrad.sim.kernel.faults import SensorFaultState


class SensorRenderer(Protocol):
    def __call__(
        self, sensor_name: str, true_pose: Pose, estimated_pose: Pose | None, timestamp: TimeStamp
    ) -> Observation | None: ...


class _Channel:
    """Rate-gated, latency-delayed scalar/vector sensor channel."""

    def __init__(self, cfg: SensorConfig) -> None:
        n = cfg.sensor_name
        self.cfg = cfg
        self.period_s = 1.0 / float(cfg.rate_hz.require(f"sensors[{n}].rate_hz"))
        self.noise_std = float(cfg.noise_std.require(f"sensors[{n}].noise_std"))
        self.bias = float(cfg.bias.require(f"sensors[{n}].bias"))
        self.drift_per_s = float(cfg.drift_per_s.require(f"sensors[{n}].drift_per_s"))
        self.latency_s = float(cfg.latency_s.require(f"sensors[{n}].latency_s"))
        self.next_due_s = 0.0
        self.queue: deque[tuple[float, float, np.ndarray]] = deque()  # (deliver_s, measured_s, values)
        self.latest: tuple[float, np.ndarray] | None = None
        self.fault = SensorFaultState()
        self.sequence = 0

    def due(self, t_s: float) -> bool:
        if t_s + 1e-9 < self.next_due_s:
            return False
        self.next_due_s = max(self.next_due_s + self.period_s, t_s - 1e-9)
        return True

    def total_bias(self, t_s: float) -> float:
        return self.bias + self.drift_per_s * t_s + self.fault.bias_at(t_s)

    def deliver(self, t_s: float) -> None:
        while self.queue and self.queue[0][0] <= t_s + 1e-9:
            _, measured, values = self.queue.popleft()
            self.latest = (measured, values)
            self.sequence += 1
