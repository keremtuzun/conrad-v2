"""Fault vocabulary and the record kept for every injection (timing + strength are logged)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FaultType(str, Enum):
    THRUSTER_FAILURE = "THRUSTER_FAILURE"
    THRUSTER_DEGRADATION = "THRUSTER_DEGRADATION"
    THRUSTER_STUCK = "THRUSTER_STUCK"
    SENSOR_DROPOUT = "SENSOR_DROPOUT"
    SENSOR_NOISE = "SENSOR_NOISE"
    SENSOR_BIAS = "SENSOR_BIAS"
    IMU_DRIFT = "IMU_DRIFT"
    LOW_POWER = "LOW_POWER"
    BATTERY_DEGRADATION = "BATTERY_DEGRADATION"
    CURRENT_GUST = "CURRENT_GUST"
    LATENCY = "LATENCY"
    LEAK_SIGNAL = "LEAK_SIGNAL"


@dataclass(frozen=True)
class FaultRecord:
    fault_type: FaultType
    target: str | None
    magnitude: float
    vector: tuple[float, float, float] | None
    injected_at_s: float
    duration_s: float | None


@dataclass
class SensorFaultState:
    dropout_until_s: float = -1.0
    noise_scale: float = 1.0
    extra_bias: float = 0.0
    drift_rate: float = 0.0
    drift_started_s: float = 0.0

    def dropped(self, t_s: float) -> bool:
        return t_s <= self.dropout_until_s

    def bias_at(self, t_s: float) -> float:
        return self.extra_bias + self.drift_rate * max(0.0, t_s - self.drift_started_s)
