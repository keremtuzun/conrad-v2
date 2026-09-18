"""Fault-injection API of SimRobotHardware (ch20/ch21 Fault injection). Timing and strength are logged."""

from __future__ import annotations

import numpy as np

from conrad.sim.kernel.dynamics import SimKernel
from conrad.sim.kernel.faults import FaultRecord, FaultType, SensorFaultState
from conrad.sim.kernel.sensors import _Channel


class FaultInjectionMixin:
    _kernel: SimKernel
    _imu: _Channel | None
    _depth: _Channel | None
    _exteroceptive: dict[str, _Channel]
    _gyro_drift: SensorFaultState
    _leak: bool
    fault_log: list[FaultRecord]

    # -- fault injection ------------------------------------------------------------------------------
    def _channel(self, name: str | None) -> _Channel:
        for ch in (self._imu, self._depth, *self._exteroceptive.values()):
            if ch is not None and ch.cfg.sensor_name == name:
                return ch
        raise KeyError(f"unknown sensor {name!r}")

    def inject_fault(
        self,
        fault_type: FaultType,
        target: str | None = None,
        magnitude: float = 0.0,
        duration_s: float | None = None,
        vector: tuple[float, float, float] | None = None,
    ) -> FaultRecord:
        k, t = self._kernel, self._kernel.t_s
        until = t + duration_s if duration_s is not None else float("inf")
        bank = k.thrusters
        if fault_type in (
            FaultType.THRUSTER_FAILURE,
            FaultType.THRUSTER_DEGRADATION,
            FaultType.THRUSTER_STUCK,
        ):
            if target is None or target not in bank.index:
                raise KeyError(f"unknown thruster {target!r}")
            f = bank.faults[bank.index[target]]
            if fault_type is FaultType.THRUSTER_FAILURE:
                f.effectiveness = 0.0
            elif fault_type is FaultType.THRUSTER_DEGRADATION:
                f.effectiveness = float(np.clip(magnitude, 0.0, 1.0))
            else:
                f.stuck_command = float(np.clip(magnitude, -1.0, 1.0))
        elif fault_type is FaultType.SENSOR_DROPOUT:
            self._channel(target).fault.dropout_until_s = until
        elif fault_type is FaultType.SENSOR_NOISE:
            self._channel(target).fault.noise_scale = max(magnitude, 0.0)
        elif fault_type is FaultType.SENSOR_BIAS:
            self._channel(target).fault.extra_bias = magnitude
        elif fault_type is FaultType.IMU_DRIFT:
            self._gyro_drift.drift_rate = magnitude
            self._gyro_drift.drift_started_s = t
        elif fault_type is FaultType.LOW_POWER:
            cap = k.params.battery_capacity_j * k.battery_capacity_scale
            k.energy_used_j = max(k.energy_used_j, (1.0 - float(np.clip(magnitude, 0.0, 1.0))) * cap)
        elif fault_type is FaultType.BATTERY_DEGRADATION:
            k.battery_capacity_scale = float(np.clip(magnitude, 1e-3, 1.0))
        elif fault_type is FaultType.CURRENT_GUST:
            if vector is None:
                raise ValueError("CURRENT_GUST requires a world velocity vector")
            k.set_gust(np.asarray(vector, dtype=np.float64), until)
        elif fault_type is FaultType.LATENCY:
            if target is None:
                bank.global_extra_latency_s = max(magnitude, 0.0)
            else:
                bank.faults[bank.index[target]].extra_latency_s = max(magnitude, 0.0)
        elif fault_type is FaultType.LEAK_SIGNAL:
            self._leak = True
        record = FaultRecord(fault_type, target, magnitude, vector, t, duration_s)
        self.fault_log.append(record)
        return record
