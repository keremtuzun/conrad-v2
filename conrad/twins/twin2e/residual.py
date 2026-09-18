"""Optional learned-residual hook F_learned-residual (ch13). DISABLED by default.

ch33 freeze: no learned Twin2E component is permitted before calibration data exists. Enabling the
hook without a calibration-data reference raises; enabling it without an explicitly supplied
residual function also raises. Nothing learned ships here.

implementation_status: OPEN_BLOCKED (awaiting calibration data)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from conrad.twins.twin2e.config import LearnedResidualConfig

ResidualFn = Callable[[str, np.ndarray, float], np.ndarray]
"""(field_name, field_values, dt_s) -> additive residual with the field's shape."""


class LearnedResidualNotCalibratedError(RuntimeError):
    pass


class LearnedResidualHook:
    def __init__(self, cfg: LearnedResidualConfig, fn: ResidualFn | None = None) -> None:
        if cfg.enabled and not cfg.calibration_data_ref:
            raise LearnedResidualNotCalibratedError(
                "learned residual enabled without calibration_data_ref (ch33 forbids an uncalibrated learned Twin2E component)"
            )
        if cfg.enabled and fn is None:
            raise LearnedResidualNotCalibratedError(
                "learned residual enabled but no residual function supplied"
            )
        self.cfg = cfg
        self.fn = fn

    @property
    def active(self) -> bool:
        return self.cfg.enabled

    def apply(self, name: str, values: np.ndarray, dt_s: float) -> np.ndarray:
        if not self.cfg.enabled or self.fn is None:
            return values
        residual = np.asarray(self.fn(name, values, dt_s), dtype=np.float64)
        if residual.shape != values.shape:
            raise ValueError(f"residual for {name} has shape {residual.shape}, expected {values.shape}")
        return values + residual
