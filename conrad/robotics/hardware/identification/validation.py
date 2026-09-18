"""Held-out validation of identified models (reality-gap metric E_traj = RMSE(X_sim, X_real)).

Every function first calls ``dataset.assert_validation_only`` so identification logs can never be
used to score the model that was fitted on them.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from conrad.robotics.hardware.identification.dataset import ExperimentKind, IdentificationDataset, LogSegment
from conrad.robotics.hardware.identification.fitting import FitResult
from conrad.robotics.hardware.identification.models import simulate_axis, thruster_response
from conrad.schemas.base import ConradModel

_UNITS = {
    ExperimentKind.SURGE_ACCELERATION: ("N", "m/s"),
    ExperimentKind.SWAY: ("N", "m/s"),
    ExperimentKind.HEAVE: ("N", "m/s"),
    ExperimentKind.YAW_ROTATION: ("N*m", "rad/s"),
}


class ValidationMetric(ConradModel):
    trajectory_id: str
    rmse: float
    nrmse: float | None
    max_abs_error: float
    units: str


class ValidationResult(ConradModel):
    kind: ExperimentKind
    per_trajectory: tuple[ValidationMetric, ...]
    pooled_rmse: float
    within_envelope: bool | None
    envelope_rmse: float | None


def _metric(seg: LogSegment, pred: np.ndarray, meas: np.ndarray, units: str) -> ValidationMetric:
    err = pred - meas
    rng = float(np.ptp(meas))
    rmse = float(np.sqrt(np.mean(err**2)))
    return ValidationMetric(
        trajectory_id=seg.trajectory_id,
        rmse=rmse,
        nrmse=None if rng == 0 else rmse / rng,
        max_abs_error=float(np.max(np.abs(err))),
        units=units,
    )


def _pool(
    kind: ExperimentKind, metrics: list[ValidationMetric], sizes: list[int], envelope: float | None
) -> ValidationResult:
    if not metrics:
        raise ValueError(f"no validation segments of kind {kind.value}")
    pooled = float(np.sqrt(sum(m.rmse**2 * n for m, n in zip(metrics, sizes, strict=True)) / sum(sizes)))
    return ValidationResult(
        kind=kind,
        per_trajectory=tuple(metrics),
        pooled_rmse=pooled,
        within_envelope=None if envelope is None else pooled <= envelope,
        envelope_rmse=envelope,
    )


def validate_axis(
    dataset: IdentificationDataset,
    segments: Sequence[LogSegment],
    kind: ExperimentKind,
    rigid_inertia: float,
    fit: FitResult,
    envelope_rmse: float | None = None,
) -> ValidationResult:
    dataset.assert_validation_only(segments)
    fu, vu = _UNITS[kind]
    bias = next((p.value for p in fit.parameters if p.name == "bias"), 0.0)
    metrics, sizes = [], []
    for s in segments:
        if s.kind is not kind:
            raise ValueError(f"{s.trajectory_id} is {s.kind.value}, not {kind.value}")
        v = s.signal("velocity", vu)
        pred = simulate_axis(
            s.time_s,
            s.signal("force", fu),
            rigid_inertia + fit.value("added_inertia"),
            fit.value("linear_damping"),
            fit.value("quadratic_damping"),
            bias,
            v0=float(v[0]),
        )
        metrics.append(_metric(s, pred, v, vu))
        sizes.append(v.size)
    return _pool(kind, metrics, sizes, envelope_rmse)


def validate_thruster(
    dataset: IdentificationDataset,
    segments: Sequence[LogSegment],
    fit: FitResult,
    deadzone: float,
    envelope_rmse: float | None = None,
) -> ValidationResult:
    dataset.assert_validation_only(segments)
    metrics, sizes = [], []
    for s in segments:
        thrust = s.signal("thrust_n", "N")
        pred = thruster_response(
            s.time_s,
            s.signal("command", "unitless"),
            fit.value("k_fwd"),
            fit.value("k_rev"),
            deadzone,
            fit.value("time_constant_s"),
            fit.value("latency_s"),
        )
        metrics.append(_metric(s, pred, thrust, "N"))
        sizes.append(thrust.size)
    return _pool(ExperimentKind.THRUSTER_STEP, metrics, sizes, envelope_rmse)
