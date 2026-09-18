"""Masked metric helpers. A metric over zero available labels has no value; it is never 0.0."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from conrad.schemas.base import ConradModel


class MaskedMetric(ConradModel):
    name: str
    value: float | None
    denominator: int
    total: int

    @property
    def evaluable(self) -> bool:
        return self.value is not None

    @property
    def coverage(self) -> float:
        return self.denominator / self.total if self.total else 0.0


def resolve_mask(mask: ArrayLike | None, length: int) -> np.ndarray:
    if mask is None:
        return np.ones(length, dtype=bool)
    resolved = np.asarray(mask)
    if resolved.dtype != np.bool_:
        raise TypeError("availability mask must be boolean")
    if resolved.shape != (length,):
        raise ValueError(f"mask shape {resolved.shape} does not match {length} samples")
    return resolved


def masked_mean(values: ArrayLike, mask: ArrayLike | None = None, *, name: str = "mean") -> MaskedMetric:
    """Mean over available entries. ``values`` may hold NaN where the mask is False."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("masked_mean expects one value per sample")
    keep = resolve_mask(mask, array.shape[0])
    selected = array[keep]
    if selected.size and not np.all(np.isfinite(selected)):
        raise ValueError(f"{name}: non-finite value at an available position")
    value = float(selected.mean()) if selected.size else None
    return MaskedMetric(name=name, value=value, denominator=int(selected.size), total=int(array.shape[0]))


def _per_sample_error(prediction: ArrayLike, target: ArrayLike, keep: np.ndarray, power: int) -> np.ndarray:
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"prediction {p.shape} and target {t.shape} differ")
    diff = np.abs(p - np.nan_to_num(t)) ** power
    per = diff if diff.ndim == 1 else diff.reshape(diff.shape[0], -1).mean(axis=1)
    return np.where(keep, per, np.nan)


def masked_mae(prediction: ArrayLike, target: ArrayLike, mask: ArrayLike | None = None) -> MaskedMetric:
    keep = resolve_mask(mask, np.asarray(prediction).shape[0])
    return masked_mean(_per_sample_error(prediction, target, keep, 1), keep, name="mae")


def masked_mse(prediction: ArrayLike, target: ArrayLike, mask: ArrayLike | None = None) -> MaskedMetric:
    keep = resolve_mask(mask, np.asarray(prediction).shape[0])
    return masked_mean(_per_sample_error(prediction, target, keep, 2), keep, name="mse")


def masked_accuracy(predicted: ArrayLike, target: ArrayLike, mask: ArrayLike | None = None) -> MaskedMetric:
    p, t = np.asarray(predicted), np.asarray(target)
    keep = resolve_mask(mask, p.shape[0])
    return masked_mean((p == t).astype(np.float64), keep, name="accuracy")
