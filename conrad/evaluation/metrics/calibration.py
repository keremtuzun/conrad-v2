"""Calibration metrics: ECE, NLL and Brier score, mask-aware with explicit denominators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from conrad.evaluation.metrics.masked import MaskedMetric, masked_mean, resolve_mask

_EPS = 1e-12


def _prepare(
    probabilities: ArrayLike, labels: ArrayLike, mask: ArrayLike | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accept ``[N]`` binary P(class 1) or ``[N, K]`` class probabilities."""
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)
    if probs.ndim != 2 or probs.shape[0] != y.shape[0]:
        raise ValueError("probabilities must be [N] or [N, K] and match labels")
    keep = resolve_mask(mask, probs.shape[0])
    p = probs[keep]
    kept_labels = np.nan_to_num(y[keep].astype(np.float64)).astype(np.int64)
    if p.size:
        if np.any(p < -1e-9) or np.any(p > 1 + 1e-9) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("class probabilities must lie in [0, 1] and sum to 1")
        if kept_labels.min() < 0 or kept_labels.max() >= p.shape[1]:
            raise ValueError("label outside the probability vector")
    return p, kept_labels, keep


def expected_calibration_error(
    probabilities: ArrayLike, labels: ArrayLike, mask: ArrayLike | None = None, *, n_bins: int = 15
) -> MaskedMetric:
    """Equal-width confidence bins over the top-class probability."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p, y, keep = _prepare(probabilities, labels, mask)
    total = int(keep.shape[0])
    if p.shape[0] == 0:
        return MaskedMetric(name="ece", value=None, denominator=0, total=total)
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    bins = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        members = bins == b
        if members.any():
            ece += float(members.mean()) * abs(
                float(correct[members].mean()) - float(confidence[members].mean())
            )
    return MaskedMetric(name="ece", value=ece, denominator=int(p.shape[0]), total=total)


def negative_log_likelihood(
    probabilities: ArrayLike, labels: ArrayLike, mask: ArrayLike | None = None
) -> MaskedMetric:
    p, y, keep = _prepare(probabilities, labels, mask)
    per_sample = np.full(keep.shape[0], np.nan)
    per_sample[keep] = -np.log(np.clip(p[np.arange(p.shape[0]), y], _EPS, 1.0))
    return masked_mean(per_sample, keep, name="nll")


def brier_score(probabilities: ArrayLike, labels: ArrayLike, mask: ArrayLike | None = None) -> MaskedMetric:
    """Multi-class Brier score: squared error summed over classes, averaged over samples."""
    p, y, keep = _prepare(probabilities, labels, mask)
    onehot = np.zeros_like(p)
    onehot[np.arange(p.shape[0]), y] = 1.0
    per_sample = np.full(keep.shape[0], np.nan)
    per_sample[keep] = ((p - onehot) ** 2).sum(axis=1)
    return masked_mean(per_sample, keep, name="brier")


def gaussian_nll(
    mean: ArrayLike, variance: ArrayLike, target: ArrayLike, mask: ArrayLike | None = None
) -> MaskedMetric:
    """Regression NLL under a diagonal Gaussian, averaged over trailing dimensions per sample."""
    mu, var, t = (np.asarray(a, dtype=np.float64) for a in (mean, variance, target))
    keep = resolve_mask(mask, mu.shape[0])
    if np.any(var[keep] <= 0):
        raise ValueError("variance must be strictly positive at available positions")
    safe_var = np.where(var > 0, var, 1.0)
    nll = 0.5 * (np.log(2 * np.pi * safe_var) + (np.nan_to_num(t) - mu) ** 2 / safe_var)
    per_sample = nll if nll.ndim == 1 else nll.reshape(nll.shape[0], -1).mean(axis=1)
    return masked_mean(np.where(keep, per_sample, np.nan), keep, name="gaussian_nll")
