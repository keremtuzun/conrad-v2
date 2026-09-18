"""Unsupported-confidence metrics (ch27 H-2S-01: UC = P(high-confidence claim | insufficient evidence))."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from conrad.evaluation.metrics.masked import MaskedMetric, resolve_mask


def _vectors(confidence: ArrayLike, flags: ArrayLike, name: str) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(confidence, dtype=np.float64)
    flag = np.asarray(flags)
    if conf.ndim != 1 or flag.dtype != np.bool_ or flag.shape != conf.shape:
        raise ValueError(f"{name} must be a boolean vector matching confidence")
    return conf, flag


def unsupported_confidence_rate(
    confidence: ArrayLike, evidence_sufficient: ArrayLike, *, threshold: float, mask: ArrayLike | None = None
) -> MaskedMetric:
    """Share of insufficient-evidence claims asserted with confidence >= threshold.

    Denominator = insufficient-evidence claims; with none, the metric has no value.
    """
    conf, sufficient = _vectors(confidence, evidence_sufficient, "evidence_sufficient")
    selected = conf[resolve_mask(mask, conf.shape[0]) & ~sufficient]
    value = float((selected >= threshold).mean()) if selected.size else None
    return MaskedMetric(
        name="unsupported_confidence_rate",
        value=value,
        denominator=int(selected.size),
        total=int(conf.shape[0]),
    )


def confident_wrong_rate(
    confidence: ArrayLike, correct: ArrayLike, *, threshold: float, mask: ArrayLike | None = None
) -> MaskedMetric:
    """Share of evaluable claims that are wrong AND asserted with confidence >= threshold."""
    conf, right = _vectors(confidence, correct, "correct")
    keep = resolve_mask(mask, conf.shape[0])
    selected = (conf[keep] >= threshold) & ~right[keep]
    value = float(selected.mean()) if selected.size else None
    return MaskedMetric(
        name="confident_wrong_rate", value=value, denominator=int(selected.size), total=int(conf.shape[0])
    )
