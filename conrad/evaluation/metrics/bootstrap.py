"""Bootstrap confidence intervals and paired seed comparison (ch27 Statistical protocol)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field

from conrad.schemas.base import ConradModel


class Interval(ConradModel):
    point: float
    low: float | None
    high: float | None
    n: int
    confidence: float
    method: str

    @property
    def excludes_zero(self) -> bool:
        return self.low is not None and self.high is not None and (self.low > 0 or self.high < 0)


class PairedComparison(ConradModel):
    """Per-seed benefit of candidate over baseline, signed so that positive means better."""

    metric: str
    direction: str = Field(pattern="^(lower|higher)$")
    seeds: tuple[int, ...]
    baseline_mean: float
    candidate_mean: float
    benefit_per_seed: tuple[float, ...]
    benefit: Interval
    fraction_of_seeds_improved: float

    @property
    def repeatable_benefit(self) -> bool:
        return self.benefit.low is not None and self.benefit.low > 0

    @property
    def repeatable_regression(self) -> bool:
        return self.benefit.high is not None and self.benefit.high < 0


def bootstrap_ci(
    values: ArrayLike, rng: np.random.Generator, *, n_resamples: int = 2000, confidence: float = 0.95
) -> Interval:
    """Percentile bootstrap of the mean. Fewer than two values give no interval."""
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    if not np.all(np.isfinite(data)):
        raise ValueError("bootstrap_ci received non-finite values")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    point = float(data.mean())
    if data.size < 2:
        return Interval(point=point, low=None, high=None, n=1, confidence=confidence, method="none_n_lt_2")
    means = data[rng.integers(0, data.size, size=(n_resamples, data.size))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return Interval(
        point=point,
        low=float(low),
        high=float(high),
        n=int(data.size),
        confidence=confidence,
        method=f"percentile_bootstrap_{n_resamples}",
    )


def paired_seed_comparison(
    baseline: Mapping[int, float],
    candidate: Mapping[int, float],
    rng: np.random.Generator,
    *,
    metric: str,
    direction: str,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> PairedComparison:
    """Seed sets must match exactly; an unmatched comparison is refused."""
    if set(baseline) != set(candidate):
        raise ValueError(
            f"seed sets differ: baseline-only {sorted(set(baseline) - set(candidate))}, "
            f"candidate-only {sorted(set(candidate) - set(baseline))}"
        )
    if not baseline:
        raise ValueError("no seeds to compare")
    if direction not in ("lower", "higher"):
        raise ValueError("direction must be 'lower' or 'higher'")
    seeds = tuple(sorted(baseline))
    b = np.asarray([baseline[s] for s in seeds], dtype=np.float64)
    c = np.asarray([candidate[s] for s in seeds], dtype=np.float64)
    benefit = (b - c) if direction == "lower" else (c - b)
    return PairedComparison(
        metric=metric,
        direction=direction,
        seeds=seeds,
        baseline_mean=float(b.mean()),
        candidate_mean=float(c.mean()),
        benefit_per_seed=tuple(float(v) for v in benefit),
        benefit=bootstrap_ci(benefit, rng, n_resamples=n_resamples, confidence=confidence),
        fraction_of_seeds_improved=float((benefit > 0).mean()),
    )
