"""Shared evaluation metrics. Every masked metric reports its denominator."""

from conrad.evaluation.metrics.bootstrap import (
    Interval,
    PairedComparison,
    bootstrap_ci,
    paired_seed_comparison,
)
from conrad.evaluation.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    gaussian_nll,
    negative_log_likelihood,
)
from conrad.evaluation.metrics.masked import (
    MaskedMetric,
    masked_accuracy,
    masked_mae,
    masked_mean,
    masked_mse,
)
from conrad.evaluation.metrics.unsupported import confident_wrong_rate, unsupported_confidence_rate

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch27 Statistical protocol",
        "ch27 H-2S-01 central metric UC",
        "ch26 acceptance dimensions",
    ],
    "configuration_keys": [
        "eval.calibration.n_bins",
        "eval.bootstrap.n_resamples",
        "eval.unsupported.threshold",
    ],
    "assumptions": [
        "ECE uses equal-width bins over the top-class probability",
        "bootstrap is the percentile bootstrap of the mean",
        "the unsupported-confidence threshold is always supplied by the caller (spec leaves it OPEN)",
    ],
    "baselines": [],
    "acceptance_tests": ["tests/unit/evaluation/test_eval_metrics.py"],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "Interval",
    "MaskedMetric",
    "PairedComparison",
    "bootstrap_ci",
    "brier_score",
    "confident_wrong_rate",
    "expected_calibration_error",
    "gaussian_nll",
    "masked_accuracy",
    "masked_mae",
    "masked_mean",
    "masked_mse",
    "negative_log_likelihood",
    "paired_seed_comparison",
    "unsupported_confidence_rate",
]
