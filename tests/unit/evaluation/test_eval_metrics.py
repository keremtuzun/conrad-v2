from __future__ import annotations

import math

import numpy as np
import pytest

from conrad.evaluation.metrics import (
    bootstrap_ci,
    brier_score,
    confident_wrong_rate,
    expected_calibration_error,
    gaussian_nll,
    masked_accuracy,
    masked_mae,
    masked_mean,
    negative_log_likelihood,
    paired_seed_comparison,
    unsupported_confidence_rate,
)


def test_all_missing_labels_give_no_value_and_zero_denominator():
    none = np.zeros(5, dtype=bool)
    for metric in (
        masked_mean(np.full(5, np.nan), none),
        masked_mae(np.ones((5, 2)), np.full((5, 2), np.nan), none),
        masked_accuracy(np.ones(5), np.zeros(5), none),
        expected_calibration_error(np.full(5, 0.7), np.ones(5), none),
        negative_log_likelihood(np.full(5, 0.7), np.ones(5), none),
        brier_score(np.full(5, 0.7), np.ones(5), none),
    ):
        assert metric.value is None and metric.denominator == 0 and metric.total == 5
        assert not metric.evaluable


def test_masked_mean_reports_denominator_and_ignores_masked_nan():
    m = masked_mae(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 1.0]), np.array([True, False, True]))
    assert m.value == pytest.approx(1.0) and m.denominator == 2 and m.coverage == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        masked_mean(np.array([np.nan, 1.0]))
    with pytest.raises(TypeError):
        masked_mean(np.ones(2), np.ones(2))


def test_calibration_known_values():
    probs = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
    labels = np.array([0, 0, 0, 0, 1])
    assert expected_calibration_error(probs, labels).value == pytest.approx(0.0)
    assert negative_log_likelihood(probs, labels).value == pytest.approx(
        (4 * -math.log(0.8) - math.log(0.2)) / 5
    )
    assert brier_score(probs, labels).value == pytest.approx((4 * 0.08 + 1.28) / 5)
    over = expected_calibration_error(np.full(4, 0.9), np.array([1, 0, 1, 0]))
    assert over.value == pytest.approx(0.4)
    with pytest.raises(ValueError):
        brier_score(np.array([[0.5, 0.6]]), np.array([0]))
    assert gaussian_nll(np.zeros(2), np.ones(2), np.zeros(2)).value == pytest.approx(
        0.5 * math.log(2 * math.pi)
    )


def test_bootstrap_is_seeded_and_needs_two_values():
    a = bootstrap_ci([1.0, 2.0, 3.0, 4.0], np.random.default_rng(0))
    b = bootstrap_ci([1.0, 2.0, 3.0, 4.0], np.random.default_rng(0))
    assert a.low is not None and a.high is not None
    assert a == b and a.low <= a.point <= a.high
    single = bootstrap_ci([1.0], np.random.default_rng(0))
    assert single.low is None and not single.excludes_zero


def test_paired_seed_comparison():
    base = {1: 1.0, 2: 1.1, 3: 0.9, 4: 1.05}
    cand = {k: v - 0.3 for k, v in base.items()}
    cmp = paired_seed_comparison(base, cand, np.random.default_rng(0), metric="err", direction="lower")
    assert cmp.repeatable_benefit and cmp.fraction_of_seeds_improved == 1.0
    worse = paired_seed_comparison(base, cand, np.random.default_rng(0), metric="acc", direction="higher")
    assert worse.repeatable_regression
    with pytest.raises(ValueError, match="seed sets differ"):
        paired_seed_comparison(base, {1: 0.0}, np.random.default_rng(0), metric="err", direction="lower")


def test_unsupported_confidence_rate():
    conf = np.array([0.95, 0.4, 0.99, 0.2])
    sufficient = np.array([False, False, True, True])
    uc = unsupported_confidence_rate(conf, sufficient, threshold=0.9)
    assert uc.value == pytest.approx(0.5) and uc.denominator == 2
    none = unsupported_confidence_rate(conf, np.ones(4, dtype=bool), threshold=0.9)
    assert none.value is None and none.denominator == 0
    cw = confident_wrong_rate(conf, np.array([False, True, True, True]), threshold=0.9)
    assert cw.value == pytest.approx(0.25)
