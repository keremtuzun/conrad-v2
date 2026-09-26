"""The v10 final prefix is the smallest design meeting frozen alpha and power."""

import pytest

from scripts.design_i5_spatial_v1_1_final import binomial_upper_tail, design


def test_exact_final_design_is_smallest_qualified_prefix() -> None:
    report = design(
        validation_successes=10,
        validation_trials=10,
        null_probability=0.5,
        alpha=0.05,
        target_power=0.8,
        maximum_n=40,
    )
    assert report["selected_n"] == 28
    assert report["minimum_nominal_warrants"] == 19
    assert report["false_positive_probability_at_null"] == pytest.approx(0.04357927665114403)
    assert report["power_at_validation_lower_bound"] == pytest.approx(0.8352075352835021)
    lower_probability = float(report["validation_one_sided_lower_probability"])
    for n in range(1, 28):
        assert not any(
            binomial_upper_tail(n, k, 0.5) <= 0.05
            and binomial_upper_tail(n, k, lower_probability) >= 0.8
            for k in range(n + 1)
        )


def test_power_design_refuses_non_all_success_validation_shortcut() -> None:
    with pytest.raises(ValueError, match="all-success"):
        design(
            validation_successes=9,
            validation_trials=10,
            null_probability=0.5,
            alpha=0.05,
            target_power=0.8,
            maximum_n=40,
        )
