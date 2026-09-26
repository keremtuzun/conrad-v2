"""Derive the smallest exact-binomial final design from the frozen v10 validation result."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def binomial_upper_tail(n: int, minimum_successes: int, probability: float) -> float:
    return sum(
        comb(n, successes) * probability**successes * (1.0 - probability) ** (n - successes)
        for successes in range(minimum_successes, n + 1)
    )


def design(
    *,
    validation_successes: int,
    validation_trials: int,
    null_probability: float,
    alpha: float,
    target_power: float,
    maximum_n: int,
) -> dict[str, float | int | str]:
    if validation_successes != validation_trials:
        raise ValueError("this preregistered exact bound is defined for an all-success validation result")
    conservative_probability = alpha ** (1.0 / validation_trials)
    for n in range(1, maximum_n + 1):
        for minimum_successes in range(n + 1):
            false_positive_probability = binomial_upper_tail(n, minimum_successes, null_probability)
            power = binomial_upper_tail(n, minimum_successes, conservative_probability)
            if false_positive_probability <= alpha and power >= target_power:
                return {
                    "method": "one-sided exact binomial with one-sided Clopper-Pearson validation bound",
                    "validation_successes": validation_successes,
                    "validation_trials": validation_trials,
                    "alpha": alpha,
                    "null_nominal_warrant_probability": null_probability,
                    "validation_one_sided_lower_probability": conservative_probability,
                    "target_power": target_power,
                    "maximum_available_n": maximum_n,
                    "selected_n": n,
                    "minimum_nominal_warrants": minimum_successes,
                    "false_positive_probability_at_null": false_positive_probability,
                    "power_at_validation_lower_bound": power,
                }
    raise ValueError("no design satisfies the declared alpha and power within the available final pool")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    report = design(
        validation_successes=10,
        validation_trials=10,
        null_probability=0.5,
        alpha=0.05,
        target_power=0.8,
        maximum_n=40,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
