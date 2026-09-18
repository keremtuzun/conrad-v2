"""Navigation benchmark suite NAV-001..008 (ch20 Navigation benchmarks, ch21 Navigation benchmarks)."""

from conrad.evaluation.nav_benchmarks.runner import run_benchmark
from conrad.evaluation.nav_benchmarks.scenarios import BENCHMARK_CONFIG, BENCHMARK_IDS, load_scenario

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch20 Navigation benchmarks",
        "ch20 Navigation metrics",
        "ch21 Navigation benchmarks",
    ],
    "configuration_keys": ["configs/sim/nav_benchmarks.yaml", "configs/robot/sim_reference.yaml"],
    "assumptions": [
        "scenarios are SYNTHETIC_ONLY on the L1 Python kernel; results say nothing about the physical robot",
        "position fixes come from a synthetic USBL-like sensor rendered on the sonar slot",
        "the launch pose is mission context supplied to the estimator (with its initial covariance)",
    ],
    "baselines": ["EKF + A* + trapezoidal trajectory + cascaded PID + BVLS allocation + rule supervisor"],
    "acceptance_tests": ["tests/simulation/nav"],
    "claim_status": "EVALUATED",
}

__all__ = ["BENCHMARK_CONFIG", "BENCHMARK_IDS", "IMPLEMENTATION_METADATA", "load_scenario", "run_benchmark"]
