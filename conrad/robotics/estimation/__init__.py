"""State estimation (ch20): StateEstimator contract and the conventional EKF baseline."""

from conrad.robotics.estimation.ekf import (
    DEPTH_REJECTED,
    ESTIMATOR_INCONSISTENT,
    FIX_REJECTED,
    LOCALIZATION_LOST,
    EkfConfig,
    EkfStateEstimator,
)
from conrad.robotics.estimation.interface import POSITION_FIX_KIND, MapConstraint, StateEstimator

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch20 State estimator interface",
        "ch20 Localization-loss behavior",
        "ch20 Robot state",
    ],
    "configuration_keys": [
        "EkfConfig",
        "RobotConfig.safety.max_pose_sigma_m",
        "configs/sim/nav_calibration.yaml",
    ],
    "assumptions": [
        "SIMULATION_DEFAULT frame: WORLD +Z up, depth = -z",
        "sensor lever arms ignored",
        "orientation block of the 6x6 covariance is the body-local rotation-vector covariance",
        "position fixes arrive as Observation with sensor_context kind=POSITION_FIX (domain-local convention)",
        "horizontal position is unobservable without fixes; covariance then grows until LOCALIZATION_LOST",
        "thrust/drag velocity prior error = white part + random-walk WORLD offset c (current, drag error)",
        "all noise densities, walk rates and NIS limits are SYNTHETIC_ONLY configured values",
        "the estimator is never told injected noise levels; IMU/AHRS noise is estimated from its own data",
    ],
    "baselines": [
        "EST-B0 15-state EKF, white-noise thrust prior (EkfConfig.legacy_baseline)",
        "EST-B1 18-state EKF with prior-mismatch state, noise adaptation and NIS monitor (default)",
    ],
    "acceptance_tests": [
        "tests/unit/robotics/test_nav_estimation.py",
        "tests/unit/robotics/test_nav_estimator_calibration.py",
        "NAV-CAL-E001 (conrad/evaluation/nav_benchmarks/calibration.py)",
    ],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "DEPTH_REJECTED",
    "ESTIMATOR_INCONSISTENT",
    "FIX_REJECTED",
    "IMPLEMENTATION_METADATA",
    "LOCALIZATION_LOST",
    "POSITION_FIX_KIND",
    "EkfConfig",
    "EkfStateEstimator",
    "MapConstraint",
    "StateEstimator",
]
