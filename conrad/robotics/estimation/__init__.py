"""State estimation (ch20): StateEstimator contract and the conventional EKF baseline."""

from conrad.robotics.estimation.ekf import LOCALIZATION_LOST, EkfConfig, EkfStateEstimator
from conrad.robotics.estimation.interface import POSITION_FIX_KIND, MapConstraint, StateEstimator

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch20 State estimator interface",
        "ch20 Localization-loss behavior",
        "ch20 Robot state",
    ],
    "configuration_keys": ["EkfConfig", "RobotConfig.safety.max_pose_sigma_m"],
    "assumptions": [
        "SIMULATION_DEFAULT frame: WORLD +Z up, depth = -z",
        "sensor lever arms ignored",
        "orientation block of the 6x6 covariance is the body-local rotation-vector covariance",
        "position fixes arrive as Observation with sensor_context kind=POSITION_FIX (domain-local convention)",
        "horizontal position is unobservable without fixes; covariance then grows until LOCALIZATION_LOST",
    ],
    "baselines": ["EST-B0 15-state error-state EKF"],
    "acceptance_tests": ["tests/unit/robotics/test_nav_estimation.py"],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "IMPLEMENTATION_METADATA",
    "LOCALIZATION_LOST",
    "POSITION_FIX_KIND",
    "EkfConfig",
    "EkfStateEstimator",
    "MapConstraint",
    "StateEstimator",
]
