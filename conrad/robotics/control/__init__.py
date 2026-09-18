"""Motion control (ch20 Controller V1). Canonical baseline: cascaded PID. MPC/RL: interfaces only."""

from conrad.robotics.control.experimental import (
    ExperimentalWrenchController,
    LearnedResidualControllerInterface,
    MpcControllerInterface,
)
from conrad.robotics.control.pid import CascadedPidController, ControlConfig, ControlReference

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": ["ch20 Controller input", "ch20 Controller V1", "ch20 Controller baselines"],
    "configuration_keys": ["ControlConfig", "RobotConfig.safety.max_speed_mps", "configs/sim/nav_kernel.yaml"],
    "assumptions": [
        "gains tuned on the SYNTHETIC_ONLY sim_reference vehicle",
        "water density / gravity for buoyancy feedforward are configured environment assumptions",
    ],
    "baselines": [
        "CTRL-B1 cascaded PID (implemented)",
        "CTRL-B3 MPC (interface only)",
        "CTRL-B6 residual (interface only)",
    ],
    "acceptance_tests": ["tests/unit/robotics/test_nav_control.py"],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "IMPLEMENTATION_METADATA",
    "CascadedPidController",
    "ControlConfig",
    "ControlReference",
    "ExperimentalWrenchController",
    "LearnedResidualControllerInterface",
    "MpcControllerInterface",
]
