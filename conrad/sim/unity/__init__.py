"""Python-side Unity V2 tooling: RobotConfig export, experiments, validity levels, truth client.

``conrad.sim.unity.truth`` is truth-plane only and must never be imported by belief/decision planes.
"""

from conrad.sim.unity.experiment import (
    ExperimentConfig,
    FaultSpec,
    ValidityAssessment,
    assess_validity_level,
    draw_fault_schedule,
)
from conrad.sim.unity.robot_export import (
    UNITY_ROBOT_CONFIG_FORMAT,
    robot_config_to_unity,
    write_unity_robot_config,
)
from conrad.sim.unity.settings import UnitySimSettings, load_unity_settings, unity_settings

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch20 ExperimentConfig",
        "ch20 Deterministic replay",
        "ch20 Fault injection",
        "ch20/ch21 Simulation validity levels",
        "ch21 True state vs reported state",
        "ch21 Mass properties / Buoyancy / Thruster model",
    ],
    "configuration_keys": [
        "sim.unity.bridge",
        "sim.unity.truth_endpoint",
        "sim.unity.physics_dt_ns",
        "sim.unity.player_path",
        "sim.unity.robot_config_export",
    ],
    "assumptions": [
        "RobotConfig travels to Unity in the Conrad BODY convention; the C# loader applies the axis map",
        "validity level is derived from parameter provenance only; L4 additionally needs a validation report",
        "the Unity player has not been executed here; Unity execution is an external blocker",
    ],
    "baselines": ["conrad.sim.kernel (Python 6-DOF kernel)"],
    "acceptance_tests": ["tests/unit/adapters/unity/test_sim_unity_tooling.py"],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "IMPLEMENTATION_METADATA",
    "UNITY_ROBOT_CONFIG_FORMAT",
    "ExperimentConfig",
    "FaultSpec",
    "UnitySimSettings",
    "ValidityAssessment",
    "assess_validity_level",
    "draw_fault_schedule",
    "load_unity_settings",
    "robot_config_to_unity",
    "unity_settings",
    "write_unity_robot_config",
]
