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
from conrad.sim.unity.player import (
    UnityPlayerError,
    UnityPlayerSession,
    experiment_document,
    find_player,
    scenario_document,
)
from conrad.sim.unity.robot_export import (
    UNITY_ROBOT_CONFIG_FORMAT,
    robot_config_to_unity,
    write_unity_robot_config,
)
from conrad.sim.unity.scene import (
    BoxPrimitive,
    CapsulePrimitive,
    HeightfieldPrimitive,
    SceneGeometry,
    load_scene_geometry,
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
        "ch21 Development progression V2.0 / Phase 4 gate U0",
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
        "the headless Windows64 player (Unity 6000.5.9f1) is launched by UnityPlayerSession on loopback TCP",
        "scene geometry travels in Conrad WORLD; Unity converts it with the documented axis map",
        "every physical parameter used by the U0 tests is SYNTHETIC_ONLY; simulation validity is L1",
    ],
    "baselines": ["conrad.sim.kernel (Python 6-DOF kernel)", "closed-form 1-DOF drag/buoyancy references"],
    "acceptance_tests": [
        "tests/unit/adapters/unity/test_sim_unity_tooling.py",
        "tests/unit/adapters/unity/test_tcp_transport_and_scene.py",
        "tests/unity_live",
    ],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "IMPLEMENTATION_METADATA",
    "UNITY_ROBOT_CONFIG_FORMAT",
    "BoxPrimitive",
    "CapsulePrimitive",
    "ExperimentConfig",
    "FaultSpec",
    "HeightfieldPrimitive",
    "SceneGeometry",
    "UnityPlayerError",
    "UnityPlayerSession",
    "UnitySimSettings",
    "ValidityAssessment",
    "assess_validity_level",
    "draw_fault_schedule",
    "experiment_document",
    "find_player",
    "load_scene_geometry",
    "load_unity_settings",
    "robot_config_to_unity",
    "scenario_document",
    "unity_settings",
    "write_unity_robot_config",
]
