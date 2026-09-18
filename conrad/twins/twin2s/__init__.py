"""Twin2S: canonical spatial truth, observability truth and OCPWE. TRUTH PLANE ONLY.

Belief/decision packages must never import this package (tests/leakage).
"""

from typing import Any

IMPLEMENTATION_METADATA: dict[str, Any] = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "component_status": {
        "geometry_contract": "FROZEN_CONTRACT",
        "sparse_voxel_octree_export": "FROZEN_CONTRACT",
        "twin_interface": "FROZEN_CONTRACT",
        "truth_observability_observation_separation": "FROZEN_CONTRACT",
        "OCPWE": "EXPERIMENTAL_CANDIDATE",
        "observability_score_formula": "EXPERIMENTAL_CANDIDATE",
        "sensor_models": "EXPERIMENTAL_CANDIDATE",
        "counterfactual_pairs": "EXPERIMENTAL_CANDIDATE",
    },
    "source_sections": [
        "ch14 Model 2S and Twin2S",
        "ch15 Twin2S Observability-Controlled World Design",
        "ch33 Model 2S and Twin2S exact implementation",
        "ch2 Twin interface",
    ],
    "configuration_keys": [
        "twin2s.octree.base_voxel_m",
        "twin2s.octree.refinement_voxels_m",
        "twin2s.octree.max_leaf_cells",
        "twin2s.raycast.max_steps",
        "twin2s.raycast.hit_epsilon_m",
        "twin2s.raycast.min_step_m",
        "twin2s.raycast.start_offset_m",
        "twin2s.raycast.refine_iterations",
        "twin2s.observed.min_quality",
        "twin2s.observed.min_incidence_cos",
        "twin2s.observed.diversity_angle_rad",
        "twin2s.observed.visibility_tolerance_m",
        "twin2s.water_attenuation_per_m",
        "twin2s.clock_domain",
    ],
    "assumptions": [
        "All geometry, sensor and water coefficients are SYNTHETIC_ONLY simulation defaults.",
        "WORLD is right-handed +Z up (scenario FrameConvention default); sensor frame +X boresight, +Z up.",
        "Observability score = LOS * sqrt(1 - r/rmax) * exp(-c r) * incidence; formula not frozen (ch14).",
        "An unobserved point is INFERABLE only when an explicit consensus mask over verified "
        "observation-compatible counterfactual worlds says so; otherwise UNKNOWN.",
        "IMU is synthesised by finite differences of the last three true poses; mount rotation assumed identity.",
    ],
    "baselines": [
        "fixed hand-built worlds",
        "random procedural worlds (style='poor'/random candidates)",
        "procedural worlds without observability control",
        "OCPWE without counterfactual pairs",
        "OCPWE without trajectory control",
        "full OCPWE",
    ],
    "acceptance_tests": ["tests/unit/twins/twin2s", "tests/property/twins/twin2s", "tests/simulation/twin2s"],
    "claim_status": "IMPLEMENTED",
}
