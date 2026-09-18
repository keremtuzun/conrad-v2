"""Model2S / UAHSM: Uncertainty-Aware Hierarchical Spatial Memory (spatial BELIEF child of Model 2).

BELIEF PLANE: this package never imports ``conrad.twins``, ``conrad.schemas.truth``, ``conrad.sim`` or
``conrad.evaluation`` and never references truth types or true poses (tests/unit/domains/spatial).
"""

from typing import Any

IMPLEMENTATION_METADATA: dict[str, Any] = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "component_status": {
        "sparse_hierarchical_map": "EXPERIMENTAL_CANDIDATE",
        "observed_inferred_predicted_unknown_status": "FROZEN_CONTRACT (ch14 'What is now frozen')",
        "pose_uncertainty_propagation": "EXPERIMENTAL_CANDIDATE (isotropic covariance approximation)",
        "relational_gap_inference": "EXPERIMENTAL_CANDIDATE",
        "temporal_change_model": "EXPERIMENTAL_CANDIDATE",
        "adaptive_refinement_policy": "EXPERIMENTAL_CANDIDATE (deterministic, ch33 V0.1)",
        "learned_cell_update_and_graph_attention": "OPEN_BLOCKED (ch33 learned UAHSM heads not implemented)",
    },
    "source_sections": [
        "ch14 Model 2S and Twin2S (UAHSM, pose uncertainty, free space, unknown != occupied, relational, TBD)",
        "ch33 Model 2S and Twin2S exact implementation (0.25/0.125/0.0625 m, deterministic refinement)",
        "ch2 Model2Child interface",
        "ch28 property-level knowledge status and provenance",
    ],
    "configuration_keys": [
        "model.grid.*",
        "model.occupancy.*",
        "model.pose.*",
        "model.sensor.*",
        "model.relational.*",
        "model.temporal.*",
        "model.refinement.*",
        "model.messages.*",
        "model.query.*",
    ],
    "assumptions": [
        "All numeric defaults in SpatialConfig are ENGINEERING_ESTIMATE / SYNTHETIC_ONLY, not calibrated.",
        "Pose covariance is reduced to isotropic RMS position and rotation sigmas; a point at lever arm L gets "
        "sigma^2 = sigma_pos^2 + (2/3) sigma_rot^2 L^2.",
        "U_A = measurement noise (range noise, sonar elevation arc, sensor health); U_E = pose-induced spread "
        "(localisation ignorance is reducible, so epistemic); U_C = change score from contradicting "
        "re-observation; U_O = 1 - coverage (+ inference / prediction terms).",
        "One observation contributes at most unit hit and unit free mass per cell (no per-ray counting).",
        "Log-odds magnitude is capped at l_max / (1 + (sigma_pose/res)^2) so a large pose sigma can never "
        "yield a confident cell.",
        "A NaN range is ambiguous (dropout vs no return) and carves no free space by default.",
        "Sonar returns are spread over the full vertical beam; free space only before the first detection.",
        "RGB has no range and does not update geometry in V1.",
        "Point clouds from the same capture as a range image are deduplicated (same sensor, same time).",
        "Relational inference bridges <= max_gap_cells between OBSERVED occupied cells on the 13 lattice "
        "lines; inferred cells never support further inference.",
        "The refined (local) levels are working memory; the regional base grid always holds the fused state.",
        "Robot sensor extrinsics/intrinsics come from mission context (SensorSpec), never from truth.",
        "The belief state embedding is an interpretable 12-feature summary; no learned latent in V1.",
    ],
    "baselines": [
        "PlainOccupancyGrid (S-B0 classical occupancy map: per-ray counting, no status, no pose covariance)",
        "MaxLikelihoodSurfaceFill (naive geometry completion; exposes unsupported confidence)",
        "Model2S with ignore_pose_config (ablation: pose covariance ignored)",
    ],
    "acceptance_tests": [
        "tests/unit/domains/spatial",
        "tests/property/domains/spatial",
        "tests/simulation/spatial",
    ],
    "claim_status": "EVALUATED",  # 2S-E001..E004 executed (SYNTHETIC_ONLY); not VALIDATED
}
