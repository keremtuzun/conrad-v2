"""2S experiments (E001-E004) for Model2S / UAHSM. TRUTH PLANE: may use Twin2S for truth and metrics."""

from typing import Any

IMPLEMENTATION_METADATA: dict[str, Any] = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": ["ch14 2S core experiments E01-E08", "ch15 OCPWE"],
    "configuration_keys": [
        "configs/eval/2s_e001.yaml",
        "configs/eval/2s_e002.yaml",
        "configs/eval/2s_e003.yaml",
        "configs/eval/2s_e004.yaml",
    ],
    "assumptions": [
        "Query-cell observability truth = VISIBLE (Twin2S oracle, tolerance half a cell diagonal) from any "
        "scheduled true pose of an active geometric sensor.",
        "A claim is confident when max(p, 1-p) >= 0.9 (Twin2S evaluation default, not a validated threshold).",
    ],
    "baselines": ["plain_grid", "ml_fill", "uahsm_no_pose_cov"],
    "acceptance_tests": ["tests/simulation/spatial"],
    "claim_status": "EVALUATED",
}
