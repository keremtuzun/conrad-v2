"""2E experiments (ch12 2E-E01/E02/E08/E09 subset). EVALUATION PLANE: may use Twin2E for truth."""

from __future__ import annotations

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch12 Core experiments, Coupling benefit, Unsupported ecological inference, Field reconstruction"
    ],
    "configuration_keys": [
        "configs/eval/2e_e001.yaml",
        "configs/eval/2e_e002.yaml",
        "configs/eval/2e_e003.yaml",
        "configs/eval/2e_e001_r2.yaml",
        "configs/eval/2e_e002_r2.yaml",
        "configs/eval/2e_e003_r2.yaml",
        "configs/eval/2e_e001_r3.yaml",
        "configs/eval/2e_e002_r3.yaml",
        "configs/eval/2e_e003_r3.yaml",
    ],
    "assumptions": ["SYNTHETIC_ONLY Twin2E small world; no real-data claim"],
    "baselines": ["field_only", "static_field", "uncoupled", "entity_only"],
    "acceptance_tests": ["tests/unit/domains/ecological"],
    "claim_status": "EVALUATED",
}
