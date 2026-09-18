"""Model2T structural experiments 2T-E001..E004 (truth plane allowed; never imported by belief code)."""

from __future__ import annotations

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch10 Model 2T training stages T2-T6, core experiments, key metrics, completion gates"
    ],
    "configuration_keys": [
        "configs/eval/2t_e001.yaml",
        "configs/eval/2t_e002.yaml",
        "configs/eval/2t_e003.yaml",
        "configs/eval/2t_e004.yaml",
    ],
    "assumptions": [
        "SYNTHETIC_ONLY Twin2T scenarios with T0 abstract observations",
        "association of observation -> registry component taken from the twin supervision label (oracle)",
        "asset registry built from design fields only (type, material, coating, nominal wall, relationships)",
        "E003 coupled/misleading episodes inject initial states and an ENVIRONMENT_CHANGE event",
        "degraded thresholds: corrosion 1.0 mm, crack 5.0 mm (experiment config)",
    ],
    "baselines": [
        "LATEST_OBSERVATION",
        "SINGLE_FRAME",
        "INDEPENDENT_COMPONENT",
        "GENERIC_RELATIONAL",
        "GRU_TEMPORAL",
        "NO_RATE_PRIOR (ablation)",
    ],
    "acceptance_tests": ["tests/unit/domains/technical/test_m2t_experiment_smoke.py"],
    "claim_status": "EVALUATED",
}
