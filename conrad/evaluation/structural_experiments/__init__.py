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
        "configs/eval/2t_e00{1,2,3,4}_r3.yaml (FINAL-3) and *_r3_dev.yaml (iteration 3)",
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

_M = "conrad.evaluation.structural_experiments."
EXPERIMENTS: dict[str, tuple[str, str]] = {
    # Iteration 3 (docs/audits/MODEL2T_REPAIR.md): DEV configs for design, FINAL-3 configs run once.
    "2T-E001-R3-DEV": (_M + "e001_r2", "configs/eval/2t_e001_r3_dev.yaml"),
    "2T-E001-R3": (_M + "e001_r2", "configs/eval/2t_e001_r3.yaml"),
    "2T-E002-R3-DEV": (_M + "e002_persistent", "configs/eval/2t_e002_r3_dev.yaml"),
    "2T-E002-R3": (_M + "e002_persistent", "configs/eval/2t_e002_r3.yaml"),
    "2T-E003-R3-DEV": (_M + "e003_r2", "configs/eval/2t_e003_r3_dev.yaml"),
    "2T-E003-R3": (_M + "e003_r2", "configs/eval/2t_e003_r3.yaml"),
    "2T-E004-R3-DEV": (_M + "e004_temporal", "configs/eval/2t_e004_r3_dev.yaml"),
    "2T-E004-R3": (_M + "e004_temporal", "configs/eval/2t_e004_r3.yaml"),
}
