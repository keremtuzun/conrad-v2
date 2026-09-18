"""Training infrastructure: checkpoints, deterministic trainer, curriculum, run directories."""

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch3 Core training sequence C0-C9",
        "ch23 Training Architecture",
        "ch26 Experiment discipline / Compute tiers",
        "ch33 Training configuration and implementation gates",
        "ch34 Configuration / run directory",
        "ch36 Interface, configuration and compatibility policy (SS-05)",
    ],
    "configuration_keys": ["train.trainer", "train.curriculum", "train.mlflow", "device", "run.seed"],
    "assumptions": [
        "schema_version compatibility is same-major (conrad.schemas.base.check_schema_compatible)",
        "curriculum parameter-name patterns are configuration; defaults assume top-level modules named "
        "encoder/association/buo/rbp/tbd",
        "effective batch is reached by accumulating batch_size-sized batches supplied by the caller",
    ],
    "baselines": [],
    "acceptance_tests": ["tests/unit/training"],
    "claim_status": "IMPLEMENTED",
}
