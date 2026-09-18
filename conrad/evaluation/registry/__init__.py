"""Research and experiment registry (ch27, ch35 Priorities 5 and 7, ch36 Model governance)."""

from conrad.evaluation.registry.experiments import (
    ExperimentRecord,
    ExperimentRegistry,
    Outcome,
    RegistryIntegrityError,
    Tier,
)
from conrad.evaluation.registry.hypotheses import (
    SEED_HYPOTHESES,
    SYSTEM_EXPERIMENTS,
    Hypothesis,
    HypothesisRegistry,
    HypothesisStatus,
)
from conrad.evaluation.registry.kill import (
    Answer,
    CostReport,
    KillReport,
    MechanismComparison,
    Verdict,
    assess,
    retain_with_adr,
)
from conrad.evaluation.registry.notebook import EntryResult, NotebookEntry, ResearchNotebook

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": ["ch26 Experiment discipline", "ch27", "ch35 Priority 5 / 7", "ch36 Model governance"],
    "configuration_keys": [],
    "assumptions": [
        "experiment IDs follow <PREFIX>-<PART>[-<PART>...] with prefixes CORE/2S/2T/2E/M1/ACTIVE/COM/NAV/SYS/XFER/X/ECMER/DATA",
        "a SUPPORTS outcome requires a paired-seed effect whose bootstrap CI excludes zero",
        "cost is NOT_EVALUABLE until RobotConfig-specific latency/memory budgets are supplied",
    ],
    "baselines": [],
    "acceptance_tests": ["tests/unit/evaluation"],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "SEED_HYPOTHESES",
    "SYSTEM_EXPERIMENTS",
    "Answer",
    "CostReport",
    "EntryResult",
    "ExperimentRecord",
    "ExperimentRegistry",
    "Hypothesis",
    "HypothesisRegistry",
    "HypothesisStatus",
    "KillReport",
    "MechanismComparison",
    "NotebookEntry",
    "Outcome",
    "RegistryIntegrityError",
    "ResearchNotebook",
    "Tier",
    "Verdict",
    "assess",
    "retain_with_adr",
]
