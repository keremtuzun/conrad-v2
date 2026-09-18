"""Hardware characterization ingestion, RobotConfig merge and the readiness / autonomy gate ledger.

The physical measurements themselves are the hardware owner's deliverable (ch22 handoff); nothing
in this package produces a physical value.
"""

from conrad.robotics.hardware.characterization.ingest import (
    CSV_COLUMNS,
    IngestError,
    load_bundle,
    load_csv_bundle,
    load_yaml_bundle,
    sha256_file,
)
from conrad.robotics.hardware.characterization.merge import (
    THRUSTER_FIELD_UNITS,
    ChangeReport,
    MergeError,
    ParameterChange,
    merge_characterization,
)
from conrad.robotics.hardware.characterization.readiness import (
    AUTONOMY_ORDER,
    DEFAULT_MIN_READINESS,
    READINESS_ORDER,
    AutonomyStage,
    GateError,
    GateLedger,
    LedgerEntry,
    ReadinessLevel,
)
from conrad.robotics.hardware.characterization.records import (
    Category,
    CharacterizationBundle,
    CharacterizationRecord,
)
from conrad.robotics.hardware.characterization.units import UnitError, convert_value

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch1 Hardware Handoff",
        "ch20 Hardware characterization pipeline",
        "ch22 Ownership Classification and Handoff",
        "ch22 Sim-to-real ladder R0..R6",
        "Phase 15 autonomy activation order",
    ],
    "configuration_keys": ["GateLedger.min_readiness"],
    "assumptions": [
        "autonomy stages need R4 controlled-water readiness, MANUAL needs R3 dry bench (configurable)",
        "body-frame vectors are delivered in the ROBOT frame; other frames are refused, not transformed",
        "a record's units must convert exactly to the target's declared units; composite units pass through only"
        " when spelled identically",
    ],
    "baselines": [],
    "acceptance_tests": ["tests/unit/characterization"],
    "claim_status": "IMPLEMENTED",
}

__all__ = [
    "AUTONOMY_ORDER",
    "CSV_COLUMNS",
    "DEFAULT_MIN_READINESS",
    "IMPLEMENTATION_METADATA",
    "READINESS_ORDER",
    "THRUSTER_FIELD_UNITS",
    "AutonomyStage",
    "Category",
    "ChangeReport",
    "CharacterizationBundle",
    "CharacterizationRecord",
    "GateError",
    "GateLedger",
    "IngestError",
    "LedgerEntry",
    "MergeError",
    "ParameterChange",
    "ReadinessLevel",
    "UnitError",
    "convert_value",
    "load_bundle",
    "load_csv_bundle",
    "load_yaml_bundle",
    "merge_characterization",
    "sha256_file",
]
