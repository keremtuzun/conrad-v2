"""Parameter identification from logs, with uncertainty and a strict identification/validation split.

Real identification is BLOCKED_EXTERNAL (no physical logs yet). The tooling is checked on SYNTHETIC
logs generated from known parameters; such results are never promoted to RobotConfig values.
"""

from conrad.robotics.hardware.identification.dataset import (
    ExperimentKind,
    IdentificationDataset,
    LogSegment,
    SplitLeakageError,
)
from conrad.robotics.hardware.identification.experiments import (
    DeadzoneBracket,
    bracket_deadzone,
    identify_axis,
    identify_displaced_volume,
    identify_station_keeping_current,
    identify_thruster,
    identify_trim,
)
from conrad.robotics.hardware.identification.fitting import (
    FitResult,
    FittedParameter,
    IdentificationError,
    fit,
)
from conrad.robotics.hardware.identification.logio import LogFormatError, load_csv_segment, load_manifest
from conrad.robotics.hardware.identification.models import (
    G0,
    simulate_axis,
    thrust_command,
    thruster_response,
    trim_moment,
)
from conrad.robotics.hardware.identification.report import (
    REAL_IDENTIFICATION_STATUS,
    IdentificationReport,
    ParameterMapping,
    ReportStatus,
    SyntheticDataError,
    build_report,
    to_characterization_records,
)
from conrad.robotics.hardware.identification.validation import (
    ValidationMetric,
    ValidationResult,
    validate_axis,
    validate_thruster,
)

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch20 Dynamic parameter identification / Identification objective / Parameter uncertainty",
        "ch21 Parameter identification / Parameter uncertainty",
        "ch22 Identification experiment sequence A..F / Validation trajectories",
    ],
    "configuration_keys": ["validation envelope_rmse (per experiment, caller supplied)"],
    "assumptions": [
        "thruster: hard deadzone, T = k u|u| with separate forward/reverse gains, first-order lag, pure delay"
        " (same form as conrad.sim.kernel)",
        "single-axis decoupled dynamics per experiment; cross-coupling is not identified",
        "deadzone is bracketed from step amplitudes (uniform-interval sigma), not least-squares fitted",
        "static trim: pitch-plane moment balance M = B (z_BG sin(theta) + x_BG cos(theta))",
    ],
    "baselines": ["parameters from RobotConfig ENGINEERING_ESTIMATE / SYNTHETIC_ONLY values"],
    "acceptance_tests": ["tests/unit/identification"],
    "claim_status": "IMPLEMENTED",
    "real_identification_status": "BLOCKED_EXTERNAL",
}

__all__ = [
    "G0",
    "IMPLEMENTATION_METADATA",
    "REAL_IDENTIFICATION_STATUS",
    "DeadzoneBracket",
    "ExperimentKind",
    "FitResult",
    "FittedParameter",
    "IdentificationDataset",
    "IdentificationError",
    "IdentificationReport",
    "LogFormatError",
    "LogSegment",
    "ParameterMapping",
    "ReportStatus",
    "SplitLeakageError",
    "SyntheticDataError",
    "ValidationMetric",
    "ValidationResult",
    "bracket_deadzone",
    "build_report",
    "fit",
    "identify_axis",
    "identify_displaced_volume",
    "identify_station_keeping_current",
    "identify_thruster",
    "identify_trim",
    "load_csv_segment",
    "load_manifest",
    "simulate_axis",
    "thrust_command",
    "thruster_response",
    "to_characterization_records",
    "trim_moment",
    "validate_axis",
    "validate_thruster",
]
