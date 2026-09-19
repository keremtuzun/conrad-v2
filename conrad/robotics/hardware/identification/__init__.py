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
from conrad.robotics.hardware.identification.intake import (
    IntakeConfig,
    IntakeFinding,
    IntakeReport,
    RepeatabilityStat,
    check_delivery,
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
    validate_station_keeping,
    validate_thruster,
    validate_trim,
)

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch20 Dynamic parameter identification / Identification objective / Parameter uncertainty",
        "ch21 Parameter identification / Parameter uncertainty",
        "ch22 Identification experiment sequence A..F / Validation trajectories",
        "ch25 Phase 13 Parameter Identification (S2R-ID-01, S2R-VALIDATE-01); EXT-HW-04 log protocol",
    ],
    "configuration_keys": [
        "validation envelope_rmse (per experiment, caller supplied; OPEN)",
        "IntakeConfig (repetitions, clock offset, gap factor, outlier z; ENGINEERING_ESTIMATE)",
        "RunnerConfig (hardware_control_enabled=False, safety_envelope, operator_acknowledged, attitude_hold)",
        "PipelineConfig (fit_rate_hz, settled fractions, envelope_rmse)",
    ],
    "assumptions": [
        "thruster: hard deadzone, T = k u|u| with separate forward/reverse gains, first-order lag, pure delay"
        " (same form as conrad.sim.kernel)",
        "single-axis decoupled dynamics per experiment; cross-coupling is not identified",
        "deadzone is bracketed from step amplitudes (uniform-interval sigma), not least-squares fitted",
        "static trim: pitch-plane moment balance M = B (z_BG sin(theta) + x_BG cos(theta))",
        "axis forces are derived from logged commands through the identified B thruster models and the"
        " RobotConfig thruster geometry",
        "station keeping: the current is along the body surge axis and the hold force balances surge drag",
    ],
    "baselines": ["parameters from RobotConfig ENGINEERING_ESTIMATE / SYNTHETIC_ONLY values"],
    "acceptance_tests": ["tests/unit/identification", "tests/unit/robotics/identification"],
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
    "IntakeConfig",
    "IntakeFinding",
    "IntakeReport",
    "LogFormatError",
    "LogSegment",
    "ParameterMapping",
    "RepeatabilityStat",
    "ReportStatus",
    "SplitLeakageError",
    "SyntheticDataError",
    "ValidationMetric",
    "ValidationResult",
    "bracket_deadzone",
    "build_report",
    "check_delivery",
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
    "validate_station_keeping",
    "validate_thruster",
    "validate_trim",
]
