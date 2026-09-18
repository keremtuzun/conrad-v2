"""SIMREAL-LEDGER-01: measured sim-to-real and randomization ledger (ch35 Priority 4).

Every randomized simulator parameter is a ledger row. UNMEASURED is a valid state, but it is
never treated as calibrated. A range that cannot be explained in physical terms is rejected.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import Field, ValidationError, model_validator

from conrad.schemas.base import ConradModel, VersionedModel
from conrad.schemas.robot import SourceKind

LEDGER_ID = "SIMREAL-LEDGER-01"
MIN_EXPLANATION_WORDS = 6
_EMPTY_EXPLANATIONS = frozenset({"tbd", "todo", "n/a", "na", "none", "open", "arbitrary", "guess", "default"})


class LedgerError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("sim-to-real ledger rejected: " + "; ".join(problems))
        self.problems = tuple(problems)


class ParameterCategory(str, Enum):
    OPTICS = "OPTICS"  # lighting / attenuation / backscatter / turbidity
    SONAR = "SONAR"
    CAMERA = "CAMERA"
    POSE = "POSE"
    CURRENT = "CURRENT"
    RIGID_BODY = "RIGID_BODY"  # mass / inertia / buoyancy
    THRUSTER = "THRUSTER"
    POWER = "POWER"
    COMMUNICATION = "COMMUNICATION"
    SENSOR_FAULT = "SENSOR_FAULT"


class SamplingLaw(str, Enum):
    UNIFORM = "UNIFORM"
    LOG_UNIFORM = "LOG_UNIFORM"
    NORMAL = "NORMAL"
    TRUNCATED_NORMAL = "TRUNCATED_NORMAL"
    BERNOULLI = "BERNOULLI"
    CATEGORICAL = "CATEGORICAL"


class MeasurementStatus(str, Enum):
    UNMEASURED = "UNMEASURED"
    MEASURED = "MEASURED"
    IDENTIFIED = "IDENTIFIED"


class RandomizedParameter(ConradModel):
    semantic_name: str = Field(min_length=1)
    category: ParameterCategory
    unit: str = Field(min_length=1)
    range_low: float
    range_high: float
    sampling_law: SamplingLaw
    prior_parameters: dict[str, float] = Field(default_factory=dict)
    source_of_range: SourceKind
    physical_explanation: str = ""
    source_reference: str | None = None
    affected_simulator_module: str = Field(min_length=1)
    physical_measurement_status: MeasurementStatus = MeasurementStatus.UNMEASURED
    calibration_revision: str | None = None
    raw_log_provenance: str | None = None
    identification_report_ref: str | None = None

    @property
    def calibrated(self) -> bool:
        return self.physical_measurement_status is not MeasurementStatus.UNMEASURED


class SimRealLedger(VersionedModel):
    ledger_id: str = LEDGER_ID
    revision: str = Field(min_length=1)
    parameters: tuple[RandomizedParameter, ...]

    @model_validator(mode="after")
    def _unique(self) -> SimRealLedger:
        names = [p.semantic_name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("duplicate semantic_name in ledger")
        return self

    def get(self, semantic_name: str) -> RandomizedParameter:
        for p in self.parameters:
            if p.semantic_name == semantic_name:
                return p
        raise KeyError(
            f"{semantic_name!r} is not in the ledger; an unlisted parameter must not be randomized"
        )

    def unmeasured(self) -> tuple[str, ...]:
        return tuple(p.semantic_name for p in self.parameters if not p.calibrated)


def validate_parameter(p: RandomizedParameter) -> list[str]:
    problems: list[str] = []
    name = p.semantic_name
    text = p.physical_explanation.strip()
    if text.lower() in _EMPTY_EXPLANATIONS or len(text.split()) < MIN_EXPLANATION_WORDS:
        problems.append(f"{name}: range has no physical explanation")
    if not (p.range_low <= p.range_high):
        problems.append(f"{name}: range_low {p.range_low} exceeds range_high {p.range_high}")
    if p.sampling_law is SamplingLaw.LOG_UNIFORM and p.range_low <= 0:
        problems.append(f"{name}: LOG_UNIFORM needs a strictly positive range")
    if p.sampling_law is SamplingLaw.BERNOULLI and not (p.range_low >= 0.0 and p.range_high <= 1.0):
        problems.append(f"{name}: BERNOULLI probability range must lie in [0, 1]")
    if p.sampling_law in (SamplingLaw.NORMAL, SamplingLaw.TRUNCATED_NORMAL):
        missing = {"mean", "std"} - set(p.prior_parameters)
        if missing:
            problems.append(f"{name}: {p.sampling_law.value} needs prior_parameters {sorted(missing)}")
    if p.source_of_range is SourceKind.OPEN:
        problems.append(f"{name}: source_of_range is OPEN; an unsourced range cannot be randomized")
    if p.source_of_range is SourceKind.LITERATURE_PRIOR and not p.source_reference:
        problems.append(f"{name}: LITERATURE_PRIOR needs a source_reference")
    measured_source = p.source_of_range in (SourceKind.MEASURED, SourceKind.IDENTIFIED)
    if p.calibrated:
        if not p.raw_log_provenance:
            problems.append(f"{name}: {p.physical_measurement_status.value} needs raw_log_provenance")
        if not p.identification_report_ref:
            problems.append(f"{name}: a measured prior needs a versioned identification_report_ref")
        if not p.calibration_revision:
            problems.append(f"{name}: a measured prior needs a calibration_revision")
        if not measured_source:
            problems.append(f"{name}: measured status but source_of_range is {p.source_of_range.value}")
    elif measured_source:
        problems.append(f"{name}: source_of_range {p.source_of_range.value} while status is UNMEASURED")
    return problems


def validate_ledger(ledger: SimRealLedger) -> list[str]:
    return [problem for p in ledger.parameters for problem in validate_parameter(p)]


def load_ledger(path: str | Path) -> SimRealLedger:
    """Load and validate; any problem raises :class:`LedgerError` (fail closed)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise LedgerError([f"{path} must contain a YAML mapping"])
    try:
        ledger = SimRealLedger.model_validate(data)
    except ValidationError as exc:
        raise LedgerError([str(exc)]) from exc
    problems = validate_ledger(ledger)
    if problems:
        raise LedgerError(problems)
    return ledger


class RangeRequest(ConradModel):
    semantic_name: str
    low: float
    high: float


def check_randomization_request(ledger: SimRealLedger, requests: list[RangeRequest]) -> list[str]:
    """A simulator may randomize only listed parameters and only inside the ledger range."""
    problems: list[str] = []
    for request in requests:
        try:
            p = ledger.get(request.semantic_name)
        except KeyError as exc:
            problems.append(str(exc.args[0]))
            continue
        if request.low < p.range_low or request.high > p.range_high:
            problems.append(
                f"{p.semantic_name}: requested [{request.low}, {request.high}] {p.unit} leaves the ledger range "
                f"[{p.range_low}, {p.range_high}]"
            )
    return problems
