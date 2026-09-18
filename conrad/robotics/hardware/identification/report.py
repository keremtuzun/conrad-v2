"""Identification report and the hand-off to characterization records (IDENTIFIED parameters).

A report built from SYNTHETIC logs proves the tooling only: it can never be turned into IDENTIFIED
RobotConfig values. Real identification is BLOCKED_EXTERNAL until the hardware owner delivers logs.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from conrad.robotics.hardware.characterization.records import Category, CharacterizationRecord
from conrad.robotics.hardware.identification.dataset import IdentificationDataset
from conrad.robotics.hardware.identification.fitting import FitResult
from conrad.robotics.hardware.identification.validation import ValidationResult
from conrad.schemas.base import ConradModel, digest_of
from conrad.schemas.robot import SourceKind

REAL_IDENTIFICATION_STATUS = "BLOCKED_EXTERNAL"  # no physical logs exist yet (hardware owner deliverable)


class ReportStatus(str, Enum):
    SYNTHETIC_TOOLING_CHECK = "SYNTHETIC_TOOLING_CHECK"
    IDENTIFIED_NOT_VALIDATED = "IDENTIFIED_NOT_VALIDATED"
    IDENTIFIED_AND_VALIDATED = "IDENTIFIED_AND_VALIDATED"
    VALIDATION_FAILED = "VALIDATION_FAILED"


class SyntheticDataError(ValueError):
    """Synthetic identification results were about to be promoted to physical parameters."""


class ParameterMapping(ConradModel):
    """Where a scalar fitted parameter lands in the RobotConfig (vector targets are assembled by the caller)."""

    fit_name: str
    target: str
    category: Category
    scale: float = 1.0


class IdentificationReport(ConradModel):
    report_version: str = "identification.v1"
    status: ReportStatus
    dataset_lineage: dict[str, Any]
    fits: dict[str, FitResult]
    validations: dict[str, ValidationResult] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()
    real_identification_status: str = REAL_IDENTIFICATION_STATUS

    def digest(self) -> str:
        return digest_of(self.model_dump(mode="json"))

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        return p


def build_report(
    dataset: IdentificationDataset,
    fits: dict[str, FitResult],
    validations: dict[str, ValidationResult],
    notes: tuple[str, ...] = (),
) -> IdentificationReport:
    if dataset.any_synthetic:
        status = ReportStatus.SYNTHETIC_TOOLING_CHECK
    elif not validations:
        status = ReportStatus.IDENTIFIED_NOT_VALIDATED
    elif all(v.within_envelope is not False for v in validations.values()):
        status = ReportStatus.IDENTIFIED_AND_VALIDATED
    else:
        status = ReportStatus.VALIDATION_FAILED
    return IdentificationReport(
        status=status,
        dataset_lineage=dataset.lineage(),
        fits=fits,
        validations=validations,
        notes=notes,
        real_identification_status="N/A (real logs)"
        if not dataset.any_synthetic
        else REAL_IDENTIFICATION_STATUS,
    )


def to_characterization_records(
    report: IdentificationReport, report_path: str, mappings: tuple[ParameterMapping, ...]
) -> tuple[CharacterizationRecord, ...]:
    """Scalar IDENTIFIED records for :func:`merge_characterization`. Refuses synthetic reports.

    Vector targets (e.g. drag per axis) are assembled by the caller into one record per target.
    """
    if report.status is ReportStatus.SYNTHETIC_TOOLING_CHECK:
        raise SyntheticDataError("a synthetic identification report cannot produce IDENTIFIED parameters")
    if report.status is ReportStatus.VALIDATION_FAILED:
        raise ValueError("held-out validation failed; parameters are not promoted")
    out: list[CharacterizationRecord] = []
    for m in mappings:
        fit = next((f for f in report.fits.values() if any(p.name == m.fit_name for p in f.parameters)), None)
        if fit is None:
            raise KeyError(f"no fitted parameter {m.fit_name!r} in the report")
        p = fit.get(m.fit_name)
        out.append(
            CharacterizationRecord(
                record_id=f"ident-{m.fit_name}",
                category=m.category,
                target=m.target,
                value=p.value * m.scale,
                units=p.units,
                uncertainty_1sigma=None if p.sigma is None else abs(p.sigma * m.scale),
                valid_range=None if p.ci95 is None else (p.ci95[0] * m.scale, p.ci95[1] * m.scale),
                source=SourceKind.IDENTIFIED,
                method=f"least-squares identification ({report.report_version})",
                provenance=f"{report_path}@sha256:{report.digest()}",
            )
        )
    return tuple(out)
