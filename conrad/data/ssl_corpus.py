"""DATA-CONRAD-SSL-01: Conrad-owned representation corpus tooling (ch35 Priority 2, ch36 Data).

The corpus is a table of capture units, not a folder of images. The acceptance report tabulates
coverage; an empty real partition is reported as EMPTY / BLOCKED_EXTERNAL and is never padded.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path

from pydantic import Field, model_validator

from conrad.data.manifest import LineageKeys, RightsReviewStatus, SourceCategory, SplitUnit
from conrad.data.splits import SampleRef, find_lineage_leakage
from conrad.schemas.base import VersionedModel, digest_of
from conrad.schemas.observation import Modality

CORPUS_ID = "DATA-CONRAD-SSL-01"
NOT_RECORDED = "NOT_RECORDED"


class Partition(str, Enum):
    SSL_SYNTHETIC_TRAIN = "ssl_synthetic_train"
    SSL_REAL_TRAIN = "ssl_real_train"
    VAL_IN_DOMAIN = "val_in_domain"
    TEST_IN_DOMAIN = "test_in_domain"
    TEST_OOD = "test_ood"


SYNTHETIC_SOURCES = frozenset({SourceCategory.TWIN_SYNTHETIC, SourceCategory.UNITY_SYNTHETIC})
_TRAIN_PARTITIONS = frozenset({Partition.SSL_SYNTHETIC_TRAIN, Partition.SSL_REAL_TRAIN})


class PartitionStatus(str, Enum):
    POPULATED = "POPULATED"
    EMPTY = "EMPTY"
    EMPTY_BLOCKED_EXTERNAL = "EMPTY_BLOCKED_EXTERNAL"


class CaptureUnit(VersionedModel):
    """One manifest row. ``None`` means the field was not recorded; it is never filled in."""

    capture_unit_id: str = Field(min_length=1)
    partition: Partition
    source: SourceCategory
    source_ref: str = Field(min_length=1, description="twin version / capture campaign / dataset id")
    owner: str = Field(min_length=1)
    rights_status: RightsReviewStatus
    modalities: tuple[Modality, ...] = Field(min_length=1)
    expected_modalities: tuple[Modality, ...] = ()
    calibration_ref: str | None = None
    clock_domain: str = Field(min_length=1)
    time_start_ns: int | None = Field(default=None, ge=0)
    time_end_ns: int | None = Field(default=None, ge=0)
    pose_frame: str | None = None
    pose_available: bool = False
    scenario_id: str | None = None
    mission_id: str | None = None
    augmentation_eligible: bool = False
    lineage: LineageKeys
    derived_from: str | None = None
    source_object_digest: str = Field(pattern="^[0-9a-f]{64}$")
    preprocessing_graph: tuple[str, ...] = ()
    scenario_regime: str | None = None
    structural_condition: str | None = None
    visibility: str | None = None
    turbidity_noise: str | None = None
    viewpoint: str | None = None
    label_availability: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _rules(self) -> CaptureUnit:
        if not self.lineage.declared():
            raise ValueError(f"capture unit {self.capture_unit_id} has no split lineage")
        if self.scenario_id is None and self.mission_id is None:
            raise ValueError(f"capture unit {self.capture_unit_id} needs a scenario_id or mission_id")
        if self.partition is Partition.SSL_SYNTHETIC_TRAIN and self.source not in SYNTHETIC_SOURCES:
            raise ValueError("ssl_synthetic_train accepts only versioned Twin/Unity observations")
        if self.partition is Partition.SSL_REAL_TRAIN:
            if self.source is not SourceCategory.CONRAD_REAL:
                raise ValueError("ssl_real_train accepts only Conrad-collected sequences")
            if self.rights_status is not RightsReviewStatus.CLEARED:
                raise ValueError("ssl_real_train accepts only rights-cleared capture units")
            if self.label_availability:
                raise ValueError("ssl_real_train is unlabelled; labels must not be attached")
        if self.source not in SYNTHETIC_SOURCES and self.rights_status is not RightsReviewStatus.CLEARED:
            raise ValueError(f"real capture unit {self.capture_unit_id} is not rights-cleared")
        if (
            self.time_start_ns is not None
            and self.time_end_ns is not None
            and self.time_end_ns < self.time_start_ns
        ):
            raise ValueError("time_end_ns precedes time_start_ns")
        return self

    @property
    def missing_modality_pattern(self) -> str:
        if not self.expected_modalities:
            return NOT_RECORDED
        missing = sorted(m.value for m in set(self.expected_modalities) - set(self.modalities))
        return "MISSING:" + "+".join(missing) if missing else "NONE_MISSING"


class CorpusAcceptanceReport(VersionedModel):
    corpus_id: str = CORPUS_ID
    rows_digest: str
    capture_unit_count: int
    partition_status: dict[str, PartitionStatus]
    partition_counts: dict[str, int]
    real_data_status: str
    coverage: dict[str, dict[str, dict[str, int]]] = Field(
        description="dimension -> partition -> value -> capture-unit count"
    )
    lineage_findings: tuple[str, ...]
    blocking: tuple[str, ...]
    accepted: bool


def _dimension_values(unit: CaptureUnit) -> dict[str, list[str]]:
    return {
        "modality": [m.value for m in unit.modalities],
        "source": [unit.source.value],
        "scenario_regime": [unit.scenario_regime or NOT_RECORDED],
        "structural_condition": [unit.structural_condition or NOT_RECORDED],
        "visibility": [unit.visibility or NOT_RECORDED],
        "turbidity_noise": [unit.turbidity_noise or NOT_RECORDED],
        "viewpoint": [unit.viewpoint or NOT_RECORDED],
        "missing_modality_pattern": [unit.missing_modality_pattern],
        "label_availability": list(unit.label_availability) or ["UNLABELLED"],
        "calibration": ["CALIBRATED_REF" if unit.calibration_ref else "NO_CALIBRATION"],
        "pose": ["POSE_AVAILABLE" if unit.pose_available else "NO_POSE"],
        "asset_lineage": [unit.lineage.asset or unit.lineage.world_family or NOT_RECORDED],
    }


def corpus_lineage_findings(units: Sequence[CaptureUnit]) -> list[str]:
    """Training partitions are pooled; every other partition must be lineage-disjoint from the rest."""
    splits: dict[str, list[SampleRef]] = {}
    for unit in units:
        name = "train" if unit.partition in _TRAIN_PARTITIONS else unit.partition.value
        splits.setdefault(name, []).append(
            SampleRef(
                sample_id=unit.capture_unit_id,
                lineage=unit.lineage,
                derived_from=unit.derived_from,
                content_sha256=unit.source_object_digest,
            )
        )
    units_checked = [u for u in SplitUnit if u is not SplitUnit.SEED]
    return find_lineage_leakage(splits, units_checked)


def build_corpus_acceptance_report(units: Sequence[CaptureUnit]) -> CorpusAcceptanceReport:
    ids = [u.capture_unit_id for u in units]
    blocking: list[str] = []
    if len(set(ids)) != len(ids):
        blocking.append("duplicate capture_unit_id values")
    counts = Counter(u.partition for u in units)
    has_real = any(u.source is SourceCategory.CONRAD_REAL for u in units)
    status: dict[str, PartitionStatus] = {}
    for partition in Partition:
        if counts[partition] > 0:
            status[partition.value] = PartitionStatus.POPULATED
        elif partition is Partition.SSL_REAL_TRAIN:
            status[partition.value] = PartitionStatus.EMPTY_BLOCKED_EXTERNAL
        else:
            status[partition.value] = PartitionStatus.EMPTY
        if counts[partition] == 0:
            blocking.append(f"partition {partition.value} is {status[partition.value].value}")
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    for unit in units:
        for dimension, values in _dimension_values(unit).items():
            cell = coverage.setdefault(dimension, {}).setdefault(unit.partition.value, {})
            for value in values:
                cell[value] = cell.get(value, 0) + 1
    findings = corpus_lineage_findings(units) if len(set(ids)) == len(ids) else []
    blocking.extend(f"lineage: {f}" for f in findings)
    return CorpusAcceptanceReport(
        rows_digest=digest_of(sorted(u.canonical_json() for u in units)),
        capture_unit_count=len(units),
        partition_status=status,
        partition_counts={p.value: counts[p] for p in Partition},
        real_data_status="PRESENT" if has_real else "BLOCKED_EXTERNAL_NO_CONRAD_REAL_CAPTURE",
        coverage={
            d: {p: dict(sorted(v.items())) for p, v in sorted(t.items())} for d, t in sorted(coverage.items())
        },
        lineage_findings=tuple(findings),
        blocking=tuple(blocking),
        accepted=not blocking,
    )


def render_report_markdown(report: CorpusAcceptanceReport) -> str:
    lines = [
        f"# {report.corpus_id} corpus acceptance report",
        "",
        f"- capture units: {report.capture_unit_count}",
        f"- rows digest: `{report.rows_digest}`",
        f"- real data status: {report.real_data_status}",
        f"- accepted: {report.accepted}",
        "",
        "## Partitions",
        "",
        "| partition | capture units | status |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {p.value} | {report.partition_counts[p.value]} | {report.partition_status[p.value].value} |"
        for p in Partition
    )
    for dimension, table in report.coverage.items():
        lines += [
            "",
            f"## Coverage by {dimension}",
            "",
            "| partition | value | capture units |",
            "|---|---|---|",
        ]
        for partition, cells in table.items():
            lines.extend(f"| {partition} | {value} | {count} |" for value, count in cells.items())
    if report.blocking:
        lines += ["", "## Blocking", ""]
        lines.extend(f"- {b}" for b in report.blocking)
    return "\n".join(lines) + "\n"


def write_capture_units(units: Iterable[CaptureUnit], path: str | Path) -> None:
    Path(path).write_text("".join(u.canonical_json() + "\n" for u in units), encoding="utf-8")


def read_capture_units(path: str | Path) -> list[CaptureUnit]:
    rows: list[CaptureUnit] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(CaptureUnit.model_validate(json.loads(line)))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: invalid capture unit row: {exc}") from exc
    return rows
