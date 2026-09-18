"""Dataset manifest contract, YAML loading and fail-closed verification (ch24, ch36 Data provenance).

A manifest is a procurement and adapter specification. It never claims that data was downloaded,
licensed or accepted unless the corresponding fields carry evidence. Verification returns a list
of problems; an empty list is the only passing result.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from conrad.data.manifest_model import (
    OPEN,
    REVIEW_REQUIRED,
    UNRESOLVED_TOKENS,
    DatasetManifest,
    FileEntry,
    LineageKeys,
    ManifestError,
    ManifestStatus,
    RightsReviewStatus,
    SourceCategory,
    SplitUnit,
    StreamSync,
    SyncStatus,
    TimestampSemantics,
    TransformationDecl,
    TransformationKind,
    TruthAvailability,
    files_digest,
    sha256_file,
)
from conrad.schemas.base import ConradModel
from conrad.settings import REPO_ROOT

__all__ = [
    "OPEN",
    "REVIEW_REQUIRED",
    "UNRESOLVED_TOKENS",
    "DatasetManifest",
    "FileEntry",
    "LineageKeys",
    "ManifestError",
    "ManifestProblem",
    "ManifestStatus",
    "ProblemCode",
    "RightsReviewStatus",
    "SourceCategory",
    "SplitUnit",
    "StreamSync",
    "SyncStatus",
    "TimestampSemantics",
    "TransformationDecl",
    "TransformationKind",
    "TruthAvailability",
    "check_manifest_declarations",
    "dump_manifest",
    "files_digest",
    "find_manifest",
    "load_manifest",
    "parse_manifest",
    "sha256_file",
    "verify_loaded_manifest",
    "verify_manifest",
    "verify_manifest_by_id",
]


class ProblemCode(str, Enum):
    MANIFEST_INVALID = "MANIFEST_INVALID"
    STATUS_NOT_USABLE = "STATUS_NOT_USABLE"
    LICENSE_MISSING = "LICENSE_MISSING"
    RIGHTS_NOT_CLEARED = "RIGHTS_NOT_CLEARED"
    TRAINING_NOT_PERMITTED = "TRAINING_NOT_PERMITTED"
    NO_FILES = "NO_FILES"
    MISSING_FILE = "MISSING_FILE"
    SIZE_MISMATCH = "SIZE_MISMATCH"
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
    DATASET_CHECKSUM_MISMATCH = "DATASET_CHECKSUM_MISMATCH"
    UNDECLARED_FILE = "UNDECLARED_FILE"
    UNDECLARED_TRANSFORMATION = "UNDECLARED_TRANSFORMATION"
    DERIVED_WITHOUT_TRANSFORMATION = "DERIVED_WITHOUT_TRANSFORMATION"
    DANGLING_DERIVATION = "DANGLING_DERIVATION"
    UNDECLARED_MODALITY = "UNDECLARED_MODALITY"
    SYNC_UNKNOWN_STREAM = "SYNC_UNKNOWN_STREAM"
    SPLIT_STRATEGY_MISSING = "SPLIT_STRATEGY_MISSING"
    LINEAGE_MISSING = "LINEAGE_MISSING"
    ADAPTER_VERSION_MISSING = "ADAPTER_VERSION_MISSING"


class ManifestProblem(ConradModel):
    code: ProblemCode
    detail: str
    path: str | None = None

    def __str__(self) -> str:
        where = f" [{self.path}]" if self.path else ""
        return f"{self.code.value}{where}: {self.detail}"


def _unresolved(value: str | None) -> bool:
    return value is None or value.strip().upper() in UNRESOLVED_TOKENS


def parse_manifest(data: dict[str, Any]) -> DatasetManifest:
    try:
        return DatasetManifest.model_validate(data)
    except ValidationError as exc:
        raise ManifestError(f"manifest does not satisfy the contract: {exc}") from exc


def load_manifest(path: str | Path) -> DatasetManifest:
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"manifest not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestError(f"{p} must contain a YAML mapping")
    return parse_manifest(data)


def dump_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False), encoding="utf-8")


def check_manifest_declarations(manifest: DatasetManifest) -> list[ManifestProblem]:
    """Checks that need no data on disk: rights, declared transformations, lineage, streams."""
    problems: list[ManifestProblem] = []

    def add(code: ProblemCode, detail: str, path: str | None = None) -> None:
        problems.append(ManifestProblem(code=code, detail=detail, path=path))

    if manifest.status not in (ManifestStatus.VERIFIED, ManifestStatus.APPROVED):
        add(ProblemCode.STATUS_NOT_USABLE, f"manifest status is {manifest.status.value}")
    if _unresolved(manifest.license) or _unresolved(manifest.license_evidence_ref):
        add(
            ProblemCode.LICENSE_MISSING,
            f"license={manifest.license!r} license_evidence_ref={manifest.license_evidence_ref!r}",
        )
    if manifest.rights_review_status is not RightsReviewStatus.CLEARED or _unresolved(manifest.usage_rights):
        add(
            ProblemCode.RIGHTS_NOT_CLEARED,
            f"rights_review_status={manifest.rights_review_status.value} usage_rights={manifest.usage_rights!r}",
        )
    if not manifest.training_allowed:
        add(ProblemCode.TRAINING_NOT_PERMITTED, "training_allowed is false")
    if _unresolved(manifest.adapter_version):
        add(ProblemCode.ADAPTER_VERSION_MISSING, "adapter_version is not pinned")
    if not manifest.split_strategy:
        add(ProblemCode.SPLIT_STRATEGY_MISSING, "split_strategy declares no lineage unit")
    if not manifest.files:
        add(ProblemCode.NO_FILES, "manifest lists no files")

    declared = {t.transformation_id for t in (*manifest.transformations, *manifest.corruptions)}
    by_path = {f.path for f in manifest.files}
    declared_modalities = set(manifest.modalities)
    for entry in manifest.files:
        for tid in entry.transformation_ids:
            if tid not in declared:
                add(
                    ProblemCode.UNDECLARED_TRANSFORMATION,
                    f"transformation {tid!r} is not declared",
                    entry.path,
                )
        if entry.derived_from is not None:
            if entry.derived_from not in by_path:
                add(
                    ProblemCode.DANGLING_DERIVATION,
                    f"parent {entry.derived_from!r} is not in the manifest",
                    entry.path,
                )
            if not entry.transformation_ids:
                add(
                    ProblemCode.DERIVED_WITHOUT_TRANSFORMATION,
                    "derived file names no transformation; silent transforms are forbidden",
                    entry.path,
                )
        if entry.modality not in declared_modalities:
            add(
                ProblemCode.UNDECLARED_MODALITY,
                f"modality {entry.modality.value} is not declared",
                entry.path,
            )
        if entry.derived_from is None and manifest.split_strategy:
            missing = [u.value for u in manifest.split_strategy if entry.lineage.value(u) is None]
            if len(missing) == len(manifest.split_strategy):
                add(ProblemCode.LINEAGE_MISSING, f"no lineage key for any of {missing}", entry.path)
    streams = set(manifest.streams)
    for sync in manifest.synchronization:
        unknown = sorted(sync.pair - streams)
        if unknown and manifest.files:
            add(ProblemCode.SYNC_UNKNOWN_STREAM, f"synchronization names streams with no files: {unknown}")
    if manifest.files and manifest.checksum != manifest.files_digest():
        add(
            ProblemCode.DATASET_CHECKSUM_MISMATCH,
            f"declared checksum {manifest.checksum!r} != inventory digest {manifest.files_digest()}",
        )
    return problems


def verify_manifest(
    path: str | Path, data_root: str | Path, *, strict_inventory: bool = False
) -> list[ManifestProblem]:
    """Fail-closed verification. An empty list is the only result that permits use of the data."""
    try:
        manifest = load_manifest(path)
    except ManifestError as exc:
        return [ManifestProblem(code=ProblemCode.MANIFEST_INVALID, detail=str(exc), path=str(path))]
    return verify_loaded_manifest(manifest, data_root, strict_inventory=strict_inventory)


def verify_loaded_manifest(
    manifest: DatasetManifest, data_root: str | Path, *, strict_inventory: bool = False
) -> list[ManifestProblem]:
    problems = check_manifest_declarations(manifest)
    root = Path(data_root)
    for entry in manifest.files:
        target = root / entry.path
        if not target.is_file():
            problems.append(
                ManifestProblem(
                    code=ProblemCode.MISSING_FILE, detail="file not found under data root", path=entry.path
                )
            )
            continue
        size = target.stat().st_size
        if size != entry.byte_length:
            problems.append(
                ManifestProblem(
                    code=ProblemCode.SIZE_MISMATCH,
                    detail=f"expected {entry.byte_length} bytes, found {size}",
                    path=entry.path,
                )
            )
        actual = sha256_file(target)
        if actual != entry.sha256:
            problems.append(
                ManifestProblem(
                    code=ProblemCode.CHECKSUM_MISMATCH,
                    detail=f"expected sha256 {entry.sha256}, found {actual}",
                    path=entry.path,
                )
            )
    if strict_inventory and root.is_dir():
        listed = {f.path for f in manifest.files}
        for found in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = found.relative_to(root).as_posix()
            if rel not in listed:
                problems.append(
                    ManifestProblem(
                        code=ProblemCode.UNDECLARED_FILE,
                        detail="file on disk is not in the manifest",
                        path=rel,
                    )
                )
    return problems


DATASETS_DIR = REPO_ROOT / "datasets"
DATA_ROOT_ENV = "CONRAD_DATA_ROOT"


def find_manifest(manifest_id: str, search_dir: Path = DATASETS_DIR) -> Path:
    """``dataset_id`` or ``dataset_id@version`` -> the single matching ``*.manifest.yaml``."""
    dataset_id, _, version = manifest_id.partition("@")
    hits: list[Path] = []
    for path in sorted(search_dir.rglob("*.manifest.yaml")):
        try:
            manifest = load_manifest(path)
        except ManifestError:
            continue
        if manifest.dataset_id == dataset_id and (not version or manifest.version == version):
            hits.append(path)
    if len(hits) != 1:
        raise ManifestError(
            f"manifest {manifest_id!r}: expected one match under {search_dir}, found {len(hits)}"
        )
    return hits[0]


def data_root_for(dataset_id: str) -> Path:
    """Local data lives outside git: ``$CONRAD_DATA_ROOT/<dataset_id>`` or ``artifacts/data/<dataset_id>``."""
    base = os.environ.get(DATA_ROOT_ENV)
    return (Path(base) if base else REPO_ROOT / "artifacts" / "data") / dataset_id


def verify_manifest_by_id(manifest_id: str, search_dir: Path = DATASETS_DIR) -> list[str]:
    """Doctor entry point. Empty list = usable; anything else fails closed."""
    try:
        path = find_manifest(manifest_id, search_dir)
    except ManifestError as exc:
        return [f"{ProblemCode.MANIFEST_INVALID.value}: {exc}"]
    manifest = load_manifest(path)
    return [str(p) for p in verify_loaded_manifest(manifest, data_root_for(manifest.dataset_id))]
