"""Dataset manifest data model (ch24 Manifest and Adapter Contract, ch36 Data provenance).

Pure contract: enums, lineage keys, file entries and the manifest itself. Loading and fail-closed
verification live in :mod:`conrad.data.manifest`, which re-exports everything here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import Enum
from pathlib import Path, PurePosixPath

from pydantic import Field, field_validator, model_validator

from conrad.schemas.base import ConradModel, VersionedModel, digest_of
from conrad.schemas.observation import Modality

OPEN = "OPEN"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
UNRESOLVED_TOKENS = frozenset({"", OPEN, REVIEW_REQUIRED, "UNKNOWN", "TBD", "NONE"})
_HASH_CHUNK = 1 << 20


class ManifestError(ValueError):
    """The manifest file cannot be parsed into the contract at all."""


class ManifestStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    UNVERIFIED_NOT_DOWNLOADED = "UNVERIFIED_NOT_DOWNLOADED"
    VERIFIED = "VERIFIED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class RightsReviewStatus(str, Enum):
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    IN_REVIEW = "IN_REVIEW"
    CLEARED = "CLEARED"
    REJECTED = "REJECTED"


class SourceCategory(str, Enum):
    PUBLIC_REAL = "PUBLIC_REAL"
    CONRAD_REAL = "CONRAD_REAL"
    TWIN_SYNTHETIC = "TWIN_SYNTHETIC"
    UNITY_SYNTHETIC = "UNITY_SYNTHETIC"


class TruthAvailability(str, Enum):
    NONE = "NONE"
    PARTIAL_LABELS = "PARTIAL_LABELS"
    FULL_SYNTHETIC = "FULL_SYNTHETIC"
    UNVERIFIED = "UNVERIFIED"


class SyncStatus(str, Enum):
    UNSYNCHRONIZED = "UNSYNCHRONIZED"
    HARDWARE_SYNCHRONIZED = "HARDWARE_SYNCHRONIZED"
    SOFTWARE_ALIGNED = "SOFTWARE_ALIGNED"
    SYNTHETIC_PAIRED = "SYNTHETIC_PAIRED"


class TransformationKind(str, Enum):
    TRANSFORM = "TRANSFORM"
    CORRUPTION = "CORRUPTION"


class SplitUnit(str, Enum):
    SEQUENCE = "sequence"
    DIVE = "dive"
    SITE = "site"
    ASSET = "asset"
    GEOMETRY_FAMILY = "geometry_family"
    WORLD_FAMILY = "world_family"
    TRAJECTORY = "trajectory"
    SEED = "seed"
    SESSION = "session"


class LineageKeys(ConradModel):
    """Every key that can make two samples non-independent. ``None`` means not recorded."""

    sequence: str | None = None
    dive: str | None = None
    site: str | None = None
    asset: str | None = None
    geometry_family: str | None = None
    world_family: str | None = None
    trajectory: str | None = None
    seed: str | None = None
    session: str | None = None

    def value(self, unit: SplitUnit) -> str | None:
        value: str | None = getattr(self, unit.value)
        return value

    def declared(self) -> dict[str, str]:
        return {u.value: v for u in SplitUnit if (v := self.value(u)) is not None}


class TransformationDecl(ConradModel):
    transformation_id: str = Field(min_length=1)
    kind: TransformationKind
    description: str = Field(min_length=1)
    version: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)


class StreamSync(ConradModel):
    """Declared relation between two streams. Never assumed: default is UNSYNCHRONIZED."""

    stream_a: str = Field(min_length=1)
    stream_b: str = Field(min_length=1)
    status: SyncStatus = SyncStatus.UNSYNCHRONIZED
    evidence_ref: str | None = None
    max_offset_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _evidence(self) -> StreamSync:
        if self.stream_a == self.stream_b:
            raise ValueError("a synchronization entry needs two distinct streams")
        if self.status is not SyncStatus.UNSYNCHRONIZED and not self.evidence_ref:
            raise ValueError(
                f"streams {self.stream_a!r}/{self.stream_b!r} claim {self.status.value} without evidence_ref"
            )
        return self

    @property
    def pair(self) -> frozenset[str]:
        return frozenset((self.stream_a, self.stream_b))


class TimestampSemantics(ConradModel):
    clock_domain: str = OPEN
    units: str = OPEN
    semantics: str = Field(default=OPEN, description="e.g. exposure start, ping transmit, file mtime")
    offset_evidence_ref: str | None = None


class FileEntry(ConradModel):
    path: str = Field(min_length=1, description="POSIX path relative to the data root")
    sha256: str = Field(pattern="^[0-9a-f]{64}$")
    byte_length: int = Field(ge=0)
    stream: str = Field(min_length=1)
    modality: Modality
    lineage: LineageKeys = LineageKeys()
    derived_from: str | None = Field(default=None, description="path of the parent file entry")
    transformation_ids: tuple[str, ...] = ()
    time_ns: int | None = Field(default=None, ge=0)
    sequence_index: int | None = Field(default=None, ge=0)

    @field_validator("path")
    @classmethod
    def _relative(cls, v: str) -> str:
        p = PurePosixPath(v.replace("\\", "/"))
        if p.is_absolute() or ".." in p.parts or ":" in v:
            raise ValueError(f"file path must be relative and stay inside the data root: {v!r}")
        return str(p)


class DatasetManifest(VersionedModel):
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    manifest_version: int = 1
    status: ManifestStatus = ManifestStatus.CANDIDATE
    source_category: SourceCategory
    source: str = Field(default=OPEN, description="source URL / capture campaign / twin generator")
    release_id: str = OPEN
    retrieved_at: str = OPEN
    license: str = REVIEW_REQUIRED
    license_evidence_ref: str = OPEN
    usage_rights: str = REVIEW_REQUIRED
    rights_review_status: RightsReviewStatus = RightsReviewStatus.REVIEW_REQUIRED
    rights_by_asset_ref: str = OPEN
    redistribution_allowed: bool = False
    deployment_allowed: bool = False
    training_allowed: bool = False
    checksum: str = Field(default=OPEN, description="sha256 over the file inventory; see files_digest()")
    modalities: tuple[Modality, ...] = ()
    labels: tuple[str, ...] = ()
    label_schema_ref: str = OPEN
    label_quality_report_ref: str = OPEN
    units: dict[str, str] = Field(default_factory=dict)
    frames: dict[str, str] = Field(default_factory=dict, description="stream -> coordinate frame id")
    coordinate_frames_ref: str = OPEN
    calibration_ref: str = OPEN
    timestamps: TimestampSemantics = TimestampSemantics()
    synchronization: tuple[StreamSync, ...] = ()
    truth_availability: TruthAvailability = TruthAvailability.UNVERIFIED
    missingness_policy: str = OPEN
    split_strategy: tuple[SplitUnit, ...] = ()
    split_manifest_ref: str = OPEN
    duplicate_audit_ref: str = OPEN
    transformations: tuple[TransformationDecl, ...] = ()
    corruptions: tuple[TransformationDecl, ...] = ()
    adapter_version: str = OPEN
    transform_config_hash: str = OPEN
    permitted_tasks: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    intended_role: str | None = None
    remaining_verification: tuple[str, ...] = ()
    required_files_note: str | None = None
    files: tuple[FileEntry, ...] = ()

    @model_validator(mode="after")
    def _structure(self) -> DatasetManifest:
        paths = [f.path for f in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate file paths in manifest")
        ids = [t.transformation_id for t in (*self.transformations, *self.corruptions)]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate transformation/corruption ids")
        for t in self.transformations:
            if t.kind is not TransformationKind.TRANSFORM:
                raise ValueError(
                    f"{t.transformation_id} listed under transformations but kind is {t.kind.value}"
                )
        for t in self.corruptions:
            if t.kind is not TransformationKind.CORRUPTION:
                raise ValueError(f"{t.transformation_id} listed under corruptions but kind is {t.kind.value}")
        pairs = [s.pair for s in self.synchronization]
        if len(set(pairs)) != len(pairs):
            raise ValueError("a stream pair is declared more than once in synchronization")
        return self

    @property
    def streams(self) -> tuple[str, ...]:
        return tuple(sorted({f.stream for f in self.files}))

    def files_digest(self) -> str:
        return files_digest(self.files)

    def manifest_digest(self) -> str:
        """Identity of the whole manifest; this is what checkpoints and runs record."""
        return self.content_digest()

    def sync_status(self, stream_a: str, stream_b: str) -> SyncStatus:
        """Undeclared pairs are UNSYNCHRONIZED. Unrelated streams are never presented as paired."""
        key = frozenset((stream_a, stream_b))
        for s in self.synchronization:
            if s.pair == key:
                return s.status
        return SyncStatus.UNSYNCHRONIZED

    def resolve_lineage(self, entry: FileEntry) -> LineageKeys:
        """Derived files inherit the lineage of their root source; their own keys cannot replace it."""
        by_path = {f.path: f for f in self.files}
        seen: set[str] = set()
        current = entry
        while current.derived_from is not None:
            if current.path in seen:
                raise ManifestError(f"derived_from cycle through {current.path}")
            seen.add(current.path)
            parent = by_path.get(current.derived_from)
            if parent is None:
                raise ManifestError(f"{current.path} derives from unknown file {current.derived_from}")
            current = parent
        return current.lineage


def files_digest(files: Iterable[FileEntry]) -> str:
    return digest_of(sorted([f.path, f.sha256, f.byte_length] for f in files))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()
