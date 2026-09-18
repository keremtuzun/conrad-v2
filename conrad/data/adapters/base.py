"""DatasetAdapter contract (ch24). Any dataset becomes shared ``Observation`` objects plus an
optional, physically separate supervision channel (:class:`PartialTruth`).

``PartialTruth`` and ``SequenceSample`` carry arrays, so they are frozen dataclasses rather than
pydantic messages; they never cross a public message boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from conrad.data.manifest import DatasetManifest, LineageKeys, ManifestProblem
from conrad.schemas.base import ConradModel
from conrad.schemas.observation import Observation


class DatasetContractError(ValueError):
    """The dataset cannot satisfy the Observation contract without fabricating a field."""


class DatasetNotAvailableError(RuntimeError):
    """The dataset is not present / not verified. Names exactly what is needed."""

    def __init__(
        self,
        dataset_id: str,
        manifest_path: str | Path,
        required: Sequence[str],
        problems: Sequence[ManifestProblem] = (),
    ) -> None:
        self.dataset_id = dataset_id
        self.manifest_path = str(manifest_path)
        self.required = tuple(required)
        self.problems = tuple(problems)
        needs = "; ".join(self.required) if self.required else "see manifest"
        issues = "; ".join(str(p) for p in self.problems[:8])
        super().__init__(
            f"dataset {dataset_id!r} is not available. Manifest: {self.manifest_path}. "
            f"Required before use: {needs}." + (f" Verification problems: {issues}" if issues else "")
        )


@dataclass(frozen=True)
class PartialTruth:
    """Supervision with explicit availability. A target without a mask entry is a contract error."""

    targets: Mapping[str, np.ndarray]
    availability_masks: Mapping[str, np.ndarray]
    label_source_refs: Mapping[str, str]
    quality_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = set(self.targets) - set(self.availability_masks)
        if missing:
            raise DatasetContractError(f"targets without availability masks: {sorted(missing)}")
        unsourced = set(self.targets) - set(self.label_source_refs)
        if unsourced:
            raise DatasetContractError(f"targets without label_source_refs: {sorted(unsourced)}")
        for name, mask in self.availability_masks.items():
            if mask.dtype != np.bool_:
                raise DatasetContractError(f"availability mask {name!r} must be boolean")
            if name in self.targets and mask.shape[0] != self.targets[name].shape[0]:
                raise DatasetContractError(f"mask {name!r} does not match its target along axis 0")


@dataclass(frozen=True)
class SequenceSample:
    """One lineage-coherent sequence. ``supervision`` never enters an inference input."""

    dataset_id: str
    manifest_digest: str
    sequence_id: str
    lineage: LineageKeys
    observations: tuple[Observation, ...]
    supervision: PartialTruth | None = None
    split: str | None = None

    def __post_init__(self) -> None:
        by_clock: dict[tuple[str, str], int] = {}
        for obs in self.observations:
            key = (str(obs.sensor_id), obs.timestamp.clock_domain)
            last = by_clock.get(key)
            if last is not None and obs.timestamp.time_ns < last:
                raise DatasetContractError(
                    f"sequence {self.sequence_id!r}: timestamps of sensor {obs.sensor_id} are not ordered"
                )
            by_clock[key] = obs.timestamp.time_ns


class DatasetAudit(ConradModel):
    dataset_id: str
    manifest_digest: str
    adapter_name: str
    adapter_version: str
    usable: bool
    problems: tuple[ManifestProblem, ...] = ()
    sequence_count: int | None = None
    file_count: int = 0
    modalities: tuple[str, ...] = ()
    label_names: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class DatasetAdapter(ABC):
    """Converts one dataset into the shared observation schema. Subclasses must not be no-ops."""

    adapter_name: str = "abstract"
    adapter_version: str = "0"

    def __init__(self, manifest: DatasetManifest) -> None:
        self.manifest = manifest

    @abstractmethod
    def inspect(self) -> DatasetAudit: ...

    @abstractmethod
    def iter_sequences(self, split: str | None = None) -> Iterator[SequenceSample]: ...

    @abstractmethod
    def normalize_observation(self, raw: Any) -> Observation: ...

    @abstractmethod
    def map_labels(self, raw_labels: Any) -> PartialTruth | None: ...
