"""Generic adapter for rights-cleared local RGB sequences.

Bytes are stored unchanged in the content-addressed object store; nothing is resized, recoded
or relabelled. Pose and calibration stay ``None`` when the manifest does not provide them, and
no label is ever produced.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import cv2
import numpy as np

from conrad.data.adapters.base import (
    DatasetAdapter,
    DatasetAudit,
    DatasetContractError,
    DatasetNotAvailableError,
    PartialTruth,
    SequenceSample,
)
from conrad.data.manifest import OPEN, DatasetManifest, FileEntry, verify_loaded_manifest
from conrad.data.splits import SplitAssignment
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.timebase import TimeStamp

_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".bmp": "image/bmp"}


class ImageFolderSequenceAdapter(DatasetAdapter):
    adapter_name = "image_folder_sequence"
    adapter_version = "1.0.0"

    def __init__(
        self,
        manifest: DatasetManifest,
        data_root: str | Path,
        store: ObjectStore,
        ids: IdFactory,
        run_id: UUID,
        *,
        stream: str = "rgb",
        manifest_path: str | Path = "<in-memory>",
        split_assignment: SplitAssignment | None = None,
    ) -> None:
        super().__init__(manifest)
        self.data_root = Path(data_root)
        self.store = store
        self.ids = ids
        self.run_id = run_id
        self.stream = stream
        self.manifest_path = str(manifest_path)
        self.split_assignment = split_assignment
        self._problems = verify_loaded_manifest(manifest, self.data_root)
        if manifest.adapter_version != self.adapter_version:
            raise DatasetContractError(
                f"manifest pins adapter_version {manifest.adapter_version!r}, this adapter is {self.adapter_version!r}"
            )

    def _entries(self) -> list[FileEntry]:
        return [f for f in self.manifest.files if f.stream == self.stream and f.modality is Modality.RGB]

    def inspect(self) -> DatasetAudit:
        entries = self._entries()
        sequences = {self.manifest.resolve_lineage(f).sequence for f in entries}
        return DatasetAudit(
            dataset_id=self.manifest.dataset_id,
            manifest_digest=self.manifest.manifest_digest(),
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            usable=not self._problems and bool(entries),
            problems=tuple(self._problems),
            sequence_count=len(sequences),
            file_count=len(entries),
            modalities=(Modality.RGB.value,),
            label_names=(),
            notes=("no labels, pose or calibration are produced by this adapter",),
        )

    def _require_available(self) -> None:
        if self._problems:
            raise DatasetNotAvailableError(
                self.manifest.dataset_id,
                self.manifest_path,
                [f"{p.path}" for p in self._problems if p.path]
                or ["a manifest that verifies with no problems"],
                self._problems,
            )

    def iter_sequences(self, split: str | None = None) -> Iterator[SequenceSample]:
        self._require_available()
        if split is not None and self.split_assignment is None:
            raise DatasetContractError("a split was requested but no SplitAssignment was supplied")
        grouped: dict[str, list[FileEntry]] = defaultdict(list)
        for entry in self._entries():
            sequence = self.manifest.resolve_lineage(entry).sequence
            if sequence is None:
                raise DatasetContractError(
                    f"{entry.path}: no sequence lineage; refusing frame-level handling"
                )
            assignment = self.split_assignment
            if split is not None and assignment is not None and assignment.split_of(entry.path) != split:
                continue
            grouped[sequence].append(entry)
        for sequence in sorted(grouped):
            entries = grouped[sequence]
            untimed = [e.path for e in entries if e.time_ns is None]
            if untimed:
                raise DatasetContractError(
                    f"sequence {sequence!r}: {len(untimed)} files have no time_ns (first: {untimed[0]}); "
                    "a sequence index is not a substitute for physical time"
                )
            entries.sort(key=lambda e: (e.time_ns or 0, e.sequence_index or 0, e.path))
            yield SequenceSample(
                dataset_id=self.manifest.dataset_id,
                manifest_digest=self.manifest.manifest_digest(),
                sequence_id=sequence,
                lineage=self.manifest.resolve_lineage(entries[0]),
                observations=tuple(self.normalize_observation(e) for e in entries),
                supervision=None,
                split=split,
            )

    def normalize_observation(self, raw: Any) -> Observation:
        if not isinstance(raw, FileEntry):
            raise DatasetContractError("ImageFolderSequenceAdapter normalizes manifest FileEntry rows only")
        clock = self.manifest.timestamps.clock_domain
        if clock == OPEN:
            raise DatasetContractError("manifest timestamps.clock_domain is OPEN")
        frame = self.manifest.frames.get(self.stream)
        if frame is None:
            raise DatasetContractError(f"manifest declares no coordinate frame for stream {self.stream!r}")
        if raw.time_ns is None:
            raise DatasetContractError(f"{raw.path}: time_ns missing")
        data = (self.data_root / raw.path).read_bytes()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise DatasetContractError(f"{raw.path}: bytes do not decode as an image")
        media = _MEDIA_TYPES.get(Path(raw.path).suffix.lower(), "application/octet-stream")
        ref = self.store.put_bytes(data, media, tuple(int(s) for s in image.shape), str(image.dtype))
        if ref.digest != raw.sha256:
            raise DatasetContractError(f"{raw.path}: stored digest differs from manifest sha256")
        lineage = self.manifest.resolve_lineage(raw)
        dataset = self.manifest.dataset_id
        calibration = None if self.manifest.calibration_ref == OPEN else self.manifest.calibration_ref
        return Observation(
            observation_id=self.ids.new(),
            mission_id=uuid5(NAMESPACE_URL, f"conrad-dataset:{dataset}:{lineage.sequence}"),
            run_id=self.run_id,
            trace_id=self.ids.new(),
            sensor_id=uuid5(NAMESPACE_URL, f"conrad-dataset:{dataset}:stream:{self.stream}"),
            modality=Modality.RGB,
            timestamp=TimeStamp(
                time_ns=raw.time_ns, clock_domain=clock, sequence_index=raw.sequence_index or 0
            ),
            sensor_frame=frame,
            robot_pose_estimate=None,
            payload_ref=ref,
            calibration_ref=calibration,
            sensor_context={
                "dataset_id": dataset,
                "manifest_digest": self.manifest.manifest_digest(),
                "source_path": raw.path,
                "source_sha256": raw.sha256,
                "lineage": lineage.declared(),
                "transformation_ids": list(raw.transformation_ids),
                "pose_available": False,
                "calibration_available": calibration is not None,
            },
        )

    def map_labels(self, raw_labels: Any) -> PartialTruth | None:
        """This adapter serves unlabelled sequences. Labels are never fabricated."""
        if raw_labels is not None:
            raise DatasetContractError("ImageFolderSequenceAdapter has no label schema; got raw labels")
        return None
