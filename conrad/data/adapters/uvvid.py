"""UVVID visual-subset adapter (DTU Data, CC BY 4.0, doi:10.11583/DTU.27694068.v5).

Converts the verified ``visual/ROV_GoPro_*.mp4`` files listed in ``datasets/public/uvvid.manifest.yaml``
into RGB :class:`Observation` objects. What the source does not provide stays ``None``:

* the dataset README says the visual subset has no calibration and no IMU, so ``calibration_ref``
  and ``robot_pose_estimate`` are always ``None``;
* there are no labels, so :meth:`map_labels` never returns supervision;
* time is the container presentation timestamp (ms, converted to integer ns) relative to the first
  frame of each video. There is no absolute clock, so each video gets its own clock domain and
  nothing is synchronized across videos.

Decoding is the declared transformation ``uvvid-decode-v1`` (H.264 -> uint8 BGR array, frame stride
from the constructor); the decoded array is stored unchanged in the object store.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.timebase import TimeStamp

DECODE_TRANSFORMATION_ID = "uvvid-decode-v1"
VIDEO_STREAM = "rov_gopro_video"


@dataclass(frozen=True)
class DecodedFrame:
    """One decoded frame plus where it came from. ``pts_ns`` is container time, not wall time."""

    entry: FileEntry
    frame_index: int
    pts_ns: int
    image: np.ndarray


class UvvidVideoAdapter(DatasetAdapter):
    adapter_name = "uvvid_video"
    adapter_version = "uvvid_video-1.0.0"

    def __init__(
        self,
        manifest: DatasetManifest,
        data_root: str | Path,
        store: ObjectStore,
        ids: IdFactory,
        run_id: UUID,
        *,
        frame_stride: int = 1,
        max_frames_per_sequence: int | None = None,
        manifest_path: str | Path = "<in-memory>",
    ) -> None:
        super().__init__(manifest)
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if manifest.adapter_version != self.adapter_version:
            raise DatasetContractError(
                f"manifest pins adapter_version {manifest.adapter_version!r}, this adapter is "
                f"{self.adapter_version!r}"
            )
        declared = {t.transformation_id for t in manifest.transformations}
        if DECODE_TRANSFORMATION_ID not in declared:
            raise DatasetContractError(f"manifest does not declare transformation {DECODE_TRANSFORMATION_ID}")
        self.data_root = Path(data_root)
        self.store = store
        self.ids = ids
        self.run_id = run_id
        self.frame_stride = frame_stride
        self.max_frames = max_frames_per_sequence
        self.manifest_path = str(manifest_path)
        self._problems = verify_loaded_manifest(manifest, self.data_root)

    def _entries(self) -> list[FileEntry]:
        return sorted(
            (f for f in self.manifest.files if f.stream == VIDEO_STREAM and f.modality is Modality.RGB),
            key=lambda f: f.path,
        )

    def inspect(self) -> DatasetAudit:
        entries = self._entries()
        return DatasetAudit(
            dataset_id=self.manifest.dataset_id,
            manifest_digest=self.manifest.manifest_digest(),
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            usable=not self._problems and bool(entries),
            problems=tuple(self._problems),
            sequence_count=len({self.manifest.resolve_lineage(f).sequence for f in entries}),
            file_count=len(entries),
            modalities=(Modality.RGB.value,),
            label_names=(),
            notes=(
                "visual subset: no calibration, no IMU, no pose, no labels (dataset README)",
                "timestamps are per-video container presentation times; videos are not synchronized",
            ),
        )

    def _require_available(self) -> None:
        if self._problems:
            raise DatasetNotAvailableError(
                self.manifest.dataset_id,
                self.manifest_path,
                [p.path for p in self._problems if p.path] or ["a manifest that verifies with no problems"],
                self._problems,
            )

    def clock_domain(self, entry: FileEntry) -> str:
        return f"{self.manifest.dataset_id}:{PurePosixPath(entry.path).stem}:container_pts"

    def decode(self, entry: FileEntry) -> Iterator[DecodedFrame]:
        """Yield every ``frame_stride``-th frame with its container timestamp. Fails on unordered time."""
        capture = cv2.VideoCapture(str(self.data_root / entry.path))
        if not capture.isOpened():
            raise DatasetContractError(f"{entry.path}: video cannot be opened by the OpenCV backend")
        try:
            index = kept = 0
            last_ns = -1
            while self.max_frames is None or kept < self.max_frames:
                ok, image = capture.read()
                if not ok:
                    break
                pts_ns = round(float(capture.get(cv2.CAP_PROP_POS_MSEC)) * 1_000_000)
                if pts_ns < 0 or pts_ns <= last_ns:
                    raise DatasetContractError(
                        f"{entry.path}: frame {index} container time {pts_ns} ns is not increasing"
                    )
                last_ns = pts_ns
                if index % self.frame_stride == 0:
                    kept += 1
                    yield DecodedFrame(entry=entry, frame_index=index, pts_ns=pts_ns, image=image)
                index += 1
        finally:
            capture.release()
        if index == 0:
            raise DatasetContractError(f"{entry.path}: no frame decoded")

    def iter_sequences(self, split: str | None = None) -> Iterator[SequenceSample]:
        self._require_available()
        if split is not None:
            raise DatasetContractError("no split manifest exists for the UVVID sample; request split=None")
        for entry in self._entries():
            lineage = self.manifest.resolve_lineage(entry)
            if lineage.sequence is None:
                raise DatasetContractError(f"{entry.path}: no sequence lineage")
            yield SequenceSample(
                dataset_id=self.manifest.dataset_id,
                manifest_digest=self.manifest.manifest_digest(),
                sequence_id=lineage.sequence,
                lineage=lineage,
                observations=tuple(self.normalize_observation(f) for f in self.decode(entry)),
                supervision=None,
                split=None,
            )

    def normalize_observation(self, raw: Any) -> Observation:
        if not isinstance(raw, DecodedFrame):
            raise DatasetContractError("UvvidVideoAdapter normalizes DecodedFrame objects only")
        frame = self.manifest.frames.get(VIDEO_STREAM)
        if frame is None:
            raise DatasetContractError(f"manifest declares no coordinate frame for {VIDEO_STREAM!r}")
        entry = raw.entry
        dataset = self.manifest.dataset_id
        lineage = self.manifest.resolve_lineage(entry)
        ref = self.store.put_array(np.ascontiguousarray(raw.image))
        calibration = (
            None if self.manifest.calibration_ref in (OPEN, "NONE") else self.manifest.calibration_ref
        )
        return Observation(
            observation_id=self.ids.new(),
            mission_id=uuid5(NAMESPACE_URL, f"conrad-dataset:{dataset}:{lineage.sequence}"),
            run_id=self.run_id,
            trace_id=self.ids.new(),
            sensor_id=uuid5(NAMESPACE_URL, f"conrad-dataset:{dataset}:stream:{VIDEO_STREAM}:{entry.path}"),
            modality=Modality.RGB,
            timestamp=TimeStamp(
                time_ns=raw.pts_ns, clock_domain=self.clock_domain(entry), sequence_index=raw.frame_index
            ),
            sensor_frame=frame,
            robot_pose_estimate=None,
            payload_ref=ref,
            calibration_ref=calibration,
            sensor_context={
                "dataset_id": dataset,
                "manifest_digest": self.manifest.manifest_digest(),
                "source_path": entry.path,
                "source_sha256": entry.sha256,
                "source_frame_index": raw.frame_index,
                "lineage": lineage.declared(),
                "transformation_ids": [DECODE_TRANSFORMATION_ID],
                "channel_order": "BGR",
                "license": self.manifest.license,
                "pose_available": False,
                "calibration_available": calibration is not None,
                "labels_available": False,
            },
        )

    def map_labels(self, raw_labels: Any) -> PartialTruth | None:
        """UVVID has no labels. Nothing is ever fabricated."""
        if raw_labels is not None:
            raise DatasetContractError("UVVID visual subset has no label schema; got raw labels")
        return None
