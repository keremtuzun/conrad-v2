"""SubPipe (SubPipeMini2.zip, "SubPipeMiniSSS") adapter. Zenodo doi:10.5281/zenodo.12666132 v3.0.1, CC BY 4.0.

Attribution: Alvarez-Tunon, O., Marnet, L. R., Antal, L., Aubard, M., Costa, M., Brodskiy, Y. (2024),
SubPipe: A Submarine Pipeline Inspection Dataset for Segmentation and Visual-inertial Localization.
Data property of Oceanscan-MST, acquired within the H2020 REMARO project.

Members are read straight from the verified archive (``zipfile`` checks each member's CRC-32), so the
~15.6 GB of uncompressed PPM/JPEG is never extracted. What the adapter does and does not provide:

* streams ``cam0`` (GoPro, RGB), ``cam1`` (grey camera), ``sss_lf`` / ``sss_hf`` (Klein 3500 side-scan,
  455 / 900 kHz). The sonar files are binary PPM (``P6``) rendered with a colormap, despite the ``.pbm``
  / ``.bpm`` names; they are not raw backscatter.
* time: the file-name stamp is UNIX epoch seconds with millisecond resolution; it becomes integer ns in
  clock domain ``subpipe:vehicle_unix`` with a 1 ms time uncertainty.
* calibration: ``config.yaml`` gives cam0 intrinsics + distortion and a mount offset (x, z, pitch) for
  cam0/cam1. ``calibration_ref`` is set only for cam0; cam1 and the sonars have no intrinsics.
* pose: ``robot_pose_estimate`` is ``None``. ``EstimatedState.csv`` is the vehicle's own navigation
  solution with undocumented frame conventions, and it is not ground truth.
* labels: the COCO "Pipeline" boxes of the sonar streams are returned only through :meth:`map_labels`
  (the evaluation-side :class:`PartialTruth` channel). No label value enters ``sensor_context``.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import cv2
import numpy as np
import yaml

from conrad.data.adapters.base import (
    DatasetAdapter,
    DatasetAudit,
    DatasetContractError,
    DatasetNotAvailableError,
    PartialTruth,
    SequenceSample,
)
from conrad.data.manifest import DatasetManifest, FileEntry, verify_loaded_manifest
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.timebase import TimeStamp

ARCHIVE_STREAM = "archive"
READ_TRANSFORMATION_ID = "subpipe-zip-member-decode-v1"
CLOCK_DOMAIN = "subpipe:vehicle_unix"
TIME_UNCERTAINTY_NS = 1_000_000  # file names carry milliseconds


@dataclass(frozen=True)
class StreamSpec:
    folder: str
    modality: Modality
    frame_id: str
    suffixes: tuple[str, ...]
    coco: str | None = None


STREAMS: dict[str, StreamSpec] = {
    "cam0": StreamSpec("DATA/Cam0_images", Modality.RGB, "subpipe.cam0_optical", (".jpg",)),
    "cam1": StreamSpec("DATA/Cam1_images", Modality.RGB, "subpipe.cam1_optical", (".jpg",)),
    "sss_lf": StreamSpec(
        "DATA/SSS_LF_images/Image",
        Modality.SONAR,
        "subpipe.sss_lf",
        (".pbm", ".bpm"),
        "DATA/SSS_LF_images/COCO_Annotation/coco_format.json",
    ),
    "sss_hf": StreamSpec(
        "DATA/SSS_HF_images/Image",
        Modality.SONAR,
        "subpipe.sss_hf",
        (".pbm", ".bpm"),
        "DATA/SSS_HF_images/COCO_Annotation/coco_format.json",
    ),
}
_STAMP = re.compile(r"^(\d+)\.(\d+)$")


def stamp_to_ns(stem: str) -> int:
    """``"1693572852.904"`` -> 1693572852904000000 (exact decimal arithmetic, no float rounding)."""
    if not _STAMP.match(stem):
        raise DatasetContractError(f"file name {stem!r} is not an epoch-seconds stamp")
    return int(Decimal(stem) * 1_000_000_000)


@dataclass(frozen=True)
class FrameRef:
    stream: str
    member: str
    time_ns: int

    @property
    def name(self) -> str:
        return PurePosixPath(self.member).name


def _flatten(items: Any) -> dict[str, Any]:
    """config.yaml stores each sensor as a list of one-key dicts; merge them (nested lists too)."""
    out: dict[str, Any] = {}
    if isinstance(items, dict):
        items = [items]
    for item in items or []:
        if isinstance(item, dict):
            for k, v in item.items():
                out[str(k)] = _flatten(v) if isinstance(v, list) and v and isinstance(v[0], dict) else v
    return out


class SubPipeAdapter(DatasetAdapter):
    adapter_name = "subpipe_zip"
    adapter_version = "subpipe_zip-1.0.0"

    def __init__(
        self,
        manifest: DatasetManifest,
        data_root: str | Path,
        store: ObjectStore,
        ids: IdFactory,
        run_id: UUID,
        *,
        streams: Sequence[str] = ("sss_lf",),
        frame_stride: int = 1,
        manifest_path: str | Path = "<in-memory>",
    ) -> None:
        super().__init__(manifest)
        unknown = sorted(set(streams) - set(STREAMS))
        if unknown:
            raise ValueError(f"unknown SubPipe streams {unknown}; known: {sorted(STREAMS)}")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if manifest.adapter_version != self.adapter_version:
            raise DatasetContractError(
                f"manifest pins adapter_version {manifest.adapter_version!r}, this adapter is "
                f"{self.adapter_version!r}"
            )
        if READ_TRANSFORMATION_ID not in {t.transformation_id for t in manifest.transformations}:
            raise DatasetContractError(f"manifest does not declare transformation {READ_TRANSFORMATION_ID}")
        archives = [f for f in manifest.files if f.stream == ARCHIVE_STREAM]
        if len(archives) != 1:
            raise DatasetContractError("the SubPipe manifest must list exactly one archive file")
        self.archive: FileEntry = archives[0]
        self.data_root = Path(data_root)
        self.store, self.ids, self.run_id = store, ids, run_id
        self.streams = tuple(streams)
        self.frame_stride = frame_stride
        self.manifest_path = str(manifest_path)
        self._problems = verify_loaded_manifest(manifest, self.data_root)
        self._zip: zipfile.ZipFile | None = None
        self._root: str | None = None

    # ------------------------------------------------------------------ archive access
    def _require_available(self) -> None:
        if self._problems:
            raise DatasetNotAvailableError(
                self.manifest.dataset_id,
                self.manifest_path,
                [p.path for p in self._problems if p.path] or ["a manifest that verifies with no problems"],
                self._problems,
            )

    def _archive(self) -> zipfile.ZipFile:
        self._require_available()
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.data_root / self.archive.path)
            roots = {n.split("/", 1)[0] for n in self._zip.namelist() if n.endswith("config.yaml")}
            if len(roots) != 1:
                raise DatasetContractError(f"expected one <root>/config.yaml in the archive, found {roots}")
            self._root = roots.pop()
        return self._zip

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def _member(self, rel: str) -> str:
        self._archive()
        return f"{self._root}/{rel}"

    def read_member(self, rel: str) -> bytes:
        return self._archive().read(self._member(rel))

    def config(self) -> dict[str, dict[str, Any]]:
        raw = yaml.safe_load(self.read_member("config.yaml").decode("utf-8", "replace")) or {}
        return {str(k): _flatten(v) for k, v in raw.items()}

    def calibration(self, stream: str) -> dict[str, Any] | None:
        """Intrinsics exactly as config.yaml states them; ``None`` when the file gives none (cam1, sonar)."""
        if stream != "cam0":
            return None
        cam = self.config().get("cam0", {})
        needed = ("fx", "fy", "cx", "cy", "k1_k2_p1_p2")
        if not all(k in cam for k in needed):
            return None
        res = cam.get("resolution", {})
        return {
            "model": "pinhole+radtan (k1, k2, p1, p2)",
            "fx": float(cam["fx"]),
            "fy": float(cam["fy"]),
            "cx": float(cam["cx"]),
            "cy": float(cam["cy"]),
            "distortion": [float(v) for v in cam["k1_k2_p1_p2"]],
            "width_px": int(res["width"]) if "width" in res else None,
            "height_px": int(res["height"]) if "height" in res else None,
            "source": "SubPipe config.yaml cam0",
        }

    def calibration_ref(self, stream: str) -> str | None:
        return (
            None if self.calibration(stream) is None else f"{self.manifest.dataset_id}:config.yaml#{stream}"
        )

    def frames(self, stream: str) -> list[FrameRef]:
        spec = STREAMS[stream]
        prefix = self._member(spec.folder) + "/"
        out = []
        for name in self._archive().namelist():
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            leaf = PurePosixPath(name)
            if "/" in name[len(prefix) :] or leaf.suffix.lower() not in spec.suffixes:
                continue
            out.append(FrameRef(stream, name, stamp_to_ns(leaf.stem)))
        return sorted(out, key=lambda r: (r.time_ns, r.member))

    def load(self, ref: FrameRef) -> np.ndarray:
        data = self._archive().read(ref.member)  # CRC-32 checked by zipfile
        flag = cv2.IMREAD_GRAYSCALE if ref.stream == "cam1" else cv2.IMREAD_COLOR
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flag)
        if image is None:
            raise DatasetContractError(f"{ref.member}: bytes do not decode as an image")
        return np.asarray(image)

    # ------------------------------------------------------------------ labels (evaluation side only)
    def coco(self, stream: str) -> dict[str, Any]:
        spec = STREAMS[stream]
        if spec.coco is None:
            raise DatasetContractError(f"stream {stream!r} has no annotations in this archive")
        data: dict[str, Any] = json.loads(self.read_member(spec.coco))
        return data

    def map_labels(self, raw_labels: Any) -> PartialTruth | None:
        """``(stream, [FrameRef, ...])`` -> pipeline presence + box count, masked by COCO coverage."""
        if raw_labels is None:
            return None
        stream, refs = raw_labels
        coco = self.coco(stream)
        ids = {img["file_name"]: img["id"] for img in coco["images"]}
        counts: dict[int, int] = {}
        for ann in coco["annotations"]:
            counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1
        listed = np.asarray([r.name in ids for r in refs], dtype=np.bool_)
        n_boxes = np.asarray(
            [counts.get(ids[r.name], 0) if r.name in ids else np.nan for r in refs], dtype=np.float64
        )
        present = np.where(listed, (np.nan_to_num(n_boxes) > 0).astype(np.float64), np.nan)
        src = f"{self.manifest.dataset_id}:{STREAMS[stream].coco}"
        return PartialTruth(
            targets={"pipeline_present": present, "pipeline_box_count": n_boxes},
            availability_masks={"pipeline_present": listed, "pipeline_box_count": listed.copy()},
            label_source_refs={"pipeline_present": src, "pipeline_box_count": src},
            quality_metadata={
                "label_kind": "human bounding boxes, single class 'Pipeline'",
                "negative_rule": "an image listed in COCO with zero boxes is a negative",
                "unlisted_images": int((~listed).sum()),
            },
        )

    # ------------------------------------------------------------------ observations
    def normalize_observation(self, raw: Any) -> Observation:
        if not isinstance(raw, FrameRef):
            raise DatasetContractError("SubPipeAdapter normalizes FrameRef objects only")
        spec = STREAMS[raw.stream]
        dataset = self.manifest.dataset_id
        ref = self.store.put_array(np.ascontiguousarray(self.load(raw)))
        cal = self.calibration_ref(raw.stream)
        sequence = self.manifest.resolve_lineage(self.archive).sequence
        return Observation(
            observation_id=self.ids.new(),
            mission_id=uuid5(NAMESPACE_URL, f"conrad-dataset:{dataset}:{sequence}"),
            run_id=self.run_id,
            trace_id=self.ids.new(),
            sensor_id=uuid5(NAMESPACE_URL, f"conrad-dataset:{dataset}:stream:{raw.stream}"),
            modality=spec.modality,
            timestamp=TimeStamp(
                time_ns=raw.time_ns, clock_domain=CLOCK_DOMAIN, time_uncertainty_ns=TIME_UNCERTAINTY_NS
            ),
            sensor_frame=spec.frame_id,
            robot_pose_estimate=None,
            payload_ref=ref,
            calibration_ref=cal,
            sensor_context={
                "dataset_id": dataset,
                "manifest_digest": self.manifest.manifest_digest(),
                "archive_sha256": self.archive.sha256,
                "archive_member": raw.member,
                "stream": raw.stream,
                "transformation_ids": [READ_TRANSFORMATION_ID],
                "channel_order": "GRAY" if raw.stream == "cam1" else "BGR",
                "sonar_rendering": "colormapped PPM, not raw backscatter"
                if spec.modality is Modality.SONAR
                else None,
                "license": self.manifest.license,
                "pose_available": False,
                "calibration_available": cal is not None,
            },
        )

    def iter_sequences(self, split: str | None = None) -> Iterator[SequenceSample]:
        """One sample per stream (the archive holds one mission). Supervision only for sonar streams."""
        if split is not None:
            raise DatasetContractError("splits are built by the experiment from time blocks; pass split=None")
        lineage = self.manifest.resolve_lineage(self.archive)
        for stream in self.streams:
            refs = self.frames(stream)[:: self.frame_stride]
            yield SequenceSample(
                dataset_id=self.manifest.dataset_id,
                manifest_digest=self.manifest.manifest_digest(),
                sequence_id=f"{lineage.sequence}/{stream}",
                lineage=lineage,
                observations=tuple(self.normalize_observation(r) for r in refs),
                supervision=self.map_labels((stream, refs)) if STREAMS[stream].coco else None,
                split=None,
            )

    def inspect(self) -> DatasetAudit:
        notes = ["pose is None (EstimatedState frame undocumented, not ground truth)"]
        counts: list[str] = []
        if not self._problems:
            for s in self.streams:
                counts.append(f"{s}: {len(self.frames(s))} frames")
        return DatasetAudit(
            dataset_id=self.manifest.dataset_id,
            manifest_digest=self.manifest.manifest_digest(),
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            usable=not self._problems,
            problems=tuple(self._problems),
            sequence_count=len(self.streams),
            file_count=len(self.manifest.files),
            modalities=tuple(sorted({STREAMS[s].modality.value for s in self.streams})),
            label_names=("pipeline_present", "pipeline_box_count"),
            notes=tuple(notes + counts),
        )
