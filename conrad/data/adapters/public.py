"""Public-dataset candidates (ch24). Per-candidate verdicts: docs/audits/PUBLIC_DATA_PROCUREMENT.md.

Only ``public.uvvid`` has a verified local sample; it is read by :class:`UvvidVideoAdapter`, not here.

Each candidate has a manifest template under ``datasets/public/``. The adapter is deliberately
not a silent stub: every data-access method raises :class:`DatasetNotAvailableError` naming the
manifest and the exact artefacts that a DATA-VERIFY task must supply first.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from conrad.data.adapters.base import (
    DatasetAdapter,
    DatasetAudit,
    DatasetNotAvailableError,
    PartialTruth,
    SequenceSample,
)
from conrad.data.manifest import OPEN, DatasetManifest, load_manifest, verify_loaded_manifest
from conrad.schemas.observation import Observation
from conrad.settings import REPO_ROOT

PUBLIC_MANIFEST_DIR = REPO_ROOT / "datasets" / "public"

# candidate id -> manifest template file name. Names come from spec ch24 only.
PUBLIC_CANDIDATES: dict[str, str] = {
    "subpipe": "subpipe.manifest.yaml",
    "aqualoc": "aqualoc.manifest.yaml",
    "benthicnet": "benthicnet.manifest.yaml",
    "fathomnet": "fathomnet.manifest.yaml",
    "uvvid": "uvvid.manifest.yaml",
    "underwater_caves_sonar": "underwater_caves_sonar.manifest.yaml",
    "uw_slam": "uw_slam.manifest.yaml",
    "seaview": "seaview.manifest.yaml",
    "suim": "suim.manifest.yaml",
    "seaclear": "seaclear.manifest.yaml",
}


class PublicDatasetAdapter(DatasetAdapter):
    adapter_name = "public_dataset_unverified"
    adapter_version = "0.0.0-unimplemented"

    def __init__(self, manifest_path: str | Path, data_root: str | Path | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        super().__init__(load_manifest(self.manifest_path))
        self.data_root = None if data_root is None else Path(data_root)

    def required_items(self) -> list[str]:
        m: DatasetManifest = self.manifest
        required: list[str] = []
        named = {
            "source URL (source)": m.source,
            "pinned release (release_id)": m.release_id,
            "license text and evidence (license, license_evidence_ref)": m.license_evidence_ref,
            "asset-level rights (rights_by_asset_ref)": m.rights_by_asset_ref,
            "calibration (calibration_ref)": m.calibration_ref,
            "label schema audit (label_schema_ref)": m.label_schema_ref,
            "split manifest (split_manifest_ref)": m.split_manifest_ref,
            "duplicate audit (duplicate_audit_ref)": m.duplicate_audit_ref,
        }
        required.extend(f"{label} is {value}" for label, value in named.items() if value == OPEN)
        if not m.files:
            required.append(
                "asset inventory: `files:` is empty; every data file needs path, sha256, byte_length, "
                "stream, modality and lineage keys"
            )
        else:
            required.extend(f"file {f.path} (sha256 {f.sha256})" for f in m.files[:20])
        if self.data_root is None:
            required.append("a local data_root containing the files (nothing is downloaded by Conrad)")
        required.append("a dataset-specific decoder reviewed under DATA-ADAPTER-01 (none exists yet)")
        if m.required_files_note:
            required.append(m.required_files_note)
        return required

    def _problems(self) -> list[Any]:
        root = (
            self.data_root if self.data_root is not None else self.manifest_path.parent / "__no_data_root__"
        )
        return verify_loaded_manifest(self.manifest, root)

    def _unavailable(self) -> DatasetNotAvailableError:
        return DatasetNotAvailableError(
            self.manifest.dataset_id, self.manifest_path, self.required_items(), self._problems()
        )

    def inspect(self) -> DatasetAudit:
        return DatasetAudit(
            dataset_id=self.manifest.dataset_id,
            manifest_digest=self.manifest.manifest_digest(),
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            usable=False,
            problems=tuple(self._problems()),
            sequence_count=None,
            file_count=len(self.manifest.files),
            notes=tuple(self.required_items()),
        )

    def iter_sequences(self, split: str | None = None) -> Iterator[SequenceSample]:
        raise self._unavailable()

    def normalize_observation(self, raw: Any) -> Observation:
        raise self._unavailable()

    def map_labels(self, raw_labels: Any) -> PartialTruth | None:
        raise self._unavailable()


def public_adapter(candidate: str, data_root: str | Path | None = None) -> PublicDatasetAdapter:
    if candidate not in PUBLIC_CANDIDATES:
        raise KeyError(f"unknown public candidate {candidate!r}; known: {sorted(PUBLIC_CANDIDATES)}")
    return PublicDatasetAdapter(PUBLIC_MANIFEST_DIR / PUBLIC_CANDIDATES[candidate], data_root)
