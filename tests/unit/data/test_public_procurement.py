from __future__ import annotations

from uuid import UUID

import cv2
import numpy as np
import pytest

from conrad.data.adapters import DatasetContractError, UvvidVideoAdapter
from conrad.data.adapters.public import PUBLIC_CANDIDATES, PUBLIC_MANIFEST_DIR
from conrad.data.inventory import build_file_entries
from conrad.data.manifest import (
    DatasetManifest,
    LineageKeys,
    ManifestError,
    ProcurementStatus,
    RightsReviewStatus,
    data_root_for,
    load_manifest,
    parse_manifest,
    verify_loaded_manifest,
    verify_manifest_by_id,
)
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality

MANIFESTS = {name: load_manifest(PUBLIC_MANIFEST_DIR / f) for name, f in PUBLIC_CANDIDATES.items()}


@pytest.mark.parametrize("name", sorted(PUBLIC_CANDIDATES))
def test_every_candidate_has_an_individual_audited_verdict(name):
    m = MANIFESTS[name]
    assert m.procurement_status is not ProcurementStatus.NOT_AUDITED
    assert m.verified_on == "2026-09-19"
    assert m.source_urls and m.conrad_tasks and m.forbidden_uses


@pytest.mark.parametrize("name", sorted(PUBLIC_CANDIDATES))
def test_unapproved_candidates_claim_no_rights_and_hold_no_files(name):
    m = MANIFESTS[name]
    if m.procurement_status.permits_download:
        assert m.license.startswith("CC-BY") and m.license_evidence_ref.startswith(
            ("Zenodo", "figshare", "4TU")
        )
        assert m.rights_review_status is RightsReviewStatus.CLEARED
    else:
        assert not m.files and not m.training_allowed
        assert m.rights_review_status is not RightsReviewStatus.CLEARED


def test_files_on_an_unapproved_public_manifest_are_rejected():
    data = MANIFESTS["uvvid"].model_dump(mode="json")
    data["procurement_status"] = ProcurementStatus.NEEDS_HUMAN_RIGHTS_REVIEW.value
    with pytest.raises(ManifestError, match="procurement_status"):
        parse_manifest(data)


def _write_video(path, n_frames: int, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened()
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8))
    writer.release()


def _manifest_for(tmp_path) -> DatasetManifest:
    (tmp_path / "raw").mkdir()
    _write_video(tmp_path / "raw" / "clip.avi", 12)
    files = build_file_entries(
        tmp_path,
        ["raw/clip.avi"],
        stream="rov_gopro_video",
        modality=Modality.RGB,
        lineage=LineageKeys(sequence="clip"),
    )
    data = MANIFESTS["uvvid"].model_dump(mode="json")
    data["files"] = [f.model_dump(mode="json") for f in files]
    data["checksum"] = DatasetManifest.model_validate({**data, "checksum": "x"}).files_digest()
    return parse_manifest(data)


def test_uvvid_adapter_fabricates_no_pose_calibration_or_labels(tmp_path):
    manifest = _manifest_for(tmp_path)
    assert verify_loaded_manifest(manifest, tmp_path) == []
    adapter = UvvidVideoAdapter(
        manifest, tmp_path, ObjectStore(tmp_path / "store"), IdFactory(seed=1), UUID(int=1), frame_stride=3
    )
    assert adapter.inspect().usable
    (sample,) = list(adapter.iter_sequences())
    assert sample.supervision is None and sample.sequence_id == "clip"
    obs = sample.observations
    assert [o.sensor_context["source_frame_index"] for o in obs] == [0, 3, 6, 9]
    times = [o.timestamp.time_ns for o in obs]
    assert times == sorted(times) and len(set(times)) == len(times)
    assert np.allclose(np.diff(times), 3 * 1e8, rtol=1e-3)  # 10 fps, stride 3 -> 0.3 s
    for o in obs:
        assert o.robot_pose_estimate is None and o.calibration_ref is None
        assert o.timestamp.clock_domain.endswith(":clip:container_pts")
        assert o.sensor_context["transformation_ids"] == ["uvvid-decode-v1"]
    assert adapter.map_labels(None) is None
    with pytest.raises(DatasetContractError):
        adapter.map_labels({"class": 1})
    with pytest.raises(DatasetContractError):
        list(adapter.iter_sequences(split="train"))


def test_uvvid_adapter_refuses_a_changed_file(tmp_path):
    manifest = _manifest_for(tmp_path)
    with (tmp_path / "raw" / "clip.avi").open("ab") as handle:
        handle.write(b"tamper")
    adapter = UvvidVideoAdapter(
        manifest, tmp_path, ObjectStore(tmp_path / "s"), IdFactory(seed=1), UUID(int=1)
    )
    assert not adapter.inspect().usable
    with pytest.raises(Exception, match="not available"):
        list(adapter.iter_sequences())


@pytest.mark.skipif(
    not (data_root_for("public.uvvid") / "raw" / "ROV_GoPro_4.mp4").exists(),
    reason="UVVID sample not downloaded (artifacts/data is gitignored)",
)
def test_downloaded_uvvid_sample_verifies():
    assert verify_manifest_by_id("public.uvvid") == []
