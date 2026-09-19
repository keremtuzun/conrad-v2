"""SubPipeAdapter on a tiny archive with SubPipeMini2's layout.

The pixels are synthetic. ``CONFIG`` repeats the cam0/cam1 entries of SubPipe's config.yaml
(CC BY 4.0, Alvarez-Tunon et al. 2024, doi:10.5281/zenodo.12666132) so that the calibration parsing
is tested on the real file format.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from uuid import UUID

import cv2
import numpy as np
import pytest

from conrad.data.adapters import DatasetContractError, SubPipeAdapter
from conrad.data.adapters.subpipe import CLOCK_DOMAIN, stamp_to_ns
from conrad.data.manifest import (
    FileEntry,
    LineageKeys,
    data_root_for,
    files_digest,
    parse_manifest,
    verify_manifest_by_id,
)
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality

CONFIG = """cam0:
  - name: "GoPro Camera"
  - resolution:
    - width: 2704
    - height: 1520
    - channels: 3
  - frame_rate: 30
  - offset_x: 619.0
  - offset_z: -120.0
  - offset_theta: -56.0
  - fx: 1612.36
  - fy: 1622.56
  - cx: 1365.43
  - cy:  741.27
  - k1_k2_p1_p2: [-0.247, 0.0869, -0.006, 0.001]
cam1:
  - name: "Gray-scale Camera"
  - offset_x: -813.0
"""
LF = ["1693569219.799", "1693569220.800", "1693569221.801"]


def _ppm(h: int, w: int, seed: int) -> bytes:
    px = np.random.default_rng(seed).integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return f"P6\n{w} {h}\n255\n".encode() + px[..., ::-1].tobytes()


def _archive(tmp_path) -> tuple[str, FileEntry]:
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "SubPipeMini2.zip"
    coco = {
        "images": [{"file_name": f"{LF[0]}.pbm", "id": 1}, {"file_name": f"{LF[1]}.bpm", "id": 2}],
        "annotations": [{"id": 1, "image_id": 2, "category_id": 1, "bbox": [1, 1, 3, 3]}],
        "categories": [{"id": 1, "name": "Pipeline"}],
    }
    ok, jpg = cv2.imencode(".jpg", np.full((8, 12, 3), 100, dtype=np.uint8))
    assert ok
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("SubPipeMiniSSS/config.yaml", CONFIG)
        z.writestr(f"SubPipeMiniSSS/DATA/SSS_LF_images/Image/{LF[0]}.pbm", _ppm(6, 10, 0))
        z.writestr(f"SubPipeMiniSSS/DATA/SSS_LF_images/Image/{LF[1]}.bpm", _ppm(6, 10, 1))
        z.writestr(f"SubPipeMiniSSS/DATA/SSS_LF_images/Image/{LF[2]}.pbm", _ppm(6, 10, 2))
        z.writestr("SubPipeMiniSSS/DATA/SSS_LF_images/COCO_Annotation/coco_format.json", json.dumps(coco))
        z.writestr("SubPipeMiniSSS/DATA/Cam0_images/1693572852.904.jpg", jpg.tobytes())
    data = path.read_bytes()
    entry = FileEntry(
        path="raw/SubPipeMini2.zip",
        sha256=hashlib.sha256(data).hexdigest(),
        byte_length=len(data),
        stream="archive",
        modality=Modality.STRUCTURED,
        lineage=LineageKeys(sequence="subpipe/mini_sss"),
    )
    return str(tmp_path), entry


def _manifest(entry: FileEntry):
    return parse_manifest(
        {
            "dataset_id": "public.subpipe",
            "version": "test",
            "status": "APPROVED",
            "procurement_status": "APPROVED",
            "source_category": "PUBLIC_REAL",
            "license": "CC-BY-4.0",
            "license_evidence_ref": "test",
            "usage_rights": "CC-BY-4.0",
            "rights_review_status": "CLEARED",
            "training_allowed": True,
            "modalities": ["STRUCTURED"],
            "split_strategy": ["sequence"],
            "adapter_version": "subpipe_zip-1.0.0",
            "transformations": [
                {
                    "transformation_id": "subpipe-zip-member-decode-v1",
                    "kind": "TRANSFORM",
                    "description": "decode",
                    "version": "1",
                    "config_hash": "x",
                }
            ],
            "files": [entry.model_dump(mode="json")],
            "checksum": files_digest([entry]),
        }
    )


def _adapter(tmp_path, **kw):
    root, entry = _archive(tmp_path)
    return SubPipeAdapter(
        _manifest(entry), root, ObjectStore(tmp_path / "store"), IdFactory(seed=3), UUID(int=3), **kw
    )


def test_stamp_to_ns_is_exact():
    assert stamp_to_ns("1693572852.904") == 1_693_572_852_904_000_000
    with pytest.raises(DatasetContractError):
        stamp_to_ns("d.860")


def test_sonar_observations_carry_no_pose_no_calibration_and_no_labels(tmp_path):
    adapter = _adapter(tmp_path)
    assert adapter.inspect().usable
    (sample,) = list(adapter.iter_sequences())
    obs = sample.observations
    assert [o.timestamp.time_ns for o in obs] == [stamp_to_ns(s) for s in LF]
    for o in obs:
        assert o.modality is Modality.SONAR and o.timestamp.clock_domain == CLOCK_DOMAIN
        assert o.robot_pose_estimate is None and o.calibration_ref is None
        assert o.sensor_frame == "subpipe.sss_lf"
        assert not {"pipeline_present", "pipeline_box_count", "bbox"} & set(o.sensor_context)
        assert adapter.store.get_array(o.payload_ref).shape == (6, 10, 3)
    truth = sample.supervision
    assert truth is not None
    assert truth.availability_masks["pipeline_present"].tolist() == [True, True, False]
    assert truth.targets["pipeline_present"][:2].tolist() == [0.0, 1.0]
    assert np.isnan(truth.targets["pipeline_present"][2])


def test_cam0_calibration_comes_from_config_only(tmp_path):
    adapter = _adapter(tmp_path, streams=("cam0",))
    cal = adapter.calibration("cam0")
    assert cal is not None and cal["fx"] == 1612.36 and cal["distortion"] == [-0.247, 0.0869, -0.006, 0.001]
    assert adapter.calibration("cam1") is None and adapter.calibration("sss_lf") is None
    (sample,) = list(adapter.iter_sequences())
    assert sample.supervision is None
    assert sample.observations[0].calibration_ref == "public.subpipe:config.yaml#cam0"


def test_a_changed_archive_fails_closed(tmp_path):
    root, entry = _archive(tmp_path)
    with (tmp_path / "raw" / "SubPipeMini2.zip").open("ab") as handle:
        handle.write(b"x")
    adapter = SubPipeAdapter(
        _manifest(entry), root, ObjectStore(tmp_path / "s"), IdFactory(seed=1), UUID(int=1)
    )
    assert not adapter.inspect().usable
    with pytest.raises(Exception, match="not available"):
        list(adapter.iter_sequences())


def test_split_requests_are_refused(tmp_path):
    with pytest.raises(DatasetContractError):
        list(_adapter(tmp_path).iter_sequences(split="test"))


@pytest.mark.skipif(
    not (data_root_for("public.subpipe") / "raw" / "SubPipeMini2.zip").exists(),
    reason="SubPipeMini2.zip (4.9 GB) not downloaded; artifacts/data is gitignored",
)
def test_downloaded_subpipe_archive_verifies():
    assert verify_manifest_by_id("public.subpipe") == []
