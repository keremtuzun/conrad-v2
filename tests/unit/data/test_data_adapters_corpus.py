from __future__ import annotations

import hashlib
from dataclasses import dataclass
from uuid import UUID

import cv2
import numpy as np
import pytest

from conrad.data.adapters import DatasetContractError, ImageFolderSequenceAdapter, SyntheticTwinAdapter
from conrad.data.adapters.base import PartialTruth
from conrad.data.inventory import build_file_entries
from conrad.data.manifest import (
    DatasetManifest,
    LineageKeys,
    ManifestStatus,
    RightsReviewStatus,
    SourceCategory,
    SplitUnit,
    TimestampSemantics,
    files_digest,
)
from conrad.data.ssl_corpus import (
    CaptureUnit,
    Partition,
    PartitionStatus,
    build_corpus_acceptance_report,
    read_capture_units,
    render_report_markdown,
    write_capture_units,
)
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.truth import SupervisionLabel
from conrad.schemas.world import Domain


def image_manifest(root):
    rels = []
    for i in range(3):
        ok, png = cv2.imencode(".png", np.full((4, 5, 3), i * 40, dtype=np.uint8))
        assert ok
        rel = f"dive1/img{i}.png"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(png.tobytes())
        rels.append(rel)
    files = tuple(
        f.model_copy(update={"time_ns": 1_000 * (i + 1)})
        for i, f in enumerate(
            build_file_entries(
                root,
                rels,
                stream="rgb",
                modality=Modality.RGB,
                lineage=LineageKeys(sequence="dive1", dive="dive1"),
            )
        )
    )
    return DatasetManifest(
        dataset_id="conrad.rgb.test",
        version="1",
        status=ManifestStatus.VERIFIED,
        source_category=SourceCategory.CONRAD_REAL,
        license="internal",
        license_evidence_ref="test",
        usage_rights="internal",
        rights_review_status=RightsReviewStatus.CLEARED,
        training_allowed=True,
        modalities=(Modality.RGB,),
        frames={"rgb": "camera_front"},
        split_strategy=(SplitUnit.SEQUENCE,),
        timestamps=TimestampSemantics(clock_domain="capture_utc", units="ns", semantics="exposure start"),
        adapter_version="1.0.0",
        checksum=files_digest(files),
        files=files,
    )


def test_image_folder_adapter_keeps_bytes_and_fabricates_nothing(tmp_path):
    m = image_manifest(tmp_path / "data")
    adapter = ImageFolderSequenceAdapter(
        m, tmp_path / "data", ObjectStore(tmp_path / "obj"), IdFactory(seed=1), UUID(int=1)
    )
    assert adapter.inspect().usable
    (seq,) = list(adapter.iter_sequences())
    assert seq.supervision is None and len(seq.observations) == 3
    for obs, entry in zip(seq.observations, m.files, strict=True):
        assert obs.robot_pose_estimate is None and obs.calibration_ref is None
        assert obs.payload_ref is not None and obs.payload_ref.digest == entry.sha256
        assert obs.sensor_frame == "camera_front"
    assert adapter.map_labels(None) is None
    with pytest.raises(DatasetContractError):
        adapter.map_labels({"class": 1})


def test_image_folder_adapter_is_deterministic(tmp_path):
    m = image_manifest(tmp_path / "data")
    runs = [
        [
            o.observation_id
            for o in next(
                iter(
                    ImageFolderSequenceAdapter(
                        m, tmp_path / "data", ObjectStore(tmp_path / "obj"), IdFactory(seed=7), UUID(int=1)
                    ).iter_sequences()
                )
            ).observations
        ]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


@dataclass(frozen=True)
class FakeTwinSample:
    observation: Observation
    supervision: SupervisionLabel | None


def twin_samples(n: int):
    ids = IdFactory(seed=3)
    mission = ids.new()
    out = []
    for i in range(n):
        obs = Observation(
            observation_id=ids.new(),
            mission_id=mission,
            run_id=ids.new(),
            trace_id=ids.new(),
            sensor_id=UUID(int=9),
            modality=Modality.PRESSURE_DEPTH,
            timestamp=TimeStamp(time_ns=(n - i) * 100, clock_domain="sim"),
            sensor_frame="depth",
            robot_pose_estimate=None,
            inline_values=(float(i),),
            inline_units="m",
        )
        targets = {"depth_m": float(i)} if i % 2 == 0 else {}
        label = SupervisionLabel(domain=Domain.SPATIAL, targets=targets, lineage="world_family=A;seed=1")
        out.append(FakeTwinSample(obs, label))
    return out


def test_synthetic_twin_adapter_orders_by_time_and_masks_labels():
    m = DatasetManifest(
        dataset_id="twin.test",
        version="1",
        source_category=SourceCategory.TWIN_SYNTHETIC,
        modalities=(Modality.PRESSURE_DEPTH,),
    )
    (seq,) = list(SyntheticTwinAdapter(m, twin_samples(4)).iter_sequences())
    times = [o.timestamp.time_ns for o in seq.observations]
    assert times == sorted(times)
    assert seq.lineage.world_family == "A" and seq.lineage.seed == "1"
    assert isinstance(seq.supervision, PartialTruth)
    assert int(seq.supervision.availability_masks["depth_m"].sum()) == 2
    with pytest.raises(DatasetContractError):
        SyntheticTwinAdapter(m.model_copy(update={"source_category": SourceCategory.PUBLIC_REAL}), [])


def unit(uid: str, partition: Partition, **kw) -> CaptureUnit:
    base = {
        "capture_unit_id": uid,
        "partition": partition,
        "source": SourceCategory.TWIN_SYNTHETIC,
        "source_ref": "twin2s@0.1",
        "owner": "conrad",
        "rights_status": RightsReviewStatus.CLEARED,
        "modalities": (Modality.RGB,),
        "expected_modalities": (Modality.RGB, Modality.SONAR),
        "clock_domain": "sim",
        "scenario_id": "sc1",
        "lineage": LineageKeys(world_family=uid),
        "source_object_digest": hashlib.sha256(uid.encode()).hexdigest(),
        "visibility": "low",
    }
    base.update(kw)
    return CaptureUnit(**base)


def test_corpus_report_marks_real_partition_blocked(tmp_path):
    units = [
        unit("wA", Partition.SSL_SYNTHETIC_TRAIN),
        unit("wB", Partition.VAL_IN_DOMAIN),
        unit("wC", Partition.TEST_IN_DOMAIN),
        unit("wD", Partition.TEST_OOD),
    ]
    report = build_corpus_acceptance_report(units)
    assert report.partition_status["ssl_real_train"] is PartitionStatus.EMPTY_BLOCKED_EXTERNAL
    assert report.real_data_status.startswith("BLOCKED_EXTERNAL") and not report.accepted
    assert report.coverage["missing_modality_pattern"]["ssl_synthetic_train"] == {"MISSING:SONAR": 1}
    assert report.coverage["turbidity_noise"]["test_ood"] == {"NOT_RECORDED": 1}
    assert "EMPTY_BLOCKED_EXTERNAL" in render_report_markdown(report)
    write_capture_units(units, tmp_path / "rows.jsonl")
    assert read_capture_units(tmp_path / "rows.jsonl") == units


def test_corpus_detects_lineage_leakage_and_rights():
    leaky = [
        unit("wA", Partition.SSL_SYNTHETIC_TRAIN),
        unit("wB", Partition.TEST_IN_DOMAIN, lineage=LineageKeys(world_family="wA")),
    ]
    assert build_corpus_acceptance_report(leaky).lineage_findings
    with pytest.raises(ValueError, match="rights-cleared"):
        unit(
            "r1",
            Partition.SSL_REAL_TRAIN,
            source=SourceCategory.CONRAD_REAL,
            rights_status=RightsReviewStatus.REVIEW_REQUIRED,
        )
    with pytest.raises(ValueError, match="Twin/Unity"):
        unit("r2", Partition.SSL_SYNTHETIC_TRAIN, source=SourceCategory.CONRAD_REAL)
