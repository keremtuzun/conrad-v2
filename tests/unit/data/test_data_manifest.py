from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from conrad.data.adapters import PUBLIC_CANDIDATES, DatasetNotAvailableError, public_adapter
from conrad.data.inventory import build_file_entries
from conrad.data.manifest import (
    DatasetManifest,
    FileEntry,
    LineageKeys,
    ManifestStatus,
    ProblemCode,
    RightsReviewStatus,
    SourceCategory,
    SplitUnit,
    StreamSync,
    SyncStatus,
    TransformationDecl,
    TransformationKind,
    dump_manifest,
    files_digest,
    load_manifest,
    verify_manifest,
    verify_manifest_by_id,
)
from conrad.data.simreal_ledger import (
    LedgerError,
    RangeRequest,
    check_randomization_request,
    load_ledger,
)
from conrad.schemas.observation import Modality
from conrad.settings import REPO_ROOT


def make_dataset(root: Path, n: int = 3) -> DatasetManifest:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        rel = f"seq_a/frame_{i}.bin"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(bytes([i]) * (10 + i))
        paths.append(rel)
    files = build_file_entries(
        root, paths, stream="rgb", modality=Modality.RGB, lineage=LineageKeys(sequence="seq_a", site="s1")
    )
    return DatasetManifest(
        dataset_id="test.local",
        version="1",
        status=ManifestStatus.VERIFIED,
        source_category=SourceCategory.CONRAD_REAL,
        source="unit-test fixture",
        license="internal-test",
        license_evidence_ref="tests/unit/data",
        usage_rights="internal testing",
        rights_review_status=RightsReviewStatus.CLEARED,
        training_allowed=True,
        modalities=(Modality.RGB,),
        split_strategy=(SplitUnit.SEQUENCE,),
        adapter_version="1.0.0",
        checksum=files_digest(files),
        files=files,
    )


def write(manifest: DatasetManifest, path: Path) -> Path:
    dump_manifest(manifest, path)
    return path


def codes(problems) -> set[ProblemCode]:
    return {p.code for p in problems}


def test_clean_manifest_verifies(tmp_path):
    m = make_dataset(tmp_path / "data")
    assert verify_manifest(write(m, tmp_path / "m.yaml"), tmp_path / "data") == []
    assert load_manifest(tmp_path / "m.yaml").manifest_digest() == m.manifest_digest()


def test_tampered_file_is_detected(tmp_path):
    m = make_dataset(tmp_path / "data")
    (tmp_path / "data" / "seq_a" / "frame_1.bin").write_bytes(b"\x09" * 11)
    assert ProblemCode.CHECKSUM_MISMATCH in codes(
        verify_manifest(write(m, tmp_path / "m.yaml"), tmp_path / "data")
    )


def test_missing_file_and_undeclared_file(tmp_path):
    m = make_dataset(tmp_path / "data")
    (tmp_path / "data" / "seq_a" / "frame_0.bin").unlink()
    (tmp_path / "data" / "extra.bin").write_bytes(b"x")
    problems = verify_manifest(write(m, tmp_path / "m.yaml"), tmp_path / "data", strict_inventory=True)
    assert {ProblemCode.MISSING_FILE, ProblemCode.UNDECLARED_FILE} <= codes(problems)


def test_manifest_edit_changes_digest_and_breaks_checksum(tmp_path):
    m = make_dataset(tmp_path / "data")
    edited = m.model_copy(update={"files": m.files[:-1]})
    assert edited.manifest_digest() != m.manifest_digest()
    problems = verify_manifest(write(edited, tmp_path / "m.yaml"), tmp_path / "data")
    assert ProblemCode.DATASET_CHECKSUM_MISMATCH in codes(problems)


def test_missing_license_fails_closed(tmp_path):
    m = make_dataset(tmp_path / "data").model_copy(update={"license": "REVIEW_REQUIRED"})
    assert ProblemCode.LICENSE_MISSING in codes(
        verify_manifest(write(m, tmp_path / "m.yaml"), tmp_path / "data")
    )


def test_rights_and_training_flags_fail_closed(tmp_path):
    m = make_dataset(tmp_path / "data").model_copy(
        update={"rights_review_status": RightsReviewStatus.IN_REVIEW, "training_allowed": False}
    )
    problems = codes(verify_manifest(write(m, tmp_path / "m.yaml"), tmp_path / "data"))
    assert {ProblemCode.RIGHTS_NOT_CLEARED, ProblemCode.TRAINING_NOT_PERMITTED} <= problems


def test_undeclared_and_silent_transformations(tmp_path):
    m = make_dataset(tmp_path / "data")
    src = m.files[0]
    derived = FileEntry(
        path="seq_a/frame_0_crop.bin",
        sha256=src.sha256,
        byte_length=src.byte_length,
        stream="rgb",
        modality=Modality.RGB,
        derived_from=src.path,
        transformation_ids=("crop_v1",),
    )
    silent = derived.model_copy(update={"path": "seq_a/frame_0_silent.bin", "transformation_ids": ()})
    files = (*m.files, derived, silent)
    bad = m.model_copy(update={"files": files, "checksum": files_digest(files)})
    for name in ("frame_0_crop.bin", "frame_0_silent.bin"):
        (tmp_path / "data" / "seq_a" / name).write_bytes((tmp_path / "data" / src.path).read_bytes())
    problems = codes(verify_manifest(write(bad, tmp_path / "m.yaml"), tmp_path / "data"))
    assert {ProblemCode.UNDECLARED_TRANSFORMATION, ProblemCode.DERIVED_WITHOUT_TRANSFORMATION} <= problems
    decl = TransformationDecl(
        transformation_id="crop_v1",
        kind=TransformationKind.TRANSFORM,
        description="center crop",
        version="1",
        config_hash="abc",
    )
    ok = bad.model_copy(update={"transformations": (decl,)})
    assert ProblemCode.UNDECLARED_TRANSFORMATION not in codes(
        verify_manifest(write(ok, tmp_path / "m.yaml"), tmp_path / "data")
    )
    assert ok.resolve_lineage(derived) == src.lineage


def test_streams_default_unsynchronized_and_sync_needs_evidence():
    m = DatasetManifest(dataset_id="x", version="1", source_category=SourceCategory.PUBLIC_REAL)
    assert m.sync_status("rgb", "ctd") is SyncStatus.UNSYNCHRONIZED
    with pytest.raises(ValueError, match="without evidence_ref"):
        StreamSync(stream_a="rgb", stream_b="ctd", status=SyncStatus.HARDWARE_SYNCHRONIZED)
    synced = m.model_copy(
        update={
            "synchronization": (
                StreamSync(
                    stream_a="rgb",
                    stream_b="imu",
                    status=SyncStatus.HARDWARE_SYNCHRONIZED,
                    evidence_ref="trigger log",
                ),
            )
        }
    )
    assert synced.sync_status("imu", "rgb") is SyncStatus.HARDWARE_SYNCHRONIZED


def test_file_paths_must_stay_inside_root():
    with pytest.raises(ValueError):
        FileEntry(path="../escape.bin", sha256="0" * 64, byte_length=1, stream="s", modality=Modality.RGB)


def test_verify_manifest_by_id(tmp_path, monkeypatch):
    m = make_dataset(tmp_path / "data" / "test.local")
    search = tmp_path / "datasets"
    search.mkdir()
    write(m, search / "local.manifest.yaml")
    monkeypatch.setenv("CONRAD_DATA_ROOT", str(tmp_path / "data"))
    assert verify_manifest_by_id("test.local", search) == []
    assert verify_manifest_by_id("test.local@1", search) == []
    assert verify_manifest_by_id("nope", search)[0].startswith("MANIFEST_INVALID")


def test_public_templates_are_unverified_and_fail_closed(tmp_path):
    for key, name in PUBLIC_CANDIDATES.items():
        path = REPO_ROOT / "datasets" / "public" / name
        m = load_manifest(path)
        assert m.status is ManifestStatus.UNVERIFIED_NOT_DOWNLOADED
        assert m.license == "REVIEW_REQUIRED" and not m.files and not m.training_allowed
        assert verify_manifest_by_id(m.dataset_id), key
        adapter = public_adapter(key)
        assert not adapter.inspect().usable
        with pytest.raises(DatasetNotAvailableError) as info:
            next(iter(adapter.iter_sequences()))
        assert name in str(info.value) and "asset inventory" in str(info.value)
        with pytest.raises(DatasetNotAvailableError):
            adapter.normalize_observation(object())


def test_simreal_ledger_loads_and_is_unmeasured():
    ledger = load_ledger(REPO_ROOT / "datasets" / "simreal_ledger.yaml")
    assert ledger.parameters and set(ledger.unmeasured()) == {p.semantic_name for p in ledger.parameters}
    assert check_randomization_request(
        ledger, [RangeRequest(semantic_name="ambient_current_speed", low=0, high=5)]
    )
    assert check_randomization_request(ledger, [RangeRequest(semantic_name="unlisted", low=0, high=1)])
    assert not check_randomization_request(
        ledger, [RangeRequest(semantic_name="ambient_current_speed", low=0, high=0.5)]
    )


def _ledger(tmp_path: Path, **row) -> Path:
    base = {
        "semantic_name": "p",
        "category": "CURRENT",
        "unit": "m/s",
        "range_low": 0.0,
        "range_high": 1.0,
        "sampling_law": "UNIFORM",
        "source_of_range": "ENGINEERING_ESTIMATE",
        "physical_explanation": "steady water current magnitude acting on the vehicle hull",
        "affected_simulator_module": "conrad.sim.kernel.dynamics",
    }
    base.update(row)
    path = tmp_path / "ledger.yaml"
    path.write_text(yaml.safe_dump({"revision": "t", "parameters": [base]}), encoding="utf-8")
    return path


def test_ledger_rejects_unexplained_or_unsourced_ranges(tmp_path):
    load_ledger(_ledger(tmp_path))
    for bad in (
        {"physical_explanation": "arbitrary"},
        {"source_of_range": "OPEN"},
        {"physical_measurement_status": "MEASURED", "source_of_range": "MEASURED"},
        {"range_low": 2.0},
    ):
        with pytest.raises(LedgerError):
            load_ledger(_ledger(tmp_path, **bad))
    measured = {
        "physical_measurement_status": "MEASURED",
        "source_of_range": "MEASURED",
        "raw_log_provenance": "logs/x",
        "identification_report_ref": "reports/id-1",
        "calibration_revision": "cal-1",
    }
    assert load_ledger(_ledger(tmp_path, **measured)).parameters[0].calibrated


def test_sha_helper_matches_hashlib(tmp_path):
    m = make_dataset(tmp_path / "d", n=1)
    assert m.files[0].sha256 == hashlib.sha256((tmp_path / "d" / m.files[0].path).read_bytes()).hexdigest()
