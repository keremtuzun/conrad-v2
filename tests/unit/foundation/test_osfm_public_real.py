from __future__ import annotations

import json
from pathlib import Path

import pytest

from conrad.data.manifest import data_root_for
from conrad.foundation.data.manifest import CorpusPartition, load_corpus_manifest
from conrad.foundation.data.osfm_public_real import (
    SUBPIPE_MANIFEST,
    build_subpipe_osfm_evidence,
    validate_subpipe_osfm_evidence,
    write_subpipe_osfm_corpus,
)
from conrad.foundation.pretraining import readiness


def _evidence() -> dict:
    return {
        "dataset_id": "public.subpipe",
        "manifest_digest": "a" * 64,
        "source_digest": "b" * 64,
        "source_frame_count": 6,
        "partition_counts": {
            "PRETRAIN_REAL": 3,
            "VALIDATION": 1,
            "FINAL_TEST": 1,
            "OOD_TEST": 1,
        },
        "lineage_counts": {
            "PRETRAIN_REAL": 1,
            "VALIDATION": 1,
            "FINAL_TEST": 1,
            "OOD_TEST": 1,
        },
        "partition_overlap_counts": {
            "PRETRAIN_REAL__VALIDATION": 0,
            "PRETRAIN_REAL__FINAL_TEST": 0,
            "PRETRAIN_REAL__OOD_TEST": 0,
            "VALIDATION__FINAL_TEST": 0,
            "VALIDATION__OOD_TEST": 0,
            "FINAL_TEST__OOD_TEST": 0,
        },
        "partition_digests": {
            "PRETRAIN_REAL": "1" * 64,
            "VALIDATION": "2" * 64,
            "FINAL_TEST": "3" * 64,
            "OOD_TEST": "4" * 64,
        },
        "train_only_normalization": {
            "fit_partitions": ["PRETRAIN_REAL"],
            "excluded_partitions": ["VALIDATION", "FINAL_TEST", "OOD_TEST"],
            "source_frame_count": 3,
            "source_digest": "1" * 64,
        },
        "license": "CC-BY-4.0",
        "rights_review_status": "CLEARED",
        "training_allowed": True,
        "verification": {
            "adapter_usable": True,
            "problem_count": 0,
            "problems": [],
            "smoke_loaded_first_payload": True,
        },
        "lineage_strategy": {
            "split_unit": "contiguous time block within each sonar stream",
            "random_frame_split": False,
            "limitation": "one mission/sequence only",
        },
        "first_payload": {
            "shape": [500, 2500, 3],
            "sonar_rendering": "colormapped PPM, not raw backscatter",
        },
    }


def test_subpipe_public_real_evidence_accepts_matching_lineage_safe_contract(monkeypatch) -> None:
    evidence = _evidence()
    monkeypatch.setattr(
        "conrad.foundation.data.osfm_public_real.build_subpipe_osfm_evidence",
        lambda: dict(evidence),
    )

    assert validate_subpipe_osfm_evidence(evidence) == []
    assert evidence["lineage_strategy"]["random_frame_split"] is False
    assert evidence["partition_overlap_counts"]["PRETRAIN_REAL__FINAL_TEST"] == 0
    assert evidence["train_only_normalization"]["fit_partitions"] == ["PRETRAIN_REAL"]
    assert "FINAL_TEST" in evidence["train_only_normalization"]["excluded_partitions"]
    assert evidence["first_payload"]["sonar_rendering"] == "colormapped PPM, not raw backscatter"


def test_subpipe_public_real_evidence_fails_closed_for_missing_or_wrong_payload(monkeypatch) -> None:
    evidence = _evidence()

    def missing_payload() -> dict:
        raise FileNotFoundError("raw/SubPipeMini2.zip")

    monkeypatch.setattr(
        "conrad.foundation.data.osfm_public_real.build_subpipe_osfm_evidence",
        missing_payload,
    )
    assert "could not be rebuilt" in validate_subpipe_osfm_evidence(evidence)[0]

    live = _evidence()
    monkeypatch.setattr(
        "conrad.foundation.data.osfm_public_real.build_subpipe_osfm_evidence",
        lambda: live,
    )
    wrong = _evidence()
    wrong["source_frame_count"] = 5
    assert "source_frame_count" in " ".join(validate_subpipe_osfm_evidence(wrong))


def test_subpipe_public_real_evidence_rejects_leakage_license_and_rendering_regressions(monkeypatch) -> None:
    live = _evidence()
    monkeypatch.setattr(
        "conrad.foundation.data.osfm_public_real.build_subpipe_osfm_evidence",
        lambda: live,
    )
    bad = _evidence()
    bad["partition_overlap_counts"]["PRETRAIN_REAL__VALIDATION"] = 1
    bad["train_only_normalization"]["fit_partitions"] = ["PRETRAIN_REAL", "VALIDATION"]
    bad["license"] = "UNKNOWN"
    bad["first_payload"]["sonar_rendering"] = "raw acoustic backscatter"

    problems = " ".join(validate_subpipe_osfm_evidence(bad))
    assert "partition overlaps" in problems
    assert "normalization/statistics" in problems
    assert "license gate" in problems
    assert "rendered side-scan sonar limitation" in problems


def test_public_real_corpus_preserves_required_partitions_and_evidence_gate(
    tmp_path: Path, monkeypatch
) -> None:
    evidence = _evidence()
    evidence["evidence_digest"] = "e" * 64
    evidence_path = tmp_path / "subpipe_p47b_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    corpus_path = tmp_path / "public_real_subpipe_p47b.yaml"

    write_subpipe_osfm_corpus(corpus_path, evidence_path)
    corpus = load_corpus_manifest(corpus_path)
    assert corpus.partition_counts() == {
        "PRETRAIN_REAL": 1,
        "PRETRAIN_SYNTHETIC": 1,
        "VALIDATION": 1,
        "FINAL_TEST": 1,
        "OOD_TEST": 1,
    }
    assert "synthetic readiness fixture" not in " ".join(corpus.notes).lower()

    def fake_loader(path: str | Path) -> dict:
        return evidence

    monkeypatch.setattr(readiness, "load_subpipe_osfm_evidence", fake_loader)
    monkeypatch.setattr(readiness, "validate_subpipe_osfm_evidence", lambda data: [])
    data, blockers, synthetic = readiness._data_findings(corpus_path)
    assert synthetic is False
    assert data["corpus_public_real_evidence"] == evidence
    assert "EXT-DATA-PUBLIC-REAL-01" not in {b.blocker_id for b in blockers}


def test_partition_ranges_do_not_random_frame_split_real_subpipe_when_available() -> None:
    if not (data_root_for("public.subpipe") / "raw" / "SubPipeMini2.zip").exists():
        pytest.skip("SubPipeMini2.zip is local and gitignored")
    assert SUBPIPE_MANIFEST.exists()
    evidence = build_subpipe_osfm_evidence(smoke_load=False)
    assert evidence["lineage_strategy"]["random_frame_split"] is False
    assert evidence["lineage_strategy"]["split_unit"] == "contiguous time block within each sonar stream"
    assert evidence["streams"]["sss_lf"]["frame_count"] == 1055
    assert evidence["partition_counts"][CorpusPartition.PRETRAIN_REAL.value] > 0
