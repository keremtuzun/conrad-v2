from __future__ import annotations

import torch

from conrad.foundation.pretraining.readiness import (
    GateDecision,
    run_p47_readiness,
    validate_representation_health,
)


def test_p47_readiness_fails_closed_for_external_assets() -> None:
    report = run_p47_readiness(
        corpus_manifest="configs/data/osfm/synthetic_ready.yaml",
        init_path="artifacts/external/dinov2/missing-for-test.pth",
        min_vram_gb=999,
    )
    assert report.status == "VALIDATED-RUN"
    assert report.decision is GateDecision.CONDITIONAL_GO
    blockers = {b.blocker_id: b for b in report.blockers}
    assert blockers["EXT-DATA-PUBLIC-REAL-01"].kind.value == "EXTERNAL"
    assert blockers["EXT-DINOV2-VITS14-01"].status == "BLOCKED_EXTERNAL"
    assert blockers["EXT-COMPUTE-P4-RESEARCH-01"].status == "BLOCKED_EXTERNAL"
    assert report.synthetic_staging_permitted is True
    assert "OSFM-S-PRETRAIN-V1-CANDIDATE only" in " ".join(report.promotion_rules)


def test_representation_health_requires_independent_support() -> None:
    small = torch.eye(32, 384)
    small_report = validate_representation_health(
        small,
        modality_counts={"rgb": 8, "sonar": 8, "range": 8, "geometry": 8},
    )
    assert small_report["rank_evaluable"] is False
    assert small_report["effective_rank_pass"] is False

    full = torch.eye(80, 384)
    full_report = validate_representation_health(
        full,
        modality_counts={"rgb": 20, "sonar": 20, "range": 20, "geometry": 20},
    )
    assert full_report["rank_evaluable"] is True
    assert full_report["effective_rank"] >= 64.0
    assert full_report["effective_rank_pass"] is True
    assert full_report["finite"] is True
    assert full_report["modality_coverage_pass"] is True


def test_stage_plan_locks_research_sequence_and_candidate_boundary() -> None:
    report = run_p47_readiness(min_vram_gb=999)
    stage_ids = [stage.stage_id for stage in report.stage_plan]
    assert stage_ids == [
        "U1-RGB-RESEARCH",
        "U1-SONAR-RESEARCH",
        "U1-RANGE-RESEARCH",
        "U1-GEOMETRY-RESEARCH",
        "U1-CONTEXT-RESEARCH",
        "M1-RESEARCH",
        "T1-RESEARCH",
        "J1-RESEARCH",
    ]
    assert report.architecture_revision["revision_id"] == "OSFM-P4-CONTEXT-R02"
    assert report.architecture_revision["architecture_freeze_doc"] == "docs/OSFM_ARCHITECTURE_FREEZE.md"
    assert "dual_path_model2" in report.architecture_revision
    j1 = report.stage_plan[-1]
    assert j1.optimizer_steps == 150_000
    assert "OSFM-FQ" in " ".join(j1.promotion_criteria)
    assert all(stage.effective_batch_target == 256 for stage in report.stage_plan)
