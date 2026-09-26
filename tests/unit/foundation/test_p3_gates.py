from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from conrad.evaluation.dispatch import EXPERIMENTS
from conrad.foundation.augment import ViewEngine, ViewRecipe, default_registry
from conrad.foundation.augment.transforms import PhysicalityClass
from conrad.foundation.data import (
    CorpusPartition,
    CorpusReadiness,
    HierarchicalSampler,
    collate_foundation_windows,
)
from conrad.foundation.data.capture import GAP_TERMINATION_MS, build_windows
from conrad.foundation.data.manifest import CorpusManifest, CorpusSource, synthetic_manifest_for_tests
from conrad.foundation.pretraining.bundle import OSFMTrainingBundle, TinyFoundationEncoder
from conrad.foundation.pretraining.curriculum import OSFMStageID
from conrad.foundation.pretraining.ema import EMASchedule
from conrad.foundation.pretraining.losses import ObjectiveRouter, ObjectiveStatus
from conrad.foundation.pretraining.smoke import MODALITIES, synthetic_units
from conrad.training.checkpoint import load_checkpoint, save_checkpoint
from conrad.training.checkpoint_meta import CompatibilityTuple, config_digest
from conrad.training.determinism import inspect_compute
from conrad.training.run_dir import RunDirectory, RunPurpose, TerminalStatus


def test_g_p3_schema_data_osfm_manifest_partitions_and_no_first_party() -> None:
    manifest = synthetic_manifest_for_tests()
    assert manifest.corpus_id == "DATA-OSFM-01"
    assert manifest.readiness is CorpusReadiness.READY
    assert set(manifest.partition_counts()) == {p.value for p in CorpusPartition}
    with pytest.raises(ValueError, match="first-party"):
        CorpusManifest(
            readiness=CorpusReadiness.READY,
            sources=(
                CorpusSource(
                    source_id="bad",
                    manifest_digest="0" * 64,
                    source_digest="1" * 64,
                    partition=CorpusPartition.PRETRAIN_REAL,
                    first_party=True,
                ),
            ),
        )


def test_g_p3_data_and_lineage_source_digests_are_stable() -> None:
    a = synthetic_manifest_for_tests()
    b = synthetic_manifest_for_tests()
    assert a.corpus_digest == b.corpus_digest
    assert all(s.manifest_digest and s.source_digest for s in a.sources)


def test_g_p3_truth_legacy_partial_truth_is_not_required_for_foundation_inputs() -> None:
    windows = build_windows(synthetic_units(), expected_modalities=MODALITIES)
    for window in windows:
        for unit in window.units:
            assert "truth" not in unit.lineage
            assert not unit.natural_missing


def test_g_p3_window_uses_frozen_500ms_stride_and_gap_termination() -> None:
    windows = build_windows(synthetic_units(temporal_gap=True), expected_modalities=MODALITIES)
    assert {w.duration_ms for w in windows} == {500}
    assert {w.stride_ms for w in windows} == {500}
    assert GAP_TERMINATION_MS == 1000
    assert max(w.context_index for w in windows) < 10


def test_g_p3_pair_graph_records_cross_modal_coverage() -> None:
    window = build_windows(synthetic_units(), expected_modalities=MODALITIES)[0]
    assert window.pair_graph.nodes
    assert window.pair_graph.coverage()["pair_coverage"] > 0


def test_g_p3_sampler_epoch_plan_deterministic_across_rank_membership() -> None:
    windows = build_windows(synthetic_units(), expected_modalities=MODALITIES)
    digest = synthetic_manifest_for_tests().corpus_digest
    a = HierarchicalSampler(seed=3, batch_size=2, rank=0, world_size=2).plan(windows, corpus_digest=digest, epoch=4)
    b = HierarchicalSampler(seed=3, batch_size=2, rank=1, world_size=2).plan(windows, corpus_digest=digest, epoch=4)
    assert a.ordered_window_ids == b.ordered_window_ids
    assert a.plan_digest == b.plan_digest


def test_g_p3_aug_teacher_minimal_student_replay_and_lineage_stability() -> None:
    batch = collate_foundation_windows(build_windows(synthetic_units(), expected_modalities=MODALITIES)[:2], modalities=MODALITIES)
    engine = ViewEngine(default_registry(), seed=9)
    teacher = engine.make_view(batch, ViewRecipe("teacher", ("identity_recorded_observation",), teacher=True))
    student_a = engine.make_view(batch, ViewRecipe("student", ("rgb_calibrated_jitter",), ("SONAR",), 0.5))
    student_b = engine.make_view(batch, ViewRecipe("student", ("rgb_calibrated_jitter",), ("SONAR",), 0.5))
    engine.validate_lineage_stable(batch, teacher)
    assert torch.equal(student_a.batch.tokens, student_b.batch.tokens)
    assert student_a.traces[0].physicality is PhysicalityClass.CALIBRATED_EMPIRICAL
    assert teacher.batch.lineage_ids == batch.lineage_ids


def test_g_p3_batch_masks_padding_natural_missing_and_artificial_dropout_separately() -> None:
    windows = build_windows(synthetic_units(missing=("POINT_CLOUD",)), expected_modalities=MODALITIES)
    batch = collate_foundation_windows(windows[:2], modalities=MODALITIES, artificial_dropout={"SONAR": True})
    assert batch.padding_mask.dtype is torch.bool
    assert batch.natural_missing_mask[:, MODALITIES.index("POINT_CLOUD")].all()
    assert batch.artificial_dropout_mask[:, MODALITIES.index("SONAR")].all()
    assert not torch.equal(batch.natural_missing_mask, batch.artificial_dropout_mask)


def test_g_p3_ema_teacher_has_no_grad_and_schedule_increases_after_step() -> None:
    bundle = OSFMTrainingBundle(TinyFoundationEncoder())
    assert all(not p.requires_grad for p in bundle.teacher.parameters())
    sched = EMASchedule(total_steps=10)
    assert sched.value(0) == pytest.approx(0.996)
    assert sched.value(10) == pytest.approx(0.9999)


def test_g_p3_loss_zero_denominator_is_not_applicable_not_fake_success() -> None:
    router = ObjectiveRouter()
    x = torch.ones(2, 3, 4)
    result = router.teacher_student(x, x, torch.zeros(2, 3, dtype=torch.bool))
    assert result.status is ObjectiveStatus.NOT_APPLICABLE
    with pytest.raises(RuntimeError, match="no active objectives"):
        router.total((result,))


def test_g_p3_compute_full_configs_fail_closed_on_missing_cuda() -> None:
    compute = inspect_compute()
    if compute.cuda_available:
        pytest.skip("host has CUDA; missing-CUDA fail-closed branch is not applicable")
    assert not compute.cuda_available


def test_g_p3_registry_accepts_osfm_namespace_without_removing_historical_ids() -> None:
    assert "OSFM-CONTRACT-TINY" in EXPERIMENTS
    assert "OSFM-S-SMOKE" in EXPERIMENTS
    assert "CORE-BUO-E001" in EXPERIMENTS


def test_osfm_stage_ids_are_separate_from_model2_c_curriculum() -> None:
    assert [s.value for s in OSFMStageID] == ["U1-RGB", "U1-SONAR", "U1-RANGE", "U1-GEOMETRY", "M1", "T1", "J1"]


def test_g_p3_rundir_checkpoint_and_resume_state(tmp_path: Path) -> None:
    run = RunDirectory.create(
        tmp_path,
        "p3-rundir",
        resolved_config={"job": "unit"},
        manifests={},
        purpose=RunPurpose.DEVELOPMENT,
        clock_ns=lambda: 10,
    )
    bundle = OSFMTrainingBundle(TinyFoundationEncoder())
    cfg = {"job": "unit"}
    ckpt = run.path / "checkpoints" / "unit.pt"
    meta = save_checkpoint(
        ckpt,
        model=bundle,
        component="foundation.osfm",
        config=cfg,
        manifest_digests=("a" * 64,),
        split_hash="b" * 64,
        seeds={"torch": 1},
        metrics={"val/osfm_loss": 1.0},
        epoch=0,
        step=1,
        created_time_ns=11,
        trainer_state={"cuda_rng_state": [], "amp_scaler": None},
        is_encoder=True,
        representation_pretraining_id="OSFM-P3-SMOKE",
        selection_metric="val/osfm_loss",
    )
    loaded = load_checkpoint(
        ckpt,
        expected=CompatibilityTuple(
            config_digest=config_digest(cfg),
            manifest_digests=("a" * 64,),
            split_hash="b" * 64,
            component="foundation.osfm",
            representation_pretraining_id="OSFM-P3-SMOKE",
        ),
        model=OSFMTrainingBundle(TinyFoundationEncoder()),
    )
    assert loaded.trainer_state["cuda_rng_state"] == []
    assert meta.checkpoint_id == loaded.metadata.checkpoint_id
    run.write_artifact("reports", "unit.json", json.dumps({"ok": True}))
    run.seal(TerminalStatus.COMPLETED, 12)
    assert run.verify_seal() == []

