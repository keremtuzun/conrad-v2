from __future__ import annotations

from pathlib import Path

import torch

from conrad.foundation.fusion.scene_fusion import ModalityTokenSet, SceneFusionConfig, SceneFusionTransformer
from conrad.foundation.pretraining.losses import ObjectiveStatus
from conrad.foundation.pretraining.m1_fusion import M1FusionTrainingBundle, parent_checkpoint_status, synthetic_m1_fixture
from conrad.training.entrypoints import run_training


def _tokens(modality: str, batch: int = 2, tokens: int = 3) -> ModalityTokenSet:
    base = torch.arange(batch * tokens * 384, dtype=torch.float32).view(batch, tokens, 384) / 1000.0
    return ModalityTokenSet(
        modality=modality,
        tokens=base,
        valid_token_mask=torch.ones(batch, tokens, dtype=torch.bool),
        naturally_missing=torch.zeros(batch, dtype=torch.bool),
        artificially_dropped=torch.zeros(batch, dtype=torch.bool),
        padded=torch.zeros(batch, dtype=torch.bool),
    )


def test_scene_fusion_contracts_all_nonempty_modality_subsets() -> None:
    model = SceneFusionTransformer(SceneFusionConfig())
    modalities = ("rgb", "sonar", "range", "geometry")
    for mask in range(1, 1 << len(modalities)):
        selected = tuple(_tokens(modality) for idx, modality in enumerate(modalities) if mask & (1 << idx))
        out = model(selected)
        assert out.scene_latents.shape == (2, 64, 384)
        assert out.global_repr.shape == (2, 384)
        assert out.modality_presence.all()
        assert out.modality_names == tuple(s.modality for s in selected)


def test_scene_fusion_has_no_anchor_modality_assumption() -> None:
    model = SceneFusionTransformer(SceneFusionConfig())
    for modality in ("rgb", "sonar", "range", "geometry"):
        out = model((_tokens(modality),))
        assert out.scene_latents.shape == (2, 64, 384)
        assert out.global_repr.shape == (2, 384)


def test_missing_dropout_and_pairgraph_objective_routing() -> None:
    fixture = synthetic_m1_fixture({"batch_size": 4}, 20260414)
    bundle = M1FusionTrainingBundle(SceneFusionTransformer(SceneFusionConfig()))
    out = bundle(fixture)
    objectives = {result.objective_id: result for result in out.results}
    assert objectives["m1_cross_modal_consistency"].status is ObjectiveStatus.ACTIVE
    assert objectives["m1_cross_modal_consistency"].denominator.item() == 1.0
    assert objectives["m1_missing_modality"].status is ObjectiveStatus.ACTIVE
    assert objectives["m1_missing_modality"].denominator.item() >= 1.0
    assert objectives["m1_geometry_consistency"].status is ObjectiveStatus.ACTIVE
    assert objectives["m1_metric_reconstruction"].status is ObjectiveStatus.ACTIVE
    assert out.student.natural_missing_mask.any()
    assert out.student.artificial_dropout_mask.any()
    assert not out.student.padding_mask.any()
    assert fixture.relation_matrix[-1][1].value == "UNPAIRED"


def test_m1a_backward_optimizer_ema_and_freeze_policy() -> None:
    fixture = synthetic_m1_fixture({"batch_size": 4}, 20260414)
    bundle = M1FusionTrainingBundle(SceneFusionTransformer(SceneFusionConfig()))
    before = bundle.teacher.scene_latents.detach().clone()
    opt = torch.optim.AdamW((p for p in bundle.parameters() if p.requires_grad), lr=2e-4)
    out = bundle(fixture)
    out.loss.backward()
    assert any(p.grad is not None for p in bundle.fusion.parameters())
    opt.step()
    momentum = bundle.update_teacher_after_optimizer(1)
    assert 0.0 < momentum < 1.0
    assert not torch.equal(before, bundle.teacher.scene_latents)


def test_m1_fixture_replay_and_parent_digest_status() -> None:
    a = synthetic_m1_fixture({"batch_size": 4}, 20260414)
    b = synthetic_m1_fixture({"batch_size": 4}, 20260414)
    assert a.sample_ids == b.sample_ids
    assert a.artificial_dropout_plan == b.artificial_dropout_plan
    assert a.relation_matrix == b.relation_matrix
    statuses = parent_checkpoint_status()
    assert set(statuses) == {"rgb", "sonar", "range", "geometry"}
    assert all("digest_matches" in status for status in statuses.values())


def test_osfm_m1_fusion_smoke_cli_registered(tmp_path: Path) -> None:
    result = run_training("configs/train/osfm/m1_fusion_smoke.yaml", runs_root=tmp_path)
    assert result["experiment_id"] == "OSFM-M1-FUSION-SMOKE-001"
    assert result["component"] == "foundation.osfm.m1_fusion"
    assert result["architecture"]["outputs"]["scene_latents"] == [4, 64, 384]
    assert result["architecture"]["outputs"]["global_repr"] == [4, 384]
    assert result["objectives"]["m1_cross_modal_consistency"]["m1_cross_modal_consistency/status"] == "ACTIVE"
    assert result["reload"]["matches"] is True
    assert result["replay"]["deterministic"] is True
