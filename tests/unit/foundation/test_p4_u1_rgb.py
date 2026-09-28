from __future__ import annotations

from pathlib import Path

import torch

from conrad.foundation.encoders.rgb import RGBEncoderConfig, RGBViTS14Encoder
from conrad.foundation.pretraining.losses import ObjectiveStatus
from conrad.foundation.pretraining.u1_rgb import (
    U1RGBTrainingBundle,
    contiguous_2d_mask,
    parameter_count,
    synthetic_rgb_views,
)
from conrad.training.entrypoints import run_training


def test_rgb_vit_s14_tensor_contract_and_dense_taps() -> None:
    torch.manual_seed(1)
    encoder = RGBViTS14Encoder(RGBEncoderConfig(image_size=28))
    out = encoder(torch.rand(2, 3, 28, 28), torch.zeros(2, 4, dtype=torch.bool))
    assert out.modality_repr.shape == (2, 384)
    assert out.patch_tokens.shape == (2, 4, 384)
    assert set(out.dense_taps) == {3, 6, 9, 12}
    assert all(t.shape == (2, 4, 384) for t in out.dense_taps.values())
    assert encoder.config.patch_size == 14
    assert encoder.config.depth == 12
    assert encoder.config.heads == 6
    assert 20_000_000 <= parameter_count(encoder) <= 23_000_000


def test_contiguous_60_percent_mask_is_deterministic_and_preserves_visible_tokens() -> None:
    a = contiguous_2d_mask(batch_size=2, grid_size=2, mask_fraction=0.60, min_visible_fraction=0.10, generator=torch.Generator().manual_seed(7))
    b = contiguous_2d_mask(batch_size=2, grid_size=2, mask_fraction=0.60, min_visible_fraction=0.10, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    assert a.float().mean().item() == 0.5
    assert (~a).float().mean().item() >= 0.10


def test_u1_rgb_objectives_backward_ema_and_no_truth_leakage() -> None:
    views = synthetic_rgb_views({"batch_size": 2, "image_size": 28, "token_mask_fraction": 0.60}, 20260401)
    assert all("truth" not in sample_id.lower() for sample_id in views.sample_ids)
    bundle = U1RGBTrainingBundle(
        RGBViTS14Encoder(RGBEncoderConfig(image_size=28)),
        rank_diversity_weight=1.0,
        rank_diversity_target=72.0,
    )
    assert all(not p.requires_grad for p in bundle.teacher.parameters())
    optimizer = torch.optim.AdamW(bundle.parameters(), lr=1e-4)
    before = [p.detach().clone() for p in bundle.teacher.parameters()]
    out = bundle(views)
    assert torch.isfinite(out.loss)
    assert out.rank_diversity_loss.item() >= 0.0
    assert out.rank_entropy.item() > 0.0
    statuses = {r.objective_id: r.status for r in out.results}
    assert statuses["u1_rgb_masked_latent_prediction"] is ObjectiveStatus.ACTIVE
    assert statuses["u1_rgb_global_consistency"] is ObjectiveStatus.ACTIVE
    assert statuses["u1_rgb_degradation"] is ObjectiveStatus.ACTIVE
    assert statuses["u1_rgb_metric"] is ObjectiveStatus.NOT_APPLICABLE
    out.loss.backward()
    assert any(p.grad is not None for p in bundle.student.parameters())
    optimizer.step()
    tau = bundle.update_teacher_after_optimizer(1)
    assert tau > 0.996
    assert any(not torch.equal(a, b) for a, b in zip(before, bundle.teacher.parameters(), strict=True))


def test_osfm_u1_rgb_smoke_checkpoint_reload_and_replay(tmp_path: Path) -> None:
    result = run_training("configs/train/osfm/u1_rgb_smoke.yaml", runs_root=tmp_path)
    assert result["experiment_id"] == "OSFM-U1-RGB-SMOKE-001"
    assert result["architecture"]["layers"] == 12
    assert result["architecture"]["heads"] == 6
    assert result["reload"]["matches"] is True
    assert result["replay"]["deterministic"] is True
    assert result["representation_health"]["finite"] is True
    assert result["representation_health"]["collapse_score"] > 0
    assert result["representation_health"]["formal_rank_guard"] == "NOT_EVALUABLE"
    assert Path(result["checkpoint"]).is_file()
