from __future__ import annotations

from pathlib import Path

import torch

from conrad.foundation.encoders.range import RangeEncoderConfig, RangeViTP8Encoder
from conrad.foundation.pretraining.losses import ObjectiveStatus
from conrad.foundation.pretraining.u1_range import U1RangeTrainingBundle, synthetic_range_views
from conrad.foundation.pretraining.u1_rgb import contiguous_2d_mask, parameter_count
from conrad.training.entrypoints import run_training


def test_range_vit_p8_contract_validity_and_dense_taps() -> None:
    torch.manual_seed(1)
    encoder = RangeViTP8Encoder(RangeEncoderConfig(image_size=32))
    x = torch.rand(2, 1, 32, 32)
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[:, :, :8, :8] = False
    out = encoder(x, valid, torch.zeros(2, 16, dtype=torch.bool))
    assert out.modality_repr.shape == (2, 384)
    assert out.patch_tokens.shape == (2, 16, 384)
    assert out.internal_patch_tokens.shape == (2, 16, 256)
    assert set(out.dense_taps) == {2, 4, 6}
    assert all(t.shape == (2, 16, 256) for t in out.dense_taps.values())
    assert all(t.shape == (2, 16, 384) for t in out.dense_taps_projected.values())
    assert out.patch_valid_fraction[:, 0].eq(0).all()
    assert encoder.config.patch_size == 8
    assert encoder.config.width == 256
    assert encoder.config.output_dim == 384
    assert encoder.config.depth == 6
    assert encoder.config.heads == 4
    assert 4_800_000 <= parameter_count(encoder) <= 5_000_000


def test_range_50_percent_contiguous_mask_is_deterministic_and_visible() -> None:
    a = contiguous_2d_mask(
        batch_size=2,
        grid_size=4,
        mask_fraction=0.50,
        min_visible_fraction=0.10,
        generator=torch.Generator().manual_seed(12),
    )
    b = contiguous_2d_mask(
        batch_size=2,
        grid_size=4,
        mask_fraction=0.50,
        min_visible_fraction=0.10,
        generator=torch.Generator().manual_seed(12),
    )
    assert torch.equal(a, b)
    assert a.float().mean().item() == 0.5
    assert (~a).float().mean().item() >= 0.10


def test_range_views_record_validity_aware_corruptions_and_no_truth_leakage() -> None:
    views = synthetic_range_views({"batch_size": 2, "image_size": 32, "token_mask_fraction": 0.50}, 20260412)
    assert views.teacher_range.shape == (2, 1, 32, 32)
    assert views.student_range.shape == (2, 1, 32, 32)
    assert views.teacher_validity_mask.shape == views.teacher_range.shape
    assert not views.teacher_validity_mask.all()
    assert views.teacher_range.masked_select(~views.teacher_validity_mask).eq(0).all()
    assert all("truth" not in sample_id.lower() for sample_id in views.sample_ids)
    assert set(views.corruption_trace) == {
        "bounded_metric_noise",
        "validity_mask_aware_dropout",
        "quantization",
        "saturation",
    }


def test_u1_range_objectives_backward_ema_and_metric_denominator_routing() -> None:
    views = synthetic_range_views({"batch_size": 2, "image_size": 32, "token_mask_fraction": 0.50}, 20260412)
    bundle = U1RangeTrainingBundle(RangeViTP8Encoder(RangeEncoderConfig(image_size=32)))
    assert all(not p.requires_grad for p in bundle.teacher.parameters())
    optimizer = torch.optim.AdamW(bundle.parameters(), lr=1e-4)
    before = [p.detach().clone() for p in bundle.teacher.parameters()]
    out = bundle(views)
    assert torch.isfinite(out.loss)
    statuses = {r.objective_id: r.status for r in out.results}
    assert statuses["u1_range_masked_latent_prediction"] is ObjectiveStatus.ACTIVE
    assert statuses["u1_range_global_consistency"] is ObjectiveStatus.ACTIVE
    assert statuses["u1_range_degradation"] is ObjectiveStatus.ACTIVE
    assert statuses["u1_range_metric_reconstruction"] is ObjectiveStatus.ACTIVE
    metric = {r.objective_id: r for r in out.results}["u1_range_metric_reconstruction"]
    expected_denominator = (
        views.metric_validity_mask.unfold(2, 8, 8).unfold(3, 8, 8)
        .contiguous()
        .view(2, 1, -1, 64)
        .squeeze(1)
        .logical_and(views.token_mask.unsqueeze(-1))
        .float()
        .sum()
    )
    assert metric.denominator.item() == expected_denominator.item()
    out.loss.backward()
    assert any(p.grad is not None for p in bundle.student.parameters())
    optimizer.step()
    tau = bundle.update_teacher_after_optimizer(1)
    assert tau > 0.996
    assert any(not torch.equal(a, b) for a, b in zip(before, bundle.teacher.parameters(), strict=True))
    assert not bundle.teacher.training


def test_u1_range_metric_not_applicable_when_no_valid_metric_support() -> None:
    views = synthetic_range_views(
        {"batch_size": 2, "image_size": 32, "token_mask_fraction": 0.50}, 20260412, all_invalid=True
    )
    bundle = U1RangeTrainingBundle(RangeViTP8Encoder(RangeEncoderConfig(image_size=32)))
    out = bundle(views)
    metric = {r.objective_id: r for r in out.results}["u1_range_metric_reconstruction"]
    assert metric.status is ObjectiveStatus.NOT_APPLICABLE
    assert metric.denominator.item() == 0.0
    assert torch.isfinite(out.loss)


def test_osfm_u1_range_smoke_checkpoint_reload_and_replay(tmp_path: Path) -> None:
    result = run_training("configs/train/osfm/u1_range_smoke.yaml", runs_root=tmp_path)
    assert result["experiment_id"] == "OSFM-U1-RANGE-SMOKE-001"
    assert result["architecture"]["encoder"] == "RangeViTP8Encoder"
    assert result["architecture"]["patch_size"] == 8
    assert result["architecture"]["internal_width"] == 256
    assert result["architecture"]["output_dim"] == 384
    assert result["architecture"]["layers"] == 6
    assert result["architecture"]["heads"] == 4
    assert result["data"]["public_real_dataset_used"] is False
    assert result["data"]["metric_objective_eligible"] is True
    assert result["reload"]["matches"] is True
    assert result["replay"]["deterministic"] is True
    assert result["metrics"]["mask_fraction_requested"] == 0.5
    assert result["metrics"]["mask_fraction_actual"] == 0.5
    assert result["metrics"]["validity_coverage"] < 1.0
    metric = result["objectives"]["u1_range_metric_reconstruction"]
    assert metric["u1_range_metric_reconstruction/status"] == "ACTIVE"
    assert result["representation_health"]["finite"] is True
    assert result["representation_health"]["collapse_score"] > 0
    assert result["representation_health"]["formal_rank_guard"] == "NOT_EVALUABLE"
    assert Path(result["checkpoint"]).is_file()
