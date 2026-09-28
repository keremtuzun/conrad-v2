from __future__ import annotations

from pathlib import Path

import torch

from conrad.foundation.encoders.geometry import GeometryEncoderConfig, GeometryGroupedEncoder
from conrad.foundation.pretraining.losses import ObjectiveStatus
from conrad.foundation.pretraining.u1_geometry import (
    U1GeometryTrainingBundle,
    deterministic_group_mask,
    synthetic_geometry_views,
)
from conrad.foundation.pretraining.u1_rgb import parameter_count
from conrad.training.entrypoints import run_training


def _cfg() -> GeometryEncoderConfig:
    return GeometryEncoderConfig(num_groups=8, group_size=16)


def test_geometry_grouped_encoder_contract_validity_and_dense_taps() -> None:
    torch.manual_seed(1)
    encoder = GeometryGroupedEncoder(_cfg())
    points = torch.rand(2, 128, 3)
    valid = torch.ones(2, 128, dtype=torch.bool)
    valid[:, ::9] = False
    out = encoder(points, valid, torch.zeros(2, 8, dtype=torch.bool))
    assert out.modality_repr.shape == (2, 384)
    assert out.group_tokens.shape == (2, 8, 384)
    assert out.internal_group_tokens.shape == (2, 8, 256)
    assert out.grouped_points.shape == (2, 8, 16, 3)
    assert out.group_valid_mask.shape == (2, 8, 16)
    assert set(out.dense_taps) == {2, 4, 6}
    assert all(t.shape == (2, 8, 256) for t in out.dense_taps.values())
    assert all(t.shape == (2, 8, 384) for t in out.dense_taps_projected.values())
    assert encoder.config.width == 256
    assert encoder.config.output_dim == 384
    assert encoder.config.depth == 6
    assert encoder.config.heads == 4
    assert 4_900_000 <= parameter_count(encoder) <= 5_100_000


def test_geometry_grouping_is_deterministic_and_uses_valid_points_only() -> None:
    views = synthetic_geometry_views({"batch_size": 2, "points_per_sample": 128, "num_groups": 8, "group_size": 16}, 99)
    encoder = GeometryGroupedEncoder(_cfg())
    a = encoder.group_points(views.teacher_points, views.teacher_validity_mask)
    b = encoder.group_points(views.teacher_points, views.teacher_validity_mask)
    assert torch.equal(a[3], b[3])
    gathered_valid = views.teacher_validity_mask.gather(1, a[3].reshape(2, -1)).view(2, 8, 16)
    assert torch.equal(a[2], gathered_valid)
    assert a[2].any()


def test_geometry_50_percent_group_mask_is_deterministic_and_visible() -> None:
    a = deterministic_group_mask(
        batch_size=2,
        num_groups=8,
        mask_fraction=0.50,
        min_visible_fraction=0.10,
        generator=torch.Generator().manual_seed(12),
    )
    b = deterministic_group_mask(
        batch_size=2,
        num_groups=8,
        mask_fraction=0.50,
        min_visible_fraction=0.10,
        generator=torch.Generator().manual_seed(12),
    )
    assert torch.equal(a, b)
    assert a.float().mean().item() == 0.5
    assert (~a).float().mean().item() >= 0.10


def test_geometry_views_record_permitted_corruptions_and_no_truth_leakage() -> None:
    views = synthetic_geometry_views({"batch_size": 2, "points_per_sample": 128, "num_groups": 8, "group_size": 16}, 20260413)
    assert views.teacher_points.shape == (2, 128, 3)
    assert views.student_points.shape == (2, 128, 3)
    assert not views.teacher_validity_mask.all()
    assert views.teacher_points.masked_select(~views.teacher_validity_mask.unsqueeze(-1)).eq(0).all()
    assert all("truth" not in sample_id.lower() for sample_id in views.sample_ids)
    assert set(views.corruption_trace) == {
        "measurement_direction_xyz_noise",
        "validity_mask_aware_point_dropout",
        "density_reduction",
        "xyz_quantization",
    }


def test_u1_geometry_objectives_backward_ema_and_metric_denominator_routing() -> None:
    views = synthetic_geometry_views({"batch_size": 2, "points_per_sample": 128, "num_groups": 8, "group_size": 16}, 20260413)
    bundle = U1GeometryTrainingBundle(
        GeometryGroupedEncoder(_cfg()),
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
    assert statuses["u1_geometry_masked_latent_prediction"] is ObjectiveStatus.ACTIVE
    assert statuses["u1_geometry_global_consistency"] is ObjectiveStatus.ACTIVE
    assert statuses["u1_geometry_degradation"] is ObjectiveStatus.ACTIVE
    assert statuses["u1_geometry_metric_reconstruction"] is ObjectiveStatus.ACTIVE
    metric = {r.objective_id: r for r in out.results}["u1_geometry_metric_reconstruction"]
    gathered_valid = views.metric_validity_mask.gather(1, out.student.neighbor_indices.reshape(2, -1)).view(2, 8, 16)
    expected_denominator = (gathered_valid & views.group_mask.unsqueeze(-1)).float().sum() * 3.0
    assert metric.denominator.item() == expected_denominator.item()
    out.loss.backward()
    assert any(p.grad is not None for p in bundle.student.parameters())
    optimizer.step()
    tau = bundle.update_teacher_after_optimizer(1)
    assert tau > 0.996
    assert any(not torch.equal(a, b) for a, b in zip(before, bundle.teacher.parameters(), strict=True))
    assert not bundle.teacher.training


def test_u1_geometry_metric_not_applicable_when_no_valid_metric_support() -> None:
    views = synthetic_geometry_views(
        {"batch_size": 2, "points_per_sample": 128, "num_groups": 8, "group_size": 16},
        20260413,
        all_invalid=True,
    )
    bundle = U1GeometryTrainingBundle(GeometryGroupedEncoder(_cfg()))
    out = bundle(views)
    metric = {r.objective_id: r for r in out.results}["u1_geometry_metric_reconstruction"]
    assert metric.status is ObjectiveStatus.NOT_APPLICABLE
    assert metric.denominator.item() == 0.0
    assert torch.isfinite(out.loss)


def test_osfm_u1_geometry_smoke_checkpoint_reload_and_replay(tmp_path: Path) -> None:
    result = run_training("configs/train/osfm/u1_geometry_smoke.yaml", runs_root=tmp_path)
    assert result["experiment_id"] == "OSFM-U1-GEOMETRY-SMOKE-001"
    assert result["architecture"]["encoder"] == "GeometryGroupedEncoder"
    assert result["architecture"]["internal_width"] == 256
    assert result["architecture"]["output_dim"] == 384
    assert result["architecture"]["layers"] == 6
    assert result["architecture"]["heads"] == 4
    assert result["architecture"]["smoke_num_groups"] == 8
    assert result["architecture"]["smoke_group_size"] == 16
    assert result["data"]["public_real_dataset_used"] is False
    assert result["data"]["metric_objective_eligible"] is True
    assert result["reload"]["matches"] is True
    assert result["replay"]["deterministic"] is True
    assert result["metrics"]["group_mask_fraction_requested"] == 0.5
    assert result["metrics"]["group_mask_fraction_actual"] == 0.5
    assert result["metrics"]["validity_coverage"] < 1.0
    metric = result["objectives"]["u1_geometry_metric_reconstruction"]
    assert metric["u1_geometry_metric_reconstruction/status"] == "ACTIVE"
    assert result["representation_health"]["finite"] is True
    assert result["representation_health"]["collapse_score"] > 0
    assert result["representation_health"]["formal_rank_guard"] == "NOT_EVALUABLE"
    assert Path(result["checkpoint"]).is_file()
