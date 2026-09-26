from __future__ import annotations

from pathlib import Path

import torch

from conrad.foundation.encoders.sonar import SonarEncoderConfig, SonarViTS14Encoder
from conrad.foundation.pretraining.losses import ObjectiveStatus
from conrad.foundation.pretraining.u1_rgb import contiguous_2d_mask, parameter_count
import pytest

from conrad.foundation.data.manifest import CorpusPartition
from conrad.foundation.pretraining import u1_sonar
from conrad.foundation.pretraining.u1_sonar import (
    U1SonarTrainingBundle,
    _require_u1_sonar_research_gate,
    real_subpipe_sonar_views,
    synthetic_sonar_views,
)
from conrad.training.entrypoints import ComputeBlockedError, run_training


def test_sonar_vit_s14_single_channel_contract_and_dense_taps() -> None:
    torch.manual_seed(1)
    encoder = SonarViTS14Encoder(SonarEncoderConfig(image_size=28))
    out = encoder(torch.rand(2, 1, 28, 28), torch.zeros(2, 4, dtype=torch.bool))
    assert out.modality_repr.shape == (2, 384)
    assert out.patch_tokens.shape == (2, 4, 384)
    assert set(out.dense_taps) == {3, 6, 9, 12}
    assert all(t.shape == (2, 4, 384) for t in out.dense_taps.values())
    assert encoder.config.in_channels == 1
    assert encoder.config.patch_size == 14
    assert encoder.config.depth == 12
    assert encoder.config.heads == 6
    assert 20_000_000 <= parameter_count(encoder) <= 23_000_000


def test_sonar_60_percent_contiguous_mask_is_deterministic_and_visible() -> None:
    a = contiguous_2d_mask(
        batch_size=2,
        grid_size=2,
        mask_fraction=0.60,
        min_visible_fraction=0.10,
        generator=torch.Generator().manual_seed(11),
    )
    b = contiguous_2d_mask(
        batch_size=2,
        grid_size=2,
        mask_fraction=0.60,
        min_visible_fraction=0.10,
        generator=torch.Generator().manual_seed(11),
    )
    assert torch.equal(a, b)
    assert a.float().mean().item() == 0.5
    assert (~a).float().mean().item() >= 0.10


def test_sonar_views_record_permitted_rendered_image_corruptions_and_no_truth_leakage() -> None:
    views = synthetic_sonar_views({"batch_size": 2, "image_size": 28, "token_mask_fraction": 0.60}, 20260411)
    assert views.teacher_sonar.shape == (2, 1, 28, 28)
    assert views.student_sonar.shape == (2, 1, 28, 28)
    assert all("truth" not in sample_id.lower() for sample_id in views.sample_ids)
    assert set(views.corruption_trace) == {
        "bounded_speckle_like_noise",
        "gain_variation",
        "attenuation_like_rolloff",
        "dropout",
    }
    assert views.metric_targets is None


def test_u1_sonar_objectives_backward_ema_and_metric_routing() -> None:
    views = synthetic_sonar_views({"batch_size": 2, "image_size": 28, "token_mask_fraction": 0.60}, 20260411)
    bundle = U1SonarTrainingBundle(SonarViTS14Encoder(SonarEncoderConfig(image_size=28)))
    assert all(not p.requires_grad for p in bundle.teacher.parameters())
    optimizer = torch.optim.AdamW(bundle.parameters(), lr=1e-4)
    before = [p.detach().clone() for p in bundle.teacher.parameters()]
    out = bundle(views)
    assert torch.isfinite(out.loss)
    statuses = {r.objective_id: r.status for r in out.results}
    assert statuses["u1_sonar_masked_latent_prediction"] is ObjectiveStatus.ACTIVE
    assert statuses["u1_sonar_global_consistency"] is ObjectiveStatus.ACTIVE
    assert statuses["u1_sonar_degradation"] is ObjectiveStatus.ACTIVE
    assert statuses["u1_sonar_metric"] is ObjectiveStatus.NOT_APPLICABLE
    out.loss.backward()
    assert any(p.grad is not None for p in bundle.student.parameters())
    optimizer.step()
    tau = bundle.update_teacher_after_optimizer(1)
    assert tau > 0.996
    assert any(not torch.equal(a, b) for a, b in zip(before, bundle.teacher.parameters(), strict=True))
    assert not bundle.teacher.training


def test_osfm_u1_sonar_smoke_checkpoint_reload_and_replay(tmp_path: Path) -> None:
    result = run_training("configs/train/osfm/u1_sonar_smoke.yaml", runs_root=tmp_path)
    assert result["experiment_id"] == "OSFM-U1-SONAR-SMOKE-001"
    assert result["architecture"]["encoder"] == "SonarViTS14Encoder"
    assert result["architecture"]["layers"] == 12
    assert result["architecture"]["heads"] == 6
    assert result["architecture"]["width"] == 384
    assert result["data"]["subpipe_used"] is False
    assert result["reload"]["matches"] is True
    assert result["replay"]["deterministic"] is True
    assert result["representation_health"]["finite"] is True
    assert result["representation_health"]["collapse_score"] > 0
    assert result["representation_health"]["formal_rank_guard"] == "NOT_EVALUABLE"
    assert result["objectives"]["u1_sonar_metric"]["u1_sonar_metric/status"] == "NOT_APPLICABLE"
    assert Path(result["checkpoint"]).is_file()


def test_p48a_research_config_is_registered_but_cpu_blocked(tmp_path: Path) -> None:
    with pytest.raises(ComputeBlockedError, match="full-scale run requires"):
        run_training("configs/train/osfm/research/u1_sonar_rehearsal.yaml", runs_root=tmp_path)


def test_p48a_gate_requires_explicit_formal_allow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(u1_sonar, "inspect_compute", lambda: type("C", (), {"cuda_available": True, "model_dump": lambda self, mode='json': {}})())
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _i: type("P", (), {"total_memory": 24 * 1024**3})())
    with pytest.raises(RuntimeError, match="not formally allowed"):
        _require_u1_sonar_research_gate({"formal_run_allowed": False})


def test_real_subpipe_views_refuse_final_and_ood_partitions() -> None:
    job = {"batch_size": 2, "image_size": 28}
    for partition in (CorpusPartition.FINAL_TEST, CorpusPartition.OOD_TEST):
        with pytest.raises(ValueError, match="PRETRAIN_REAL or VALIDATION"):
            real_subpipe_sonar_views(job, 1, partition=partition)


def test_p48a_rehearsal_config_preserves_partition_policy() -> None:
    import yaml

    cfg = yaml.safe_load(Path("configs/train/osfm/research/u1_sonar_rehearsal.yaml").read_text())
    assert cfg["formal_run_allowed"] is True
    assert cfg["rehearsal_only"] is True
    assert cfg["partition_access"]["train"] == ["PRETRAIN_REAL"]
    assert cfg["partition_access"]["validation"] == ["VALIDATION"]
    assert set(cfg["partition_access"]["forbidden_training_access"]) == {"FINAL_TEST", "OOD_TEST"}
    assert "rendered side-scan imagery" in " ".join(cfg["notes"])
