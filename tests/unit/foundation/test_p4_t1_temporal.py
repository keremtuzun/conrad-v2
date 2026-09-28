from __future__ import annotations

from pathlib import Path

import torch

from conrad.foundation.pretraining.losses import ObjectiveStatus
from conrad.foundation.pretraining.t1_temporal import T1TemporalTrainingBundle, synthetic_t1_fixture
from conrad.foundation.temporal.memory import TemporalMemoryConfig, TemporalMemoryTransformer
from conrad.training.entrypoints import run_training


def test_temporal_memory_contract_and_frozen_architecture() -> None:
    model = TemporalMemoryTransformer(TemporalMemoryConfig())
    fused = torch.randn(4, 10, 384)
    timestamps = torch.tensor([[idx * 0.5 for idx in range(10)]] * 4, dtype=torch.float32)
    valid = torch.ones(4, 10, dtype=torch.bool)
    out = model(fused, timestamps, valid)
    assert model.memory_tokens.shape == (1, 16, 384)
    assert out.window_repr.shape == (4, 10, 384)
    assert out.memory_states.shape == (4, 10, 16, 384)
    assert out.history_lengths[0].tolist() == list(range(1, 11))
    assert out.adjacent_eligible_mask[:, 1:].all()


def test_gap_boundary_short_sequence_and_no_cross_sequence_leakage() -> None:
    fixture = synthetic_t1_fixture({"sequence_windows": 10}, 20260415)
    model = TemporalMemoryTransformer(TemporalMemoryConfig())
    out = model(fixture.fused_global, fixture.timestamps_s, fixture.valid_mask, fixture.boundary_reset_mask)
    assert out.gap_reset_mask[1].nonzero().flatten().tolist() == [3]
    assert out.reset_mask[1, 3]
    assert out.history_lengths[1, 2].item() == 3
    assert out.history_lengths[1, 3].item() == 1
    assert out.reset_mask[2, 5]
    assert out.history_lengths[2, 4].item() == 5
    assert out.history_lengths[2, 5].item() == 1
    assert out.adjacent_eligible_mask[3].sum().item() == 0
    assert not torch.equal(out.window_repr[0], out.window_repr[1])


def test_t1_objectives_backward_optimizer_ema_and_denominators() -> None:
    fixture = synthetic_t1_fixture({"sequence_windows": 10}, 20260415)
    bundle = T1TemporalTrainingBundle(
        TemporalMemoryTransformer(TemporalMemoryConfig()),
        rank_diversity_weight=1.0,
        rank_diversity_target=72.0,
    )
    opt = torch.optim.AdamW((p for p in bundle.parameters() if p.requires_grad), lr=2e-4)
    out = bundle(fixture)
    assert out.rank_diversity_loss.item() >= 0.0
    assert out.rank_entropy.item() > 0.0
    objectives = {result.objective_id: result for result in out.results}
    assert objectives["t1_temp"].status is ObjectiveStatus.ACTIVE
    assert objectives["t1_temp"].denominator.item() == 25.0
    assert objectives["t1_masked_latent_prediction"].status is ObjectiveStatus.ACTIVE
    out.loss.backward()
    assert any(p.grad is not None for p in bundle.temporal.parameters())
    opt.step()
    momentum = bundle.update_teacher_after_optimizer(1)
    assert 0.0 < momentum < 1.0


def test_t1_temp_not_applicable_for_single_window_fixture() -> None:
    fixture = synthetic_t1_fixture({"sequence_windows": 10}, 20260415)
    one_row = type(fixture)(
        fused_global=fixture.fused_global[3:4],
        timestamps_s=fixture.timestamps_s[3:4],
        valid_mask=fixture.valid_mask[3:4],
        boundary_reset_mask=fixture.boundary_reset_mask[3:4],
        sequence_ids=("short-single",),
        structure=(fixture.structure[3],),
        m1_relation_rows=fixture.m1_relation_rows,
        m1_parent_status=fixture.m1_parent_status,
    )
    out = T1TemporalTrainingBundle(TemporalMemoryTransformer(TemporalMemoryConfig()))(one_row)
    objectives = {result.objective_id: result for result in out.results}
    assert objectives["t1_temp"].status is ObjectiveStatus.NOT_APPLICABLE
    assert objectives["t1_temp"].denominator.item() == 0.0


def test_temporal_fixture_deterministic_replay_and_no_truth_fields() -> None:
    a = synthetic_t1_fixture({"sequence_windows": 10}, 20260415)
    b = synthetic_t1_fixture({"sequence_windows": 10}, 20260415)
    assert a.sequence_ids == b.sequence_ids
    assert torch.equal(a.timestamps_s, b.timestamps_s)
    assert torch.equal(a.valid_mask, b.valid_mask)
    assert torch.equal(a.boundary_reset_mask, b.boundary_reset_mask)
    forbidden = "truth"
    assert forbidden not in repr(a).lower()


def test_osfm_t1_temporal_smoke_cli_registered(tmp_path: Path) -> None:
    result = run_training("configs/train/osfm/t1_temporal_smoke.yaml", runs_root=tmp_path)
    assert result["experiment_id"] == "OSFM-T1-TEMPORAL-SMOKE-001"
    assert result["component"] == "foundation.osfm.t1_temporal"
    assert result["architecture"]["memory_tokens"] == [16, 384]
    assert result["architecture"]["transformer_blocks"] == 4
    assert result["fixture"]["gap_reset_mask"][1][3] == 1
    assert result["objectives"]["t1_temp"]["t1_temp/status"] == "ACTIVE"
    assert result["objectives"]["t1_temp"]["t1_temp/denominator"] == 25.0
    assert result["reload"]["matches"] is True
    assert result["replay"]["deterministic"] is True
