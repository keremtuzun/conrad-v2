from __future__ import annotations

import torch

from conrad.foundation.context import (
    ContextEncoder,
    ContextObservation,
    collate_context_observations,
    direct_model2_evidence,
    fit_context_normalization,
)
from conrad.foundation.fusion.scene_fusion import ModalityTokenSet, SceneFusionConfig, SceneFusionTransformer


def _rows() -> tuple[tuple[ContextObservation, ...], ...]:
    return (
        (
            ContextObservation("temperature_c", (12.8,), True, 1000, "degC", "sensor:env:train", uncertainty=(0.1,)),
            ContextObservation("vehicle_depth_m", (8.0,), True, 1000, "m", "sensor:nav:train", uncertainty=(0.2,)),
        ),
        (
            ContextObservation("temperature_c", (99.0,), False, 1500, "degC", "sensor:env:missing"),
            ContextObservation("vehicle_depth_m", (9.0,), True, 1500, "m", "sensor:nav:val", uncertainty=(0.2,)),
        ),
    )


def test_context_contract_missing_is_not_zero_filled_as_measured() -> None:
    stats = fit_context_normalization((_rows()[0],), fit_partitions=("PRETRAIN_REAL",))
    batch = collate_context_observations(_rows(), stats=stats)
    temp_idx = batch.field_names.index("temperature_c")
    assert batch.present_mask[0, temp_idx]
    assert not batch.present_mask[1, temp_idx]
    assert batch.values[1, temp_idx].abs().sum() == 0.0
    assert batch.provenance[1][temp_idx] == "sensor:env:missing"
    assert batch.units[temp_idx] == "degC"


def test_context_normalization_uses_declared_training_rows_only() -> None:
    stats = fit_context_normalization((_rows()[0],), fit_partitions=("PRETRAIN_REAL",))
    assert stats.fit_partitions == ("PRETRAIN_REAL",)
    assert stats.means["temperature_c"] == (12.800000190734863,)
    assert stats.means["vehicle_depth_m"] == (8.0,)
    assert stats.means["vehicle_depth_m"] != (8.5,)


def test_context_encoder_forward_backward_reload_and_replay() -> None:
    torch.manual_seed(17)
    stats = fit_context_normalization((_rows()[0],), fit_partitions=("PRETRAIN_REAL",))
    batch = collate_context_observations(_rows(), stats=stats)
    encoder = ContextEncoder()
    out = encoder(batch)
    assert list(out.tokens.shape) == [2, len(batch.field_names), 384]
    assert list(out.pooled.shape) == [2, 384]
    assert torch.equal(out.available_mask, batch.present_mask)
    loss = out.pooled.square().mean()
    loss.backward()
    assert any(param.grad is not None and float(param.grad.abs().sum()) > 0.0 for param in encoder.parameters())

    fresh = ContextEncoder()
    fresh.load_state_dict(encoder.state_dict())
    with torch.no_grad():
        replay = fresh(batch)
    assert torch.allclose(out.tokens.detach(), replay.tokens)


def test_scene_fusion_accepts_context_and_context_free_paths() -> None:
    model = SceneFusionTransformer(SceneFusionConfig())
    rgb = ModalityTokenSet(
        "rgb",
        torch.randn(2, 3, 384),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    ctx = ModalityTokenSet(
        "context",
        torch.randn(2, 5, 384),
        torch.tensor([[True, True, False, False, False], [False, False, False, False, False]]),
        torch.tensor([False, True]),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    assert model((rgb,)).global_repr.shape == (2, 384)
    out = model((rgb, ctx))
    assert out.modality_names == ("rgb", "context")
    assert out.modality_presence[:, 0].all()
    assert out.modality_presence[0, 1]
    assert not out.modality_presence[1, 1]


def test_model2_direct_evidence_keeps_exact_parallel_measurement_path() -> None:
    stats = fit_context_normalization((_rows()[0],), fit_partitions=("PRETRAIN_REAL",))
    batch = collate_context_observations(_rows(), stats=stats)
    evidence = direct_model2_evidence(batch)
    by_field = {item["field"]: item for item in evidence[0]["model2_direct_measurements"]}
    assert by_field["temperature_c"]["units"] == "degC"
    assert by_field["temperature_c"]["provenance"] == "sensor:env:train"
    assert all(item["field"] != "temperature_c" for item in evidence[1]["model2_direct_measurements"])
