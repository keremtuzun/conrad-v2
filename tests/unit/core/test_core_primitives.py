import pytest
import torch
from pydantic import ValidationError

from conrad.core.config import CoreConfig, tiny_config
from conrad.core.primitives import (
    AttentionPooling,
    PreNormTransformerBlock,
    ResidualMLP,
    SetAggregator,
    masked_loss,
    masked_mean,
    masked_softmax,
)


def test_residual_mlp_shapes():
    assert ResidualMLP(8, 16, 8)(torch.randn(3, 8)).shape == (3, 8)
    assert ResidualMLP(8, 16, 4)(torch.randn(2, 5, 8)).shape == (2, 5, 4)


def test_masked_softmax_all_false_row_is_zero_not_nan():
    w = masked_softmax(torch.randn(2, 4), torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool))
    assert torch.isfinite(w).all()
    assert torch.allclose(w[0].sum(), torch.tensor(1.0))
    assert w[0, 2:].eq(0).all() and w[1].eq(0).all()


def test_cc05_all_missing_labels_zero_loss_and_zero_denominator():
    values = torch.randn(5, requires_grad=True)
    loss, den = masked_loss(values * 10, torch.zeros(5, dtype=torch.bool))
    assert float(loss) == 0.0 and float(den) == 0.0
    loss.backward()  # still differentiable, contributes nothing
    assert values.grad is not None and values.grad.abs().sum() == 0


def test_masked_loss_ignores_nan_in_masked_entries():
    loss, den = masked_loss(torch.tensor([1.0, float("nan"), 3.0]), torch.tensor([True, False, True]))
    assert float(loss) == 2.0 and float(den) == 2.0


def test_masked_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        masked_loss(torch.zeros(3), torch.ones(4, dtype=torch.bool))


def test_masked_mean_empty_set_is_zero():
    out = masked_mean(torch.randn(2, 3, 4), torch.zeros(2, 3, dtype=torch.bool))
    assert out.eq(0).all()


def test_prenorm_block_padding_does_not_change_valid_tokens():
    torch.manual_seed(0)
    block = PreNormTransformerBlock(8, 2, 16, 0.0).eval()
    x = torch.randn(1, 3, 8)
    padded = torch.cat([x, torch.randn(1, 2, 8) * 100], dim=1)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    a = block(x)
    b = block(padded, key_padding_mask=mask)
    assert torch.allclose(a, b[:, :3], atol=1e-5)
    assert b[:, 3:].eq(0).all()


def test_pooling_empty_set_pools_to_zero():
    pool = AttentionPooling(8, 2)
    assert pool(torch.randn(2, 3, 8), torch.zeros(2, 3, dtype=torch.bool)).eq(0).all()
    agg = SetAggregator(8, 2)
    assert agg(torch.randn(2, 3, 8), torch.zeros(2, 3, dtype=torch.bool)).eq(0).all()


def test_config_defaults_follow_ch33():
    c = CoreConfig()
    assert (c.evidence_dim, c.belief_dim, c.uncertainty_dim, c.temporal_dim) == (256, 256, 64, 64)
    assert (c.history_dim, c.relation_dim, c.mechanism_dim, c.heads, c.head_dim) == (128, 64, 128, 8, 32)
    assert c.buo_input_dim == 1156
    assert c.association_pair_dim == 1109  # ch33 lists these parts; its stated total (853) is inconsistent
    assert c.association.accept_probability == 0.60 and c.association.accept_margin == 0.15


def test_config_is_frozen_and_validated():
    c = tiny_config()
    with pytest.raises(ValidationError):
        c.belief_dim = 3  # type: ignore[misc]
    with pytest.raises(ValueError):
        CoreConfig(uncertainty_dim=10)
    with pytest.raises(ValueError):
        CoreConfig(evidence_dim=128)
