"""Learned BUO / RBP / TBD: shapes, forward+backward smoke, masks, trust != innovation, delta_t."""

import pytest
import torch

from conrad.core.buo import BeliefUpdateOperator, collapse_independence_groups, reject_duplicates
from conrad.core.config import tiny_config
from conrad.core.rbp import RelationalBeliefPropagation, contamination_loss, segment_softmax
from conrad.core.tbd import (
    GruTBD,
    HoldLastTBD,
    LinearExtrapolationTBD,
    MlpTBD,
    build_tbd,
    latent_temporal_surprise,
)


def _buo_inputs(cfg, b=3, k=4):
    return (
        torch.randn(b, cfg.belief_dim),
        torch.randn(b, cfg.uncertainty_dim),
        torch.randn(b, cfg.temporal_dim),
        torch.randn(b, k, cfg.evidence_dim),
        torch.rand(b, k, cfg.buo.quality_dim),
    )


def test_buo_shapes_and_backward(cfg):
    buo = BeliefUpdateOperator(cfg)
    z, u, t, e, q = _buo_inputs(cfg)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    out = buo(z, u, t, e, q, mask)
    assert out.z.shape == z.shape and out.u.shape == u.shape and out.channels.shape == (3, 4)
    assert out.trust.shape == (3, 1) and out.gate.shape == z.shape  # separate heads
    assert (out.channels >= 0).all()
    assert out.updated.tolist() == [True, False, True]
    assert torch.equal(out.z[1], z[1]) and torch.equal(out.u[1], u[1])  # no evidence -> no update
    (out.z.sum() + out.channels.sum() + out.pair_compatibility.sum()).backward()
    assert all(p.grad is not None for p in buo.trust.parameters())


def test_buo_padding_invariance(cfg):
    buo = BeliefUpdateOperator(cfg).eval()
    z, u, t, e, q = _buo_inputs(cfg, b=1, k=2)
    a = buo(z, u, t, e, q, torch.ones(1, 2, dtype=torch.bool))
    e_pad = torch.cat([e, torch.randn(1, 3, cfg.evidence_dim) * 50], 1)
    q_pad = torch.cat([q, torch.rand(1, 3, 4)], 1)
    b = buo(z, u, t, e_pad, q_pad, torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool))
    assert torch.allclose(a.z, b.z, atol=1e-5) and torch.allclose(a.u, b.u, atol=1e-5)


def test_buo_same_group_repeats_equal_single_item(cfg):
    buo = BeliefUpdateOperator(cfg).eval()
    z, u, t, e, q = _buo_inputs(cfg, b=1, k=1)
    single = buo(z, u, t, e, q, torch.ones(1, 1, dtype=torch.bool), torch.tensor([[7]]))
    rep = buo(
        z,
        u,
        t,
        e.repeat(1, 5, 1),
        q.repeat(1, 5, 1),
        torch.ones(1, 5, dtype=torch.bool),
        torch.full((1, 5), 7),
    )
    assert torch.allclose(single.z, rep.z, atol=1e-5)


def test_reject_duplicates_before_block():
    keys = torch.tensor([[1, 2, 2, 3]])
    mask = reject_duplicates(keys, torch.tensor([[3]]), torch.ones(1, 4, dtype=torch.bool))
    assert mask.tolist() == [[True, True, False, False]]
    e, _, first = collapse_independence_groups(
        torch.randn(1, 3, 4),
        torch.rand(1, 3, 4),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[5, 5, 6]]),
    )
    assert first.tolist() == [[True, False, True]] and torch.allclose(e[0, 0], e[0, 1])


@pytest.mark.parametrize(
    "flag", ["use_trust", "use_innovation_gate", "use_contradiction", "collapse_trust_innovation"]
)
def test_buo_ablation_switches_run(flag):
    base = tiny_config()
    value = not getattr(base.buo, flag)
    cfg = base.model_copy(update={"buo": base.buo.model_copy(update={flag: value})})
    out = BeliefUpdateOperator(cfg)(*_buo_inputs(cfg), torch.ones(3, 4, dtype=torch.bool))
    assert torch.isfinite(out.z).all()


def test_rbp_shapes_backward_and_isolated_nodes(cfg):
    rbp = RelationalBeliefPropagation(cfg)
    z = torch.randn(5, cfg.belief_dim, requires_grad=True)
    u = torch.randn(5, cfg.uncertainty_dim)
    ei = torch.tensor([[0, 1, 3], [1, 2, 2]])
    out = rbp(z, u, ei, torch.tensor([0, 1, 1]), torch.rand(3, cfg.rbp.relation_numeric_dim))
    assert out.z.shape == z.shape and out.layers_run == cfg.rbp.layers
    assert out.updated.tolist() == [False, True, True, False, False]
    assert torch.equal(out.z[0], z[0]) and torch.equal(out.z[4], z[4])
    out.z.sum().backward()
    assert z.grad is not None


def test_rbp_masked_edge_and_absent_node_contribute_nothing(cfg):
    rbp = RelationalBeliefPropagation(cfg).eval()
    z, u = torch.randn(3, cfg.belief_dim), torch.randn(3, cfg.uncertainty_dim)
    ei, et, en = (
        torch.tensor([[0, 1], [2, 2]]),
        torch.tensor([0, 0]),
        torch.rand(2, cfg.rbp.relation_numeric_dim),
    )
    only0 = rbp(z, u, ei[:, :1], et[:1], en[:1])
    masked = rbp(z, u, ei, et, en, edge_mask=torch.tensor([True, False]))
    absent = rbp(z, u, ei, et, en, node_mask=torch.tensor([True, False, True]))
    assert torch.allclose(only0.z, masked.z, atol=1e-6) and torch.allclose(only0.z, absent.z, atol=1e-6)


def test_rbp_depth_and_untyped_ablation(cfg):
    untyped = cfg.model_copy(update={"rbp": cfg.rbp.model_copy(update={"typed": False})})
    rbp = RelationalBeliefPropagation(untyped)
    out = rbp(
        torch.randn(2, cfg.belief_dim),
        torch.randn(2, cfg.uncertainty_dim),
        torch.tensor([[0], [1]]),
        torch.tensor([1]),
        torch.rand(1, cfg.rbp.relation_numeric_dim),
        depth=1,
    )
    assert out.layers_run == 1
    with pytest.raises(ValueError):
        rbp(
            torch.randn(2, cfg.belief_dim),
            torch.randn(2, cfg.uncertainty_dim),
            torch.tensor([[0], [1]]),
            torch.tensor([1]),
            torch.rand(1, cfg.rbp.relation_numeric_dim),
            depth=9,
        )


def test_segment_softmax_and_contamination_loss():
    w = segment_softmax(
        torch.randn(4, 2), torch.tensor([0, 0, 1, 1]), 2, torch.tensor([True, True, True, False])
    )
    assert torch.allclose(w[:2].sum(0), torch.ones(2)) and w[3].eq(0).all()
    loss, den = contamination_loss(
        torch.tensor([0.1, 0.1]), torch.tensor([0.3, 0.0]), torch.tensor([True, False])
    )
    assert float(den) == 1.0 and abs(float(loss) - 0.2) < 1e-6


@pytest.mark.parametrize("kind", ["mlp", "gru"])
def test_learned_tbd_uses_physical_delta_t(cfg, kind):
    tbd = build_tbd(cfg, kind).eval()
    z, u, t = (
        torch.randn(2, cfg.belief_dim),
        torch.randn(2, cfg.uncertainty_dim),
        torch.randn(2, cfg.temporal_dim),
    )
    a = tbd(z, u, t, torch.tensor([5.0, 5.0]))
    b = tbd(z, u, t, torch.tensor([10.0, 10.0]))
    again = tbd(z, u, t, torch.tensor([5.0, 5.0]))
    assert not torch.allclose(a.z, b.z)  # doubling delta_t changes the prediction
    assert torch.equal(a.z, again.z)  # nothing else (e.g. call count / sequence index) enters
    assert a.z.shape == z.shape and a.u.shape == u.shape and a.temporal.shape == t.shape


def test_tbd_backward_and_distinct_prediction(cfg):
    tbd = GruTBD(cfg)
    z = torch.randn(3, cfg.belief_dim, requires_grad=True)
    pred = tbd(z, torch.zeros(3, cfg.uncertainty_dim), torch.zeros(3, cfg.temporal_dim), torch.ones(3))
    assert pred.z is not z
    pred.z.sum().backward()
    assert z.grad is not None
    assert latent_temporal_surprise(pred.z.detach(), z.detach()).shape == (3,)
    assert isinstance(MlpTBD(cfg)(z, torch.zeros(3, 8), torch.zeros(3, 8), torch.ones(3)).z, torch.Tensor)


def test_tbd_baselines_and_bad_delta_t(cfg):
    z = torch.randn(2, cfg.belief_dim)
    u, t = torch.zeros(2, cfg.uncertainty_dim), torch.zeros(2, cfg.temporal_dim)
    assert torch.equal(HoldLastTBD(cfg)(z, u, t, torch.ones(2)).z, z)
    lin = LinearExtrapolationTBD(cfg)(
        z, u, t, torch.tensor([2.0, 2.0]), z_prev=z - 1, delta_t_prev_s=torch.ones(2)
    )
    assert torch.allclose(lin.z, z + 2)
    with pytest.raises(ValueError):
        HoldLastTBD(cfg)(z, u, t, torch.tensor([-1.0, 1.0]))
    with pytest.raises(ValueError):
        build_tbd(cfg, "neural_ode")
