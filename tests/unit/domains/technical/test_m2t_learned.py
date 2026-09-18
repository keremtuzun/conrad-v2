from __future__ import annotations

import numpy as np
import pytest
import torch
from m2t_helpers import DAY, context, evidence, registry_ids

from conrad.domains.technical import CORROSION_DEPTH, AssetRegistry, LearnedTCDPConfig, PropagationMode
from conrad.domains.technical.baselines import (
    EngineEstimator,
    GRUTemporal,
    LatestObservation,
    Model2TEstimator,
    SingleFrame,
)
from conrad.domains.technical.learned_tcdp import GraphBatch, LearnedTCDP, tcdp_loss, train_learned_tcdp
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp

TINY = LearnedTCDPConfig(
    d_node_in=5, d_z=20, d_r=3, d_mech=4, d_hidden=16, d_gate=8, d_head=8, partitions=(4, 4, 4, 4, 4)
)


def _batch(seed: int, n: int = 6) -> GraphBatch:
    g = torch.Generator().manual_seed(seed)
    src = torch.tensor([0, 1, 2, 3, 4, 1])
    dst = torch.tensor([1, 2, 3, 4, 5, 0])
    e = src.numel()
    target = torch.rand(n, 2, generator=g) * 3.0
    observed = torch.tensor([True, False, True, False, False, False])
    feats = torch.cat(
        [target * observed.unsqueeze(-1), observed.float().unsqueeze(-1), torch.rand(n, 2, generator=g)], -1
    )
    return GraphBatch(
        node_features=feats,
        edge_index=torch.stack([src, dst]),
        edge_attr=torch.rand(e, 3, generator=g),
        mechanism=torch.randint(0, 6, (e,), generator=g),
        reliability=torch.rand(e, generator=g) * 0.9 + 0.1,
        delta_t=torch.rand(e, generator=g),
        observed=observed,
        target=target,
        target_mask=torch.ones(n, 2, dtype=torch.bool),
        misleading=torch.tensor([False, False, False, True, False, False]),
    )


def test_learned_tcdp_forward_backward_and_constraints():
    torch.manual_seed(0)
    model = LearnedTCDP(TINY)
    assert len(model.layers) == 3
    b = _batch(1)
    out = model(b)
    assert out.mean.shape == (6, 2) and out.z.shape == (6, 20) and len(out.gates) == 3
    loss = tcdp_loss(out, b)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    model.eval()
    z0 = model.encoder(b.node_features)
    z = model(b).z
    frozen = ~model.layers[0].mutable
    assert torch.allclose(z[:, frozen], z0[:, frozen])  # material / geometry never rewritten
    assert torch.allclose(z[b.observed], z0[b.observed])  # observed nodes never updated


def test_learned_tcdp_training_reduces_loss():
    model = LearnedTCDP(TINY)
    torch.manual_seed(0)
    hist = train_learned_tcdp(model, [_batch(s) for s in range(4)], epochs=25, lr=3e-3, seed=0)
    assert hist[-1] < hist[0]


def test_learned_tcdp_rejects_bad_partitions():
    with pytest.raises(ValueError):
        LearnedTCDP(LearnedTCDPConfig(d_node_in=5, d_z=10, partitions=(4, 4)))


def _run(est, r, ev):
    reg = AssetRegistry.from_context(context(r))
    est.reset(reg, stamp(0.0, "sim"))
    est.step(stamp(DAY, "sim"), [evidence(ev, r["seg_a"], DAY, wall=2e-3, crack=4e-3)])
    est.step(stamp(2 * DAY, "sim"), [])
    return est


def test_baselines_share_one_interface():
    r = registry_ids()
    ev = IdFactory(seed=4)
    latest = _run(LatestObservation(), r, ev)
    frame = _run(SingleFrame(), r, ev)
    indep = _run(EngineEstimator(PropagationMode.NONE), r, ev)
    tcdp = _run(EngineEstimator(PropagationMode.TCDP), r, ev)
    full = _run(Model2TEstimator(), r, ev)
    assert latest.estimate(r["seg_a"], CORROSION_DEPTH)[0] == pytest.approx(2e-3)
    assert frame.estimate(r["seg_a"], CORROSION_DEPTH) is None  # not in the current frame
    assert indep.estimate(r["seg_b"], CORROSION_DEPTH) is None
    assert tcdp.estimate(r["seg_b"], CORROSION_DEPTH) is not None
    assert full.estimate(r["seg_a"], CORROSION_DEPTH) == pytest.approx(
        tcdp.estimate(r["seg_a"], CORROSION_DEPTH)
    )


def test_gru_baseline_fits_and_estimates():
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(8):
        y = np.cumsum(rng.uniform(0, 0.1, size=(6, 2)), axis=0)
        x = np.zeros((6, 6))
        x[:, :2], x[:, 3], x[:, 4], x[:, 5] = y, 1.0, 0.9, 0.1
        seqs.append((x, y, np.ones((6, 2), dtype=bool)))
    gru = GRUTemporal(hidden=8, epochs=40, seed=0)
    losses = gru.fit(seqs)
    assert losses[-1] < losses[0]
    r = registry_ids()
    _run(gru, r, IdFactory(seed=4))
    assert gru.estimate(r["seg_a"], CORROSION_DEPTH) is not None
    assert gru.estimate(r["seg_b"], CORROSION_DEPTH) is None
