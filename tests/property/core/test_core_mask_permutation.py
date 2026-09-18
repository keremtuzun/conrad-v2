"""CC-06 and mask properties: padding / permutation leave set aggregation and valid-sample loss unchanged."""

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from conrad.core.buo import AnalyticBUO, BeliefUpdateOperator
from conrad.core.config import tiny_config
from conrad.core.primitives import SetAggregator, masked_loss, masked_mean, masked_softmax
from conrad.core.rbp import RelationalBeliefPropagation
from conrad.core.state import unknown_state
from conrad.evaluation.core_experiments.evidence_factory import EvidenceFactory
from conrad.schemas.ids import IdFactory

CFG = tiny_config()
torch.manual_seed(0)
AGG = SetAggregator(CFG.evidence_dim, CFG.heads).eval()
BUO = BeliefUpdateOperator(CFG).eval()
RBP = RelationalBeliefPropagation(CFG).eval()
SETTINGS = settings(max_examples=25, deadline=None)


@SETTINGS
@given(st.integers(1, 6), st.integers(0, 5), st.randoms(use_true_random=False))
def test_set_aggregation_permutation_and_padding_invariant(k, pad, rnd):
    g = torch.Generator().manual_seed(rnd.randint(0, 10**6))
    items = torch.randn(1, k, CFG.evidence_dim, generator=g)
    base = AGG(items, torch.ones(1, k, dtype=torch.bool))
    perm = torch.randperm(k, generator=g)
    padded = torch.cat([items[:, perm], torch.randn(1, pad, CFG.evidence_dim, generator=g) * 100], 1)
    mask = torch.cat([torch.ones(1, k), torch.zeros(1, pad)], 1).bool()
    assert torch.allclose(base, AGG(padded, mask), atol=1e-5)


@SETTINGS
@given(st.integers(1, 8), st.integers(0, 5), st.randoms(use_true_random=False))
def test_masked_loss_and_mean_padding_invariant(n, pad, rnd):
    g = torch.Generator().manual_seed(rnd.randint(0, 10**6))
    vals = torch.rand(n, generator=g)
    loss, den = masked_loss(vals, torch.ones(n, dtype=torch.bool))
    padded = torch.cat([vals, torch.full((pad,), 1e6)])
    loss2, den2 = masked_loss(padded, torch.cat([torch.ones(n), torch.zeros(pad)]).bool())
    assert torch.allclose(loss, loss2) and float(den) == float(den2) == n
    x = torch.randn(1, n, 3, generator=g)
    xp = torch.cat([x, torch.randn(1, pad, 3, generator=g) * 1e3], 1)
    mp = torch.cat([torch.ones(1, n), torch.zeros(1, pad)], 1).bool()
    assert torch.allclose(masked_mean(x, torch.ones(1, n, dtype=torch.bool)), masked_mean(xp, mp), atol=1e-5)


@SETTINGS
@given(st.lists(st.booleans(), min_size=1, max_size=8), st.randoms(use_true_random=False))
def test_masked_softmax_support(mask_list, rnd):
    g = torch.Generator().manual_seed(rnd.randint(0, 10**6))
    mask = torch.tensor(mask_list)
    w = masked_softmax(torch.randn(len(mask_list), generator=g) * 50, mask)
    assert w[~mask].eq(0).all()
    assert abs(float(w.sum()) - (1.0 if mask.any() else 0.0)) < 1e-5


@SETTINGS
@given(st.integers(1, 5), st.randoms(use_true_random=False))
def test_learned_buo_permutation_invariant(k, rnd):
    g = torch.Generator().manual_seed(rnd.randint(0, 10**6))
    z, u, t = (
        torch.randn(1, d, generator=g) for d in (CFG.belief_dim, CFG.uncertainty_dim, CFG.temporal_dim)
    )
    e, q = torch.randn(1, k, CFG.evidence_dim, generator=g), torch.rand(1, k, 4, generator=g)
    perm = torch.randperm(k, generator=g)
    m = torch.ones(1, k, dtype=torch.bool)
    a, b = BUO(z, u, t, e, q, m), BUO(z, u, t, e[:, perm], q[:, perm], m)
    assert torch.allclose(a.z, b.z, atol=1e-5) and torch.allclose(a.channels, b.channels, atol=1e-5)


@SETTINGS
@given(st.integers(2, 6), st.integers(1, 8), st.randoms(use_true_random=False))
def test_rbp_edge_order_invariant(n, m, rnd):
    g = torch.Generator().manual_seed(rnd.randint(0, 10**6))
    z, u = torch.randn(n, CFG.belief_dim, generator=g), torch.randn(n, CFG.uncertainty_dim, generator=g)
    ei = torch.randint(0, n, (2, m), generator=g)
    et = torch.randint(0, CFG.rbp.relation_types, (m,), generator=g)
    en = torch.rand(m, CFG.rbp.relation_numeric_dim, generator=g)
    perm = torch.randperm(m, generator=g)
    a, b = RBP(z, u, ei, et, en), RBP(z, u, ei[:, perm], et[perm], en[perm])
    assert torch.allclose(a.z, b.z, atol=1e-4)


@SETTINGS
@given(st.lists(st.floats(0.0, 1.0), min_size=1, max_size=5), st.randoms(use_true_random=False))
def test_analytic_buo_evidence_order_invariant(values, rnd):
    fac = EvidenceFactory(IdFactory(rnd.randint(0, 10**6)), 4, np.random.default_rng(0))
    evs = [fac.make(1.0, {"p": v}, independence_group=f"g{i}")[0] for i, v in enumerate(values)]
    order = list(range(len(evs)))
    rnd.shuffle(order)
    a = AnalyticBUO().update(unknown_state(), evs).state
    b = AnalyticBUO().update(unknown_state(), [evs[i] for i in order]).state
    assert a.estimates == b.estimates and a.channels == b.channels
