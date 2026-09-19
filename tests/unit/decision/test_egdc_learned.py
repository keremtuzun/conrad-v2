"""ch33 learned EGDC scorer: shapes, masking, forward/backward, imitation training, policy interface."""

import torch

from conrad.decision import EGDC
from conrad.decision.learned import (
    EGDCScorerConfig,
    LearnedEGDCPolicy,
    build_scorer,
    collate,
    egdc_loss,
    featurize,
    train_imitation,
)
from conrad.evaluation.decision_experiments.fixtures import make_belief, make_context, make_requirement, unc
from conrad.schemas.decision import ActionType
from conrad.schemas.ids import IdFactory

TINY = EGDCScorerConfig(
    claim_embedding_dim=24,
    uncertainty_dim=8,
    provenance_dim=8,
    context_dim=8,
    input_hidden_dim=32,
    model_dim=16,
    num_layers=2,
    num_heads=2,
    action_dim=8,
    resource_dim=8,
    risk_dim=8,
    score_hidden=(16, 8),
    max_nodes=32,
)


def _cycle(seed, u):
    ids = IdFactory(seed)
    b = make_belief(ids, uncertainty=u, n_evidence=0 if seed % 2 else 2)
    ctx = make_context(ids, [b], [make_requirement(ids, belief_ids=[b.belief_id])])
    egdc = EGDC(ids)
    graph = egdc.builder.build(ctx)
    cands = egdc.generator.generate(graph, ctx)
    egdc.builder.attach_actions(graph, ctx, cands)
    cons = [egdc.estimator.estimate(a, graph, ctx) for a in cands]
    return egdc, graph, cands, cons, ctx


def test_forward_backward_and_masks():
    _, graph, cands, cons, ctx = _cycle(1, unc(uo=0.9))
    batch = featurize(graph, cands, cons, ctx, TINY)
    assert batch.nodes.shape == (1, TINY.max_nodes, TINY.node_input_dim)
    model = build_scorer(TINY)
    out = model(batch)
    assert out.scores.shape == (1, len(cands))
    valid = batch.node_mask[0]
    assert torch.allclose(out.claim_support[0][:, valid].sum(-1), torch.ones(len(cands)), atol=1e-5)
    assert (out.claim_support[0][:, ~valid] == 0).all()
    batch.target_action = torch.tensor([0])
    loss, parts = egdc_loss(out, batch, TINY)
    loss.backward()
    assert set(parts) == {"action", "support", "outcome", "uir"}
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_imitation_training_reduces_loss_and_policy_only_ranks():
    batches = []
    for seed, u in [(2, unc(uo=0.9)), (3, unc()), (4, unc(ua=0.9)), (5, unc(uc=0.9))]:
        _, graph, cands, cons, ctx = _cycle(seed, u)
        b = featurize(graph, cands, cons, ctx, TINY)
        target = next(i for i, a in enumerate(cands) if a.action_type is ActionType.ESCALATE_TO_OPERATOR)
        b.target_action = torch.tensor([target])
        batches.append(b)
    model = build_scorer(TINY)
    hist = train_imitation(model, [collate(batches)], TINY, epochs=20)
    assert hist[-1]["total"] < hist[0]["total"]
    ids = IdFactory(9)
    belief = make_belief(ids, uncertainty=unc(uo=0.9))
    ctx = make_context(ids, [belief], [make_requirement(ids, belief_ids=[belief.belief_id])], battery=0.05)
    out = EGDC(ids, policy=LearnedEGDCPolicy(model)).decide(ctx)
    # whatever the learned scores say, the deterministic constraint engine still decides permission
    assert out.record.chosen is None or out.record.constraint_decisions[-1].accepted
    assert all(not d.accepted for d in out.record.constraint_decisions[:-1])
    motion = {ActionType.CONTINUE_MISSION, ActionType.REQUEST_INFORMATION, ActionType.REVISIT_REGION}
    assert out.record.chosen is None or out.record.chosen.action_type not in motion  # battery below reserve
