import torch

from conrad.core.association import (
    AssociationEngine,
    AssociationScorer,
    accept_associations,
    association_loss,
    nearest_neighbour_decision,
    pair_extra_features,
    retrieve_candidates,
)
from conrad.schemas.belief import BeliefCell, KnowledgeStatus, Lifecycle
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.timebase import stamp
from conrad.schemas.uncertainty import unknown_uncertainty
from conrad.schemas.world import Domain


def _cell(ids, cfg, center, lifecycle=Lifecycle.ACTIVE, frame="WORLD", domain=Domain.TECHNICAL):
    return BeliefCell(
        belief_id=ids.new(),
        domain=domain,
        entity_type="component",
        lifecycle=lifecycle,
        revision=0,
        timestamp=stamp(0.0, "sandbox_clock"),
        state_embedding=(0.0,) * cfg.belief_dim,
        knowledge_status=KnowledgeStatus.OBSERVED,
        uncertainty=unknown_uncertainty(),
        spatial_support=SpatialSupport(frame_id=frame, center_m=center, position_sigma_m=0.05),
        provenance_root=ids.new(),
        model_version="t",
    )


def test_retrieval_gates_frame_lifecycle_domain_distance_and_k(ids, cfg, fac):
    ev, _ = fac.make(1.0, {"p": 0.5}, center_m=(0.0, 0.0, 0.0))
    near = _cell(ids, cfg, (0.5, 0, 0))
    cells = [
        near,
        _cell(ids, cfg, (0.1, 0, 0), frame="ODOM"),
        _cell(ids, cfg, (0.1, 0, 0), lifecycle=Lifecycle.RETIRED),
        _cell(ids, cfg, (0.1, 0, 0), domain=Domain.SPATIAL),
        _cell(ids, cfg, (100.0, 0, 0)),
    ]
    got = retrieve_candidates(ev, cells, cfg.association, domain=Domain.TECHNICAL)
    assert [c.belief_id for c in got] == [near.belief_id]
    many = [_cell(ids, cfg, (0.01 * i, 0, 0)) for i in range(50)]
    assert len(retrieve_candidates(ev, many, cfg.association)) == cfg.association.max_candidates


def test_no_match_with_empty_candidate_set(cfg):
    scorer = AssociationScorer(cfg)
    _, probs = scorer(
        torch.randn(2, cfg.evidence_dim),
        torch.zeros(2, 3, cfg.belief_dim),
        torch.zeros(2, 3, scorer.extra_dim),
        torch.zeros(2, 3, dtype=torch.bool),
    )
    assert torch.allclose(probs[:, -1], torch.ones(2)) and probs[:, :3].eq(0).all()
    assert accept_associations(probs, 0.6, 0.15).index.tolist() == [-1, -1]


def test_acceptance_rule_probability_and_margin():
    probs = torch.tensor([[0.70, 0.20, 0.10], [0.55, 0.05, 0.40], [0.62, 0.50, 0.00], [0.10, 0.10, 0.80]])
    res = accept_associations(probs, 0.60, 0.15)
    assert res.index.tolist() == [0, -1, -1, -1]  # last row: NO_MATCH is the top choice


def test_pair_features_width_and_loss(cfg, ids, fac):
    ev, _ = fac.make(1.0, {"p": 0.5})
    extra = pair_extra_features(ev, _cell(ids, cfg, (0.2, 0, 0)), cfg)
    assert extra.shape == (cfg.association_pair_dim - 4 * cfg.evidence_dim,)
    scorer = AssociationScorer(cfg)
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    logits, _ = scorer(
        torch.randn(2, cfg.evidence_dim),
        torch.randn(2, 3, cfg.belief_dim),
        torch.randn(2, 3, scorer.extra_dim),
        mask,
    )
    loss = association_loss(logits, mask, torch.tensor([1, 3]), torch.tensor([True, True]), cfg)
    assert torch.isfinite(loss.total)
    assert loss.denominators == {"cross_entropy": 2.0, "focal_no_match": 1.0, "ranking": 2.0}
    loss.total.backward()
    none = association_loss(logits.detach(), mask, torch.tensor([1, 3]), torch.tensor([False, False]), cfg)
    assert float(none.total) == 0.0 and none.denominators["cross_entropy"] == 0.0


def test_nearest_neighbour_baseline_and_engine(cfg, ids, fac):
    ev, _ = fac.make(1.0, {"p": 0.5}, center_m=(0.0, 0.0, 0.0))
    near, far = _cell(ids, cfg, (0.3, 0, 0)), _cell(ids, cfg, (3.0, 0, 0))
    assert nearest_neighbour_decision(ev, [far, near], cfg.association).belief_id == near.belief_id
    assert nearest_neighbour_decision(ev, [far], cfg.association).belief_id is None
    learned = AssociationEngine(cfg, AssociationScorer(cfg)).associate(ev, [near, far])
    assert learned.method == "learned_scorer" and 0 <= learned.probability <= 1
