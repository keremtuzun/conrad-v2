"""MCBR baselines A-B0..A-B10 (ch16 'MCBR baselines', ch18 'Baselines') behind ``ObservationPlanner``.

All baselines share candidate generation and the feasibility filter (filter-before-rank is a safety
rule, not an MCBR feature); they differ only in the ranking rule. Baselines never refuse to plan
(``value_gate=False``) because a stop condition is part of the MCBR hypothesis under test.
A-B8 (RL active perception) is OPEN: no RL baseline is implemented.
"""

from __future__ import annotations

import numpy as np

from conrad.active.config import MCBRConfig
from conrad.active.gap import KnowledgeGap
from conrad.active.planner import MCBRPlanner, PlanningRequest, ScoredCandidate
from conrad.schemas.ids import IdFactory


def _travel(c: ScoredCandidate) -> float:
    return c.action.expected_cost.travel_m


def _novelty(gap: KnowledgeGap, c: ScoredCandidate, cfg: MCBRConfig) -> float:
    p = c.raw.pose.position_m
    best = 0.0
    for v in gap.prior_views:
        d2 = sum((p[i] - v.position_m[i]) ** 2 for i in range(3))
        best = max(best, float(np.exp(-d2 / (2 * cfg.redundancy_sigma_m**2))))
    return 1.0 - best


def random_view(gap: KnowledgeGap, c: ScoredCandidate, req: PlanningRequest) -> float:
    if req.rng is None:
        raise ValueError("random baseline needs a seeded numpy Generator")
    return float(req.rng.random())


def fixed_pattern(gap: KnowledgeGap, c: ScoredCandidate, req: PlanningRequest) -> float:
    """Fixed inspection route: walks the candidate list in generation order, one stride per view taken."""
    stride = 5
    offset = (len(gap.prior_views) * stride) % 128
    return -float((c.raw.index - offset) % 128)


def make_planners(id_factory: IdFactory, config: MCBRConfig | None = None) -> dict[str, MCBRPlanner]:
    cfg = config or MCBRConfig()

    def coverage(gap: KnowledgeGap, c: ScoredCandidate, req: PlanningRequest) -> float:
        return c.action.predicted_visibility * _novelty(gap, c, cfg)

    def frontier(gap: KnowledgeGap, c: ScoredCandidate, req: PlanningRequest) -> float:
        # nearest candidate that looks at not-yet-viewed space
        return (1.0 if _novelty(gap, c, cfg) > 0.5 else 0.0) * 1000.0 - _travel(c)

    def geometric_nbv(gap: KnowledgeGap, c: ScoredCandidate, req: PlanningRequest) -> float:
        return c.action.predicted_visibility * _novelty(gap, c, cfg) / (1.0 + 0.1 * _travel(c))

    def entropy_nbv(gap: KnowledgeGap, c: ScoredCandidate, req: PlanningRequest) -> float:
        return c.action.predicted_visibility * sum(gap.uncertainty.as_tuple()) * _novelty(gap, c, cfg)

    def standard_eig(gap: KnowledgeGap, c: ScoredCandidate, req: PlanningRequest) -> float:
        return c.gain.total

    def uncertainty_nbv(gap: KnowledgeGap, c: ScoredCandidate, req: PlanningRequest) -> float:
        return c.action.predicted_visibility * max(gap.uncertainty.as_tuple()) - c.cost

    def mcbr_no_mission(gap: KnowledgeGap, c: ScoredCandidate, req: PlanningRequest) -> float:
        return c.gain.total - c.cost

    rules = {
        "A-B0_random": random_view,
        "A-B1_fixed_inspection": fixed_pattern,
        "A-B2_coverage": coverage,
        "A-B3_frontier": frontier,
        "A-B4_geometric_nbv": geometric_nbv,
        "A-B5_entropy_nbv": entropy_nbv,
        "A-B6_standard_eig": standard_eig,
        "A-B7_uncertainty_nbv": uncertainty_nbv,
        "A-B9_mcbr_no_mission": mcbr_no_mission,
    }
    planners = {
        name: MCBRPlanner(id_factory, cfg, rule, name=name, value_gate=False) for name, rule in rules.items()
    }
    planners["A-B10_mcbr_full"] = MCBRPlanner(id_factory, cfg, name="A-B10_mcbr_full")
    return planners
