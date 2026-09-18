"""Association: deterministic candidate retrieval, pair features, NO_MATCH decisions (ch3, ch9, ch33).

Retrieval never uses truth identifiers. Identity hints come only from ``Evidence.entity_candidates``
(belief-plane hints or a mission-supplied asset registry).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

import torch
from torch import Tensor

from conrad.core.association_scorer import (
    AcceptanceResult,
    AssociationLoss,
    AssociationScorer,
    accept_associations,
    association_loss,
)
from conrad.core.config import AssociationConfig, CoreConfig
from conrad.schemas.belief import BeliefCell, Lifecycle
from conrad.schemas.observation import Evidence
from conrad.schemas.world import Domain

__all__ = [
    "AcceptanceResult",
    "AssociationDecision",
    "AssociationEngine",
    "AssociationLoss",
    "AssociationScorer",
    "accept_associations",
    "association_loss",
    "nearest_neighbour_decision",
    "pair_extra_features",
    "retrieve_candidates",
]

TERMINAL = frozenset({Lifecycle.MERGED, Lifecycle.SPLIT, Lifecycle.RETIRED, Lifecycle.REJECTED})


@dataclass(frozen=True)
class AssociationDecision:
    evidence_id: UUID
    belief_id: UUID | None  # None = NO_MATCH (new/tentative entity)
    probability: float
    margin: float
    candidate_ids: tuple[UUID, ...]
    method: str


def _distance(ev: Evidence, cell: BeliefCell) -> float | None:
    """Centre distance in metres, or None when either side has no support or the frames differ."""
    a, b = ev.spatial_support, cell.spatial_support
    if a is None or b is None or a.frame_id != b.frame_id:
        return None
    return math.dist(a.center_m, b.center_m)


def _gate_radius(ev: Evidence, cell: BeliefCell, cfg: AssociationConfig) -> float:
    a, b = ev.spatial_support, cell.spatial_support
    assert a is not None and b is not None
    sigma = (a.position_sigma_m or 0.0) + (b.position_sigma_m or 0.0)
    extent = math.hypot(*a.half_extent_m) + math.hypot(*b.half_extent_m)
    return cfg.gate_distance_m + cfg.gate_sigma_multiplier * sigma + extent


def retrieve_candidates(
    evidence: Evidence,
    cells: Sequence[BeliefCell],
    cfg: AssociationConfig,
    domain: Domain | None = None,
    entity_type: str | None = None,
) -> list[BeliefCell]:
    """Deterministic gating: lifecycle, domain, entity type, time, frame + spatial proximity; K <= max."""
    hinted = {c.belief_id for c in evidence.entity_candidates if c.belief_id is not None}
    registry = {c.registry_entity_id for c in evidence.entity_candidates if c.registry_entity_id is not None}
    ranked: list[tuple[int, float, int, BeliefCell]] = []
    for cell in cells:
        if cell.lifecycle in TERMINAL:
            continue
        if domain is not None and cell.domain is not domain:
            continue
        if entity_type is not None and cell.entity_type != entity_type:
            continue
        if cell.timestamp.clock_domain != evidence.timestamp.clock_domain:
            continue
        age_s = abs(evidence.timestamp.time_ns - cell.timestamp.time_ns) / 1e9
        if cfg.gate_max_age_s is not None and age_s > cfg.gate_max_age_s:
            continue
        identity = cell.belief_id in hinted or (
            cell.registry_entity_id is not None and cell.registry_entity_id in registry
        )
        dist = _distance(evidence, cell)
        if evidence.spatial_support is not None and cell.spatial_support is not None:
            if dist is None:  # different frames: not comparable, never silently mixed
                continue
            if not identity and dist > _gate_radius(evidence, cell, cfg):
                continue
        ranked.append(
            (0 if identity else 1, dist if dist is not None else math.inf, cell.belief_id.int, cell)
        )
    ranked.sort(key=lambda r: r[:3])
    return [r[3] for r in ranked[: cfg.max_candidates]]


def pair_extra_features(
    evidence: Evidence, cell: BeliefCell, cfg: CoreConfig, context: Tensor | None = None
) -> Tensor:
    """relative_pose(12) + time(4) + edge/context(Dr) + reliability(1) + uncertainty summary(4)."""
    a, b = evidence.spatial_support, cell.spatial_support
    if a is not None and b is not None and a.frame_id == b.frame_id:
        delta = [x - y for x, y in zip(a.center_m, b.center_m, strict=True)]
        pose = [*delta, *[abs(d) for d in delta], *a.half_extent_m, *b.half_extent_m]
    else:
        pose = [0.0] * 12
    pose = (pose + [0.0] * cfg.association.relative_pose_dim)[: cfg.association.relative_pose_dim]
    dt_s = (evidence.timestamp.time_ns - cell.timestamp.time_ns) / 1e9
    time = [
        math.log1p(abs(dt_s)),
        1.0 if dt_s < 0 else 0.0,
        math.log1p(cell.independent_observation_count),
        1.0 if (a is not None and b is not None) else 0.0,
    ]
    time = (time + [0.0] * cfg.association.time_feature_dim)[: cfg.association.time_feature_dim]
    ctx = torch.zeros(cfg.relation_dim) if context is None else context.to(torch.float32)
    if ctx.shape != (cfg.relation_dim,):
        raise ValueError(f"context must have shape ({cfg.relation_dim},)")
    tail = torch.tensor([evidence.reliability, *cell.uncertainty.as_tuple()], dtype=torch.float32)
    return torch.cat([torch.tensor(pose + time, dtype=torch.float32), ctx, tail])


def nearest_neighbour_decision(
    evidence: Evidence, candidates: Sequence[BeliefCell], cfg: AssociationConfig
) -> AssociationDecision:
    """Non-learned baseline: nearest gated candidate inside ``baseline_gate_distance_m``, else NO_MATCH."""
    ids = tuple(c.belief_id for c in candidates)
    registry = {c.registry_entity_id for c in evidence.entity_candidates if c.registry_entity_id is not None}
    for cell in candidates:
        if cell.registry_entity_id is not None and cell.registry_entity_id in registry:
            return AssociationDecision(
                evidence.evidence_id, cell.belief_id, 1.0, 1.0, ids, "registry_identity"
            )
    dists = [(d, c) for c in candidates if (d := _distance(evidence, c)) is not None]
    dists.sort(key=lambda t: (t[0], t[1].belief_id.int))
    if not dists:
        return AssociationDecision(evidence.evidence_id, None, 1.0, 1.0, ids, "nearest_neighbour")
    best_d, best = dists[0]
    sigma = (evidence.spatial_support.position_sigma_m or 0.0) if evidence.spatial_support else 0.0
    gate = cfg.baseline_gate_distance_m + cfg.gate_sigma_multiplier * sigma
    if best_d > gate:
        return AssociationDecision(evidence.evidence_id, None, 1.0, 1.0, ids, "nearest_neighbour")
    # pseudo-probability from relative distances; reported for diagnostics, not calibrated
    runner = dists[1][0] if len(dists) > 1 else math.inf
    margin = 1.0 if math.isinf(runner) else max(0.0, (runner - best_d) / max(runner, 1e-9))
    return AssociationDecision(
        evidence.evidence_id,
        best.belief_id,
        1.0 - best_d / max(gate, 1e-9) * 0.5,
        margin,
        ids,
        "nearest_neighbour",
    )


class AssociationEngine:
    """Retrieval + decision. Uses the learned scorer only when one is supplied; otherwise the baseline."""

    def __init__(self, cfg: CoreConfig, scorer: AssociationScorer | None = None) -> None:
        self.cfg = cfg
        self.scorer = scorer

    def associate(
        self,
        evidence: Evidence,
        cells: Sequence[BeliefCell],
        domain: Domain | None = None,
        entity_type: str | None = None,
    ) -> AssociationDecision:
        candidates = retrieve_candidates(evidence, cells, self.cfg.association, domain, entity_type)
        if self.scorer is None:
            return nearest_neighbour_decision(evidence, candidates, self.cfg.association)
        return self._learned(evidence, candidates)

    def _learned(self, evidence: Evidence, candidates: Sequence[BeliefCell]) -> AssociationDecision:
        assert self.scorer is not None
        ids = tuple(c.belief_id for c in candidates)
        dz = self.cfg.belief_dim
        usable = [c for c in candidates if len(c.state_embedding) == dz]
        if len(evidence.embedding) != self.cfg.evidence_dim:
            raise ValueError("evidence embedding width does not match CoreConfig.evidence_dim")
        k = max(len(usable), 1)
        z = torch.zeros(1, k, dz)
        extra = torch.zeros(1, k, self.scorer.extra_dim)
        mask = torch.zeros(1, k, dtype=torch.bool)
        for i, cell in enumerate(usable):
            z[0, i] = torch.tensor(cell.state_embedding)
            extra[0, i] = pair_extra_features(evidence, cell, self.cfg)
            mask[0, i] = True
        self.scorer.eval()
        with torch.no_grad():
            _, probs = self.scorer(torch.tensor(evidence.embedding).unsqueeze(0), z, extra, mask)
        a = self.cfg.association
        res = accept_associations(probs, a.accept_probability, a.accept_margin)
        idx = int(res.index[0])
        belief_id = usable[idx].belief_id if idx >= 0 else None
        return AssociationDecision(
            evidence.evidence_id,
            belief_id,
            float(res.probability[0]),
            float(res.margin[0]),
            ids,
            "learned_scorer",
        )
