"""Tensorisation of one EGDC decision cycle for the ch33 scorer (graph + candidates -> tensors).

Only claim-graph and context content is used; never truth. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from conrad.decision.claims import ClaimGraph
from conrad.decision.consequence import ConsequenceVector
from conrad.decision.context import DecisionContext
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.decision import (
    WORLD_DEPENDENT_CLAIMS,
    ActionProposal,
    ActionType,
    ClaimEdgeType,
    ClaimType,
    GroundingStatus,
)
from conrad.schemas.timebase import NS_PER_S

if TYPE_CHECKING:
    from conrad.decision.learned import EGDCScorerConfig

_ACTIONS = list(ActionType)
_CLAIM_TYPES = list(ClaimType)
_EDGE_TYPES = list(ClaimEdgeType)
_GROUNDING = list(GroundingStatus)
_STATUS = list(KnowledgeStatus)
EDGE_NONE, EDGE_SELF, EDGE_MISSION = 0, len(_EDGE_TYPES) + 1, len(_EDGE_TYPES) + 2
NUM_EDGE_CODES = len(_EDGE_TYPES) + 3


def expand_features(values: Sequence[float], dim: int) -> list[float]:
    """Deterministic sinusoidal expansion of a few scalars to ``dim`` slots (no learned state, no hashing)."""
    out: list[float] = []
    k = 0
    while len(out) < dim:
        for v in values:
            out.append(math.sin((2.0**k) * v))
            out.append(math.cos((2.0**k) * v))
        k += 1
        if not values:
            out.extend([0.0] * dim)
    return out[:dim]


@dataclass
class EGDCBatch:
    nodes: Tensor  # [B, N, node_input_dim]
    node_mask: Tensor  # [B, N] bool
    unsupported_mask: Tensor  # [B, N] bool, world claims that are UNSUPPORTED
    edges: Tensor  # [B, N+1, N+1] long edge codes (index 0 = [MISSION] token)
    action_types: Tensor  # [B, A] long
    action_mask: Tensor  # [B, A] bool
    action_claims: Tensor  # [B, A, N] bool: claims an action depends on / would resolve
    resource: Tensor  # [B, resource_dim]
    risk: Tensor  # [B, A, risk_dim]
    target_action: Tensor | None = None  # [B] long
    target_support: Tensor | None = None  # [B, A, N] float in {0,1}
    target_outcome: Tensor | None = None  # [B, A, outcome_dim]


def featurize(
    graph: ClaimGraph,
    candidates: Sequence[ActionProposal],
    consequences: Sequence[ConsequenceVector],
    ctx: DecisionContext,
    cfg: EGDCScorerConfig,
) -> EGDCBatch:
    """One decision cycle -> tensors (batch of 1). Only graph/context content is used; never truth."""
    claims = [c for c in graph.claims if c.claim_type is not ClaimType.CANDIDATE_ACTION][: cfg.max_nodes]
    index = {c.claim_id: i for i, c in enumerate(claims)}
    n, a = cfg.max_nodes, len(candidates)
    nodes = torch.zeros(1, n, cfg.node_input_dim)
    mask = torch.zeros(1, n, dtype=torch.bool)
    unsupported = torch.zeros(1, n, dtype=torch.bool)
    for i, c in enumerate(claims):
        if c.content_embedding:
            emb = list(c.content_embedding)[: cfg.claim_embedding_dim]
        else:
            value = c.structured_value.get("value")
            emb = [float(c.claim_type is t) for t in _CLAIM_TYPES]
            emb += [float(c.grounding is g) for g in _GROUNDING]
            emb += [float(c.structured_value.get("knowledge_status") == s.value) for s in _STATUS]
            emb += [
                float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0,
                float(bool(c.structured_value.get("stale"))),
                float(len(c.structured_value.get("issues", []))),
                float(c.structured_value.get("consequence", 0.0) or 0.0),
                float(bool(c.structured_value.get("context"))),
            ]
            emb = emb[: cfg.claim_embedding_dim]
        emb = emb + [0.0] * (cfg.claim_embedding_dim - len(emb))
        u = (0.0, 0.0, 0.0, 0.0) if c.uncertainty is None else c.uncertainty.as_tuple()
        prov = [
            float(len(c.evidence_refs)),
            float(len(c.source_belief_ids)),
            float(c.uncertainty is not None),
        ]
        age = max(0.0, (ctx.timestamp.time_ns - c.timestamp.time_ns) / NS_PER_S)
        row = (
            emb
            + expand_features(u, cfg.uncertainty_dim)
            + expand_features([math.log1p(p) for p in prov], cfg.provenance_dim)
            + expand_features([math.log1p(age)], cfg.context_dim)
        )
        nodes[0, i] = torch.tensor(row)
        mask[0, i] = True
        unsupported[0, i] = (
            c.claim_type in WORLD_DEPENDENT_CLAIMS and c.grounding is GroundingStatus.UNSUPPORTED
        )
    edges = torch.zeros(1, n + 1, n + 1, dtype=torch.long)
    edges[0, 0, :] = EDGE_MISSION
    edges[0, :, 0] = EDGE_MISSION
    for i in range(n + 1):
        edges[0, i, i] = EDGE_SELF
    for e in graph.edges:
        if e.source_claim_id in index and e.target_claim_id in index:
            code = _EDGE_TYPES.index(e.edge_type) + 1
            edges[0, index[e.source_claim_id] + 1, index[e.target_claim_id] + 1] = code
            edges[0, index[e.target_claim_id] + 1, index[e.source_claim_id] + 1] = code
    action_types = torch.tensor([[_ACTIONS.index(c.action_type) for c in candidates]], dtype=torch.long)
    action_claims = torch.zeros(1, a, n, dtype=torch.bool)
    for j, cand in enumerate(candidates):
        related = [*cand.supporting_claims, *cand.parameters.get("motivating_claims", ())]
        for claim_id in related:
            key = claim_id if not isinstance(claim_id, str) else None
            if key is None:
                key = next((k for k in index if str(k) == claim_id), None)
            if key in index:
                action_claims[0, j, index[key]] = True
    r = ctx.resource_state
    resource = expand_features(
        [
            -1.0 if r is None or r.battery_fraction is None else r.battery_fraction,
            float(ctx.motion_permitted),
            0.0 if ctx.link_state is None else math.log1p(ctx.link_state.bandwidth_bps) / 20.0,
        ],
        cfg.resource_dim,
    )
    risk = [
        expand_features(
            [c.risk, c.uncertainty_exposure, c.information, c.mission, c.time, c.energy], cfg.risk_dim
        )
        for c in consequences
    ]
    return EGDCBatch(
        nodes=nodes,
        node_mask=mask,
        unsupported_mask=unsupported,
        edges=edges,
        action_types=action_types,
        action_mask=torch.ones(1, a, dtype=torch.bool),
        action_claims=action_claims,
        resource=torch.tensor([resource]),
        risk=torch.tensor([risk]) if a else torch.zeros(1, 0, cfg.risk_dim),
    )


def collate(batches: Sequence[EGDCBatch]) -> EGDCBatch:
    """Pad the action axis and stack single-cycle batches."""
    a = max(b.action_types.shape[1] for b in batches)

    def pad(t: Tensor, dim: int) -> Tensor:
        short = a - t.shape[dim]
        if short == 0:
            return t
        shape = list(t.shape)
        shape[dim] = short
        return torch.cat([t, torch.zeros(shape, dtype=t.dtype)], dim=dim)

    def opt(name: str, dim: int | None) -> Tensor | None:
        vals = [getattr(b, name) for b in batches]
        if any(v is None for v in vals):
            return None
        return torch.cat([v if dim is None else pad(v, dim) for v in vals])

    return EGDCBatch(
        nodes=torch.cat([b.nodes for b in batches]),
        node_mask=torch.cat([b.node_mask for b in batches]),
        unsupported_mask=torch.cat([b.unsupported_mask for b in batches]),
        edges=torch.cat([b.edges for b in batches]),
        action_types=torch.cat([pad(b.action_types, 1) for b in batches]),
        action_mask=torch.cat([pad(b.action_mask, 1) for b in batches]),
        action_claims=torch.cat([pad(b.action_claims, 1) for b in batches]),
        resource=torch.cat([b.resource for b in batches]),
        risk=torch.cat([pad(b.risk, 1) for b in batches]),
        target_action=opt("target_action", None),
        target_support=opt("target_support", 1),
        target_outcome=opt("target_outcome", 1),
    )
