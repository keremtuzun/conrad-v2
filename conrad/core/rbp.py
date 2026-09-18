"""RBP: relation-conditioned graph-transformer layers (ch33 RBP exact implementation).

Edge convention: ``edge_index[0]`` = source j, ``edge_index[1]`` = target i (message j -> i).
The learned module changes z only. It never touches the uncertainty latent, so it cannot lower
observational uncertainty; every committed result is labelled INFERRED by the pipeline/PMBL.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from conrad.core.config import CoreConfig
from conrad.core.primitives import masked_loss, mlp
from conrad.core.rbp_analytic import AnalyticRBP, RelationalInference

__all__ = [
    "AnalyticRBP",
    "RbpOutput",
    "RelationEmbedding",
    "RelationalBeliefPropagation",
    "RelationalInference",
    "contamination_loss",
    "segment_softmax",
]


def segment_softmax(scores: Tensor, segment: Tensor, n_segments: int, mask: Tensor) -> Tensor:
    """Softmax of ``scores`` [M, H] within each target segment; masked edges get exactly zero weight."""
    m = mask.to(scores.dtype).unsqueeze(-1)
    neg = torch.finfo(scores.dtype).min
    filled = torch.where(m > 0, scores, torch.full_like(scores, neg))
    idx = segment.unsqueeze(-1).expand_as(filled)
    peak = torch.full((n_segments, scores.shape[1]), neg, dtype=scores.dtype, device=scores.device)
    peak = peak.scatter_reduce(0, idx, filled, reduce="amax", include_self=True)
    ex = torch.exp(filled - peak[segment]) * m
    total = torch.zeros_like(peak).index_add(0, segment, ex)
    return ex / total[segment].clamp_min(torch.finfo(scores.dtype).tiny)


class RelationEmbedding(nn.Module):
    """r_ji from typed relation id + geometry, distance, age, edge confidence: R* -> 128 -> 64."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.typed = cfg.rbp.typed
        self.type_embedding = nn.Embedding(cfg.rbp.relation_types, cfg.relation_dim)
        self.net = mlp(
            [cfg.relation_dim + cfg.rbp.relation_numeric_dim, cfg.rbp.relation_hidden, cfg.relation_dim]
        )

    def forward(self, edge_type: Tensor, edge_numeric: Tensor) -> Tensor:
        typed = self.type_embedding(edge_type)
        if not self.typed:  # ablation: untyped RBP
            typed = torch.zeros_like(typed)
        numeric = torch.cat([edge_numeric[:, :3], torch.log1p(edge_numeric[:, 3:].clamp_min(0))], dim=-1)
        out: Tensor = self.net(torch.cat([typed, numeric], dim=-1))
        return out


class RbpLayer(nn.Module):
    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        dz, du, dr = cfg.belief_dim, cfg.uncertainty_dim, cfg.relation_dim
        self.heads, self.head_dim = cfg.heads, cfg.head_dim
        self.wq = nn.Linear(dz + du, dz)
        self.wk = nn.Linear(dz + dr + 1, dz)
        self.wv = nn.Linear(dz + dr + 4, dz)
        self.wo = nn.Linear(dz, dz, bias=False)
        self.relation_bias = nn.Linear(dr, cfg.heads)
        self.gate = mlp([2 * dz + du, cfg.rbp.gate_hidden, dz])
        self.norm = nn.LayerNorm(dz)

    def forward(
        self,
        z: Tensor,
        u: Tensor,
        r: Tensor,
        edge_index: Tensor,
        edge_mask: Tensor,
        provenance_strength: Tensor,
        source_uncertainty: Tensor,
    ) -> tuple[Tensor, Tensor]:
        n = z.shape[0]
        src, dst = edge_index[0], edge_index[1]
        q = self.wq(torch.cat([z, u], -1))[dst].view(-1, self.heads, self.head_dim)
        k = self.wk(torch.cat([z[src], r, provenance_strength[src].unsqueeze(-1)], -1))
        v = self.wv(torch.cat([z[src], r, source_uncertainty[src]], -1))
        k, v = k.view(-1, self.heads, self.head_dim), v.view(-1, self.heads, self.head_dim)
        scores = (q * k).sum(-1) / math.sqrt(self.head_dim) + self.relation_bias(r)
        alpha = segment_softmax(scores, dst, n, edge_mask)
        m = torch.zeros(n, self.heads, self.head_dim, dtype=z.dtype, device=z.device)
        m = m.index_add(0, dst, alpha.unsqueeze(-1) * v).reshape(n, -1)
        gate = torch.sigmoid(self.gate(torch.cat([z, m, u], -1)))
        return self.norm(z + gate * self.wo(m)), alpha


@dataclass(frozen=True)
class RbpOutput:
    z: Tensor
    updated: Tensor  # [N] bool: nodes that received at least one valid message
    attention: Tensor  # [M, H] last-layer attention
    layers_run: int
    delta_norm: Tensor  # [N] RMS latent change


class RelationalBeliefPropagation(nn.Module):
    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.relation = RelationEmbedding(cfg)
        self.layers = nn.ModuleList(RbpLayer(cfg) for _ in range(cfg.rbp.layers))

    def forward(
        self,
        z: Tensor,
        u: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_numeric: Tensor,
        node_mask: Tensor | None = None,
        edge_mask: Tensor | None = None,
        provenance_strength: Tensor | None = None,
        source_uncertainty: Tensor | None = None,
        depth: int | None = None,
    ) -> RbpOutput:
        n, m_edges = z.shape[0], edge_index.shape[1]
        nodes = torch.ones(n, dtype=torch.bool) if node_mask is None else node_mask.to(torch.bool)
        edges = torch.ones(m_edges, dtype=torch.bool) if edge_mask is None else edge_mask.to(torch.bool)
        edges = edges & nodes[edge_index[0]] & nodes[edge_index[1]]  # absent nodes contribute nothing
        strength = z.new_ones(n) if provenance_strength is None else provenance_strength
        src_unc = z.new_zeros(n, 4) if source_uncertainty is None else source_uncertainty
        receives = torch.zeros(n, dtype=torch.bool).index_fill(0, edge_index[1][edges], True)
        depth = len(self.layers) if depth is None else depth
        if not 0 < depth <= len(self.layers):
            raise ValueError(f"depth must be in 1..{len(self.layers)}")
        r = self.relation(edge_type, edge_numeric)
        current = z
        alpha = z.new_zeros(m_edges, self.cfg.heads)
        run = 0
        for layer in list(self.layers)[:depth]:
            proposed, alpha = layer(current, u, r, edge_index, edges, strength, src_unc)
            nxt = torch.where(receives.unsqueeze(-1), proposed, current)
            run += 1
            change = (nxt - current).abs().max() if n else z.new_zeros(())
            current = nxt
            if self.cfg.rbp.adaptive_stop and float(change) < self.cfg.rbp.adaptive_epsilon:
                break
        delta = torch.sqrt(((current - z) ** 2).mean(-1)) if n else z.new_zeros(0)
        return RbpOutput(current, receives, alpha, run, delta)


def contamination_loss(
    error_before: Tensor, error_after: Tensor, misleading_target: Tensor
) -> tuple[Tensor, Tensor]:
    """Penalty when propagation over a misleading edge makes the target WORSE (ch3 L_unsupported).

    ``error_*`` [N] are per-node state errors against synthetic truth (training only);
    ``misleading_target`` [N] marks nodes whose incoming edges are known to be misleading.
    Returns ``(loss, denominator)``.
    """
    return masked_loss(torch.relu(error_after - error_before), misleading_target.to(torch.bool))
