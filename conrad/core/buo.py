"""BUO: learned belief update operator with TRUST != INNOVATION (ch5, ch33 BUO exact implementation).

The learned operator is an EXPERIMENTAL_CANDIDATE. :class:`AnalyticBUO` (re-exported here) is the
default runtime operator until a learned one is validated.

Assumption (spec gap): ch33 defines ``z_direct = LN(z + gate * innovation)`` and a separate ``trust``
head without saying where trust enters; ch5 gives ``z + g_trust * g_innovation * dz``. This module
uses ``LN(z + trust * gate * innovation)`` and exposes ablation switches for each factor.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from conrad.core.buo_analytic import AnalyticBUO, AnalyticUpdateResult, EvidenceAssessment
from conrad.core.config import CoreConfig
from conrad.core.primitives import ResidualMLP, SetAggregator, masked_mean, mlp
from conrad.core.uncertainty import UncertaintyHeads

__all__ = [
    "AnalyticBUO",
    "AnalyticUpdateResult",
    "BeliefUpdateOperator",
    "BuoOutput",
    "EvidenceAssessment",
    "collapse_independence_groups",
    "reject_duplicates",
]

SUPPORT, CONFLICT, AMBIGUOUS = 0, 1, 2


def reject_duplicates(evidence_keys: Tensor, consumed_keys: Tensor, mask: Tensor) -> Tensor:
    """Mask out evidence already consumed by the belief and repeats inside the batch row.

    ``evidence_keys`` [B, K] integer keys (e.g. a hash of evidence_id); ``consumed_keys`` [B, C].
    Runs BEFORE the BUO block (ch33: duplicate evidence is rejected before this block).
    """
    seen = (evidence_keys.unsqueeze(-1) == consumed_keys.unsqueeze(1)).any(-1)
    same = evidence_keys.unsqueeze(-1) == evidence_keys.unsqueeze(1)
    earlier = torch.tril(torch.ones_like(same[0]), diagonal=-1).unsqueeze(0)
    repeat = (same & earlier & mask.unsqueeze(1)).any(-1)
    return mask & ~seen & ~repeat


def collapse_independence_groups(
    e: Tensor, quality: Tensor, mask: Tensor, group_ids: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Average evidence sharing an independence group into ONE item so repeats cannot add certainty."""
    same = (group_ids.unsqueeze(-1) == group_ids.unsqueeze(1)) & mask.unsqueeze(1) & mask.unsqueeze(-1)
    w = same.to(e.dtype)
    count = w.sum(-1, keepdim=True).clamp_min(1.0)
    e_mean = torch.einsum("bkj,bjd->bkd", w, e) / count
    q_mean = torch.einsum("bkj,bjd->bkd", w, quality) / count
    earlier = torch.tril(torch.ones_like(same[0]), diagonal=-1).unsqueeze(0)
    first = mask & ~(same & earlier).any(-1)
    return e_mean, q_mean, first


@dataclass(frozen=True)
class BuoOutput:
    z: Tensor  # [B, Dz] corrected latent
    u: Tensor  # [B, Du] corrected uncertainty latent
    channels: Tensor  # [B, 4] softplus (UA, UE, UC, UO)
    trust: Tensor  # [B, 1]
    gate: Tensor  # [B, Dz] innovation gate
    innovation: Tensor  # [B, Dz]
    pair_trust: Tensor  # [B, K]
    pair_compatibility: Tensor  # [B, K, 3] logits support/conflict/ambiguous
    pair_novelty: Tensor  # [B, K]
    contradiction_memory: Tensor  # [B, K] bool: credible conflicting evidence kept for later
    used_mask: Tensor  # [B, K] items that entered the update
    updated: Tensor  # [B] bool


class BeliefUpdateOperator(nn.Module):
    def __init__(self, cfg: CoreConfig, credible_trust: float = 0.5) -> None:
        super().__init__()
        self.cfg = cfg
        self.credible_trust = credible_trust
        b, dz, du = cfg.buo, cfg.belief_dim, cfg.uncertainty_dim
        self.item = nn.Linear(cfg.evidence_dim + b.quality_dim, cfg.evidence_dim)
        self.aggregate = SetAggregator(cfg.evidence_dim, cfg.heads, b.pooling_blocks, cfg.dropout)
        self.trunk = ResidualMLP(cfg.buo_input_dim, b.trunk_hidden, dz, cfg.dropout)
        self.trust = mlp([dz, b.trust_hidden, 1], cfg.dropout)
        self.innovation = mlp([dz, b.innovation_hidden, dz], cfg.dropout)
        self.gate = mlp([dz, b.gate_hidden, dz], cfg.dropout)
        self.compatibility = mlp([dz, b.trust_hidden, 3], cfg.dropout)
        self.novelty = mlp([dz, b.trust_hidden, 1], cfg.dropout)
        h1, h2 = b.uncertainty_hidden
        self.uncertainty_update = mlp([du + dz + b.quality_dim, h1, h2, du], cfg.dropout)
        self.heads = UncertaintyHeads(du)
        self.norm = nn.LayerNorm(dz)

    def _trunk(self, z: Tensor, e: Tensor, u: Tensor, quality: Tensor, temporal: Tensor) -> Tensor:
        out: Tensor = self.trunk(torch.cat([z, e, z - e, z * e, u, quality, temporal], dim=-1))
        return out

    def forward(
        self,
        z: Tensor,
        u: Tensor,
        temporal: Tensor,
        e: Tensor,
        quality: Tensor,
        mask: Tensor,
        group_ids: Tensor | None = None,
    ) -> BuoOutput:
        """``z`` [B,Dz], ``u`` [B,Du], ``temporal`` [B,Dt], ``e`` [B,K,De], ``quality`` [B,K,4], ``mask`` [B,K]."""
        b_cfg = self.cfg.buo
        mask = mask.to(torch.bool)
        k = e.shape[1]
        # per-pair diagnostics: trust, compatibility and novelty stay separate signals
        zk, uk, tk = (t.unsqueeze(1).expand(-1, k, -1) for t in (z, u, temporal))
        h_pair = self._trunk(zk, e, uk, quality, tk)
        pair_trust = torch.sigmoid(self.trust(h_pair)).squeeze(-1) * mask
        pair_compat = self.compatibility(h_pair)
        pair_novelty = torch.sigmoid(self.novelty(h_pair)).squeeze(-1) * mask
        conflict = pair_compat.argmax(-1) == CONFLICT
        memory = conflict & (pair_trust >= self.credible_trust) & mask
        if not b_cfg.use_contradiction:
            memory = torch.zeros_like(memory)

        if group_ids is not None:
            e_set, q_set, used = collapse_independence_groups(e, quality, mask, group_ids)
        else:
            e_set, q_set, used = e, quality, mask
        pooled = self.aggregate(self.item(torch.cat([e_set, q_set], dim=-1)), used)
        q_pooled = masked_mean(q_set, used)
        h = self._trunk(z, pooled, u, q_pooled, temporal)
        trust = torch.sigmoid(self.trust(h))
        innovation = self.innovation(h)
        gate = torch.sigmoid(self.gate(h))
        if b_cfg.collapse_trust_innovation:
            gate = trust.expand_as(gate)
        if not b_cfg.use_innovation_gate:
            gate = torch.ones_like(gate)
        strength = gate * trust if b_cfg.use_trust and not b_cfg.collapse_trust_innovation else gate
        updated = used.any(dim=1)
        keep = updated.unsqueeze(-1)
        z_direct = torch.where(keep, self.norm(z + strength * innovation), z)
        delta_u = self.uncertainty_update(torch.cat([u, h, q_pooled], dim=-1))
        if not b_cfg.use_contradiction:
            p = self.cfg.partition_dim
            delta_u = torch.cat(
                [delta_u[:, : 2 * p], torch.zeros_like(delta_u[:, 2 * p : 3 * p]), delta_u[:, 3 * p :]], -1
            )
        u_direct = torch.where(keep, u + delta_u, u)
        return BuoOutput(
            z_direct,
            u_direct,
            self.heads(u_direct),
            trust,
            gate,
            innovation,
            pair_trust,
            pair_compat,
            pair_novelty,
            memory,
            used,
            updated,
        )
