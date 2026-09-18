"""Learned pair scorer with a learned NO_MATCH candidate, acceptance rule and losses (ch33 Association).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from conrad.core.config import CoreConfig
from conrad.core.primitives import masked_loss, masked_softmax, mlp


class AssociationScorer(nn.Module):
    """pair_dim -> 512 -> 256 -> 128 -> 1 (ch33); NO_MATCH is scored by a separate De -> 128 -> 1."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h1, h2, h3 = cfg.association.hidden
        self.extra_dim = cfg.association_pair_dim - 4 * cfg.evidence_dim
        self.pair = mlp([cfg.association_pair_dim, h1, h2, h3, 1], cfg.dropout, norm_after=(0, 1))
        self.no_match = mlp([cfg.evidence_dim, cfg.association.no_match_hidden, 1], cfg.dropout)

    def forward(self, e: Tensor, z: Tensor, pair_extra: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """``e`` [B, De]; ``z`` [B, K, Dz]; ``pair_extra`` [B, K, X]; ``mask`` [B, K] True=valid.

        Returns ``(logits, probabilities)`` of shape [B, K+1]; the LAST column is NO_MATCH, which is
        always valid, so an empty candidate set yields P(NO_MATCH) = 1.
        """
        if pair_extra.shape[-1] != self.extra_dim:
            raise ValueError(f"pair_extra last dim {pair_extra.shape[-1]} != expected {self.extra_dim}")
        ek = e.unsqueeze(1).expand_as(z)
        x = torch.cat([ek, z, ek - z, ek * z, pair_extra], dim=-1)
        pair_logits = self.pair(x).squeeze(-1)
        logits = torch.cat([pair_logits, self.no_match(e)], dim=-1)
        full_mask = torch.cat([mask.to(torch.bool), torch.ones_like(mask[:, :1], dtype=torch.bool)], dim=-1)
        logits = logits.masked_fill(~full_mask, 0.0)
        return logits, masked_softmax(logits, full_mask)


@dataclass(frozen=True)
class AcceptanceResult:
    index: Tensor  # [B] long; -1 = NO_MATCH / tentative
    probability: Tensor  # [B] probability of the top candidate
    margin: Tensor  # [B] top minus runner-up
    accepted: Tensor  # [B] bool


def accept_associations(probabilities: Tensor, min_probability: float, min_margin: float) -> AcceptanceResult:
    """Accept only if p >= ``min_probability`` AND p exceeds the runner-up by ``min_margin`` (ch33)."""
    k = probabilities.shape[-1] - 1
    top = torch.topk(probabilities, k=min(2, k + 1), dim=-1)
    best_p, best_i = top.values[:, 0], top.indices[:, 0]
    runner = top.values[:, 1] if top.values.shape[-1] > 1 else torch.zeros_like(best_p)
    margin = best_p - runner
    accepted = (best_i < k) & (best_p >= min_probability) & (margin >= min_margin)
    index = torch.where(accepted, best_i, torch.full_like(best_i, -1))
    return AcceptanceResult(index, best_p, margin, accepted)


@dataclass(frozen=True)
class AssociationLoss:
    total: Tensor
    cross_entropy: Tensor
    focal_no_match: Tensor
    ranking: Tensor
    denominators: dict[str, float]


def association_loss(
    logits: Tensor, mask: Tensor, target: Tensor, label_mask: Tensor, cfg: CoreConfig
) -> AssociationLoss:
    """``target`` [B] in [0, K] where K means NO_MATCH; ``label_mask`` [B] marks supervised rows.

    CE over valid candidates + NO_MATCH, focal loss on the (rare) no-match rows, and a pairwise
    ranking term requiring the true logit to exceed every valid negative by the configured margin.
    """
    a = cfg.association
    k = logits.shape[-1] - 1
    full_mask = torch.cat([mask.to(torch.bool), torch.ones_like(mask[:, :1], dtype=torch.bool)], dim=-1)
    safe_target = target.clamp(0, k)
    target_valid = full_mask.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    supervised = label_mask.to(torch.bool) & target_valid
    log_p = torch.log(masked_softmax(logits, full_mask).clamp_min(1e-12))
    nll = -log_p.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    ce, n_ce = masked_loss(nll, supervised)

    is_no_match = supervised & (safe_target == k)
    p_true = torch.exp(-nll)
    focal, n_focal = masked_loss((1 - p_true) ** a.focal_gamma * nll, is_no_match)

    true_logit = logits.gather(1, safe_target.unsqueeze(1))
    negatives = full_mask & ~F.one_hot(safe_target, k + 1).to(torch.bool) & supervised.unsqueeze(1)
    hinge = F.relu(a.ranking_margin - (true_logit - logits))
    rank, n_rank = masked_loss(hinge, negatives)

    total = a.loss_weight_ce * ce + a.loss_weight_focal * focal + a.loss_weight_rank * rank
    dens = {"cross_entropy": float(n_ce), "focal_no_match": float(n_focal), "ranking": float(n_rank)}
    return AssociationLoss(total, ce, focal, rank, dens)
