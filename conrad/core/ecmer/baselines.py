"""ECMER fusion baselines over pre-fusion modality tokens (ch4 Baselines).

All take ``modality_tokens`` {name: [B, De]} and ``availability`` [B, n_modalities] and return [B, De].

implementation_status: EXPERIMENTAL_CANDIDATE (comparators, never the canonical path)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from conrad.core.config import CoreConfig
from conrad.core.ecmer.fusion import MODALITIES
from conrad.core.primitives import PreNormTransformerBlock, masked_mean, masked_softmax, mlp


def _stack(tokens: dict[str, Tensor], availability: Tensor, de: int) -> tuple[Tensor, Tensor]:
    b = availability.shape[0]
    items = torch.stack([tokens.get(n, availability.new_zeros(b, de)) for n in MODALITIES], dim=1)
    present = torch.tensor([n in tokens for n in MODALITIES]).unsqueeze(0)
    return items, (availability > 0) & present


class SingleModalityBaseline(nn.Module):
    def __init__(self, cfg: CoreConfig, modality: str) -> None:
        super().__init__()
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}")
        self.modality, self.de = modality, cfg.evidence_dim
        self.head = mlp([cfg.evidence_dim, cfg.evidence_dim, cfg.evidence_dim])

    def forward(self, tokens: dict[str, Tensor], availability: Tensor) -> Tensor:
        idx = MODALITIES.index(self.modality)
        x = tokens.get(self.modality, availability.new_zeros(availability.shape[0], self.de))
        out: Tensor = self.head(x) * availability[:, idx : idx + 1]
        return out


class ConcatBaseline(nn.Module):
    """Raw feature concatenation (missing modality -> zeros in its slot)."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.de = cfg.evidence_dim
        self.head = mlp([len(MODALITIES) * cfg.evidence_dim, cfg.evidence_dim, cfg.evidence_dim])

    def forward(self, tokens: dict[str, Tensor], availability: Tensor) -> Tensor:
        items, mask = _stack(tokens, availability, self.de)
        out: Tensor = self.head((items * mask.unsqueeze(-1)).flatten(1))
        return out


class EarlyFusionBaseline(nn.Module):
    """Features summed into ONE vector before any joint processing (identity of modality lost)."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.de = cfg.evidence_dim
        self.head = mlp([cfg.evidence_dim, cfg.evidence_dim, cfg.evidence_dim])

    def forward(self, tokens: dict[str, Tensor], availability: Tensor) -> Tensor:
        items, mask = _stack(tokens, availability, self.de)
        out: Tensor = self.head((items * mask.unsqueeze(-1)).sum(1))
        return out


class LateFusionBaseline(nn.Module):
    """Independent per-modality heads, outputs averaged over available modalities."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.de = cfg.evidence_dim
        self.heads = nn.ModuleList(mlp([self.de, self.de, self.de]) for _ in MODALITIES)

    def forward(self, tokens: dict[str, Tensor], availability: Tensor) -> Tensor:
        items, mask = _stack(tokens, availability, self.de)
        outs = torch.stack([h(items[:, i]) for i, h in enumerate(self.heads)], dim=1)
        return masked_mean(outs, mask)


class CrossAttentionBaseline(nn.Module):
    """Standard cross-attention: a learned query attends over modality tokens (single layer)."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.de = cfg.evidence_dim
        self.query = nn.Parameter(torch.randn(self.de) * 0.02)
        self.k = nn.Linear(self.de, self.de)
        self.v = nn.Linear(self.de, self.de)
        self.block = PreNormTransformerBlock(self.de, cfg.heads, cfg.ffn_hidden, cfg.dropout)

    def forward(self, tokens: dict[str, Tensor], availability: Tensor) -> Tensor:
        items, mask = _stack(tokens, availability, self.de)
        scores = (self.k(items) @ self.query) / self.de**0.5
        w = masked_softmax(scores, mask)
        pooled = (w.unsqueeze(-1) * self.v(items)).sum(1, keepdim=True)
        out: Tensor = self.block(pooled)[:, 0]
        return out


def build_fusion_baseline(cfg: CoreConfig, name: str) -> nn.Module:
    if name.startswith("single_"):
        return SingleModalityBaseline(cfg, name.removeprefix("single_"))
    table: dict[str, type[nn.Module]] = {
        "concat": ConcatBaseline,
        "early": EarlyFusionBaseline,
        "late": LateFusionBaseline,
        "cross_attention": CrossAttentionBaseline,
    }
    if name not in table:
        raise ValueError(f"unknown ECMER baseline {name!r}")
    return table[name](cfg)
