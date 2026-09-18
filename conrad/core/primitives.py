"""Shared neural primitives (ch33 Shared dimensions and primitives, ch28 Tensor and Mask Semantics).

Mask convention everywhere in ``conrad.core``: ``True`` means VALID. Padding contributes zero
attention weight and zero loss-denominator contribution (CC-05, CC-06).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ResidualMLP(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout -> residual add (ch33).

    When ``d_in != d_out`` the residual path is a learned linear projection.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(dropout)
        self.skip: nn.Module = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)

    def forward(self, x: Tensor) -> Tensor:
        h = self.drop(F.gelu(self.fc1(self.norm(x))))
        out: Tensor = self.skip(x) + self.drop(self.fc2(h))
        return out


def mlp(dims: list[int], dropout: float = 0.0, norm_after: tuple[int, ...] = ()) -> nn.Sequential:
    """Plain GELU MLP. ``norm_after`` lists hidden-layer indices (0-based) followed by LayerNorm."""
    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            if idx in norm_after:
                layers.append(nn.LayerNorm(dims[idx + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def masked_softmax(scores: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Softmax over valid entries only. Rows with no valid entry return exact zeros (not NaN)."""
    mask = mask.to(torch.bool)
    neg = torch.finfo(scores.dtype).min
    filled = scores.masked_fill(~mask, neg)
    weights = torch.softmax(filled, dim=dim) * mask.to(scores.dtype)
    total = weights.sum(dim=dim, keepdim=True)
    return weights / total.clamp_min(torch.finfo(scores.dtype).tiny)


def masked_mean(values: Tensor, mask: Tensor, dim: int = -2) -> Tensor:
    """Mean of ``values`` [..., K, D] over valid rows of ``mask`` [..., K]. Empty sets give zeros."""
    m = mask.to(values.dtype).unsqueeze(-1)
    total = (values * m).sum(dim=dim)
    count = m.sum(dim=dim)
    return total / count.clamp_min(1.0)


def masked_loss(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """L = sum(m * l) / max(1, sum(m)). Returns ``(loss, denominator)``.

    All-false masks return a zero loss that is still attached to the graph and a reported
    denominator of exactly zero so sparse supervision is visible in logs (CC-05).
    """
    if values.shape != mask.shape:
        raise ValueError(f"values {tuple(values.shape)} and mask {tuple(mask.shape)} differ in shape")
    m = mask.to(torch.float32)
    safe = torch.where(mask.to(torch.bool), values.to(torch.float32), torch.zeros_like(m))
    denominator = m.sum()
    loss = (safe * m).sum() / denominator.clamp_min(1.0)
    return loss, denominator.detach()


class PreNormTransformerBlock(nn.Module):
    """Pre-norm multi-head self-attention followed by the residual feed-forward block (ch33)."""

    def __init__(self, d_model: int, nhead: int, ffn: int, dropout: float = 0.10) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn = ResidualMLP(d_model, ffn, d_model, dropout)

    def forward(
        self, tokens: Tensor, key_padding_mask: Tensor | None = None, attn_bias: Tensor | None = None
    ) -> Tensor:
        """``tokens`` [B, L, D]; ``key_padding_mask`` [B, L] True=valid; ``attn_bias`` [B, L, L] or [B, H, L, L]."""
        b, length, d = tokens.shape
        valid = (
            torch.ones(b, length, dtype=torch.bool, device=tokens.device)
            if key_padding_mask is None
            else key_padding_mask.to(torch.bool)
        )
        q, k, v = self.qkv(self.norm(tokens)).view(b, length, 3, self.nhead, self.head_dim).unbind(dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            scores = scores + (attn_bias.unsqueeze(1) if attn_bias.dim() == 3 else attn_bias)
        weights = self.drop(masked_softmax(scores, valid[:, None, None, :].expand_as(scores)))
        mixed = torch.einsum("bhqk,bkhd->bqhd", weights, v).reshape(b, length, d)
        tokens = tokens + self.drop(self.out(mixed))
        out: Tensor = self.ffn(tokens)
        # padded positions carry no information forward
        return out * valid.unsqueeze(-1).to(out.dtype)


class AttentionPooling(nn.Module):
    """Permutation- and padding-invariant set pooling with a learned query (CC-06)."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.query, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, items: Tensor, mask: Tensor, query: Tensor | None = None) -> Tensor:
        """``items`` [B, K, D], ``mask`` [B, K] -> [B, D]. An empty set pools to the zero vector."""
        b, k_max, d = items.shape
        q_vec = self.query.expand(b, d) if query is None else query
        q = q_vec.view(b, self.nhead, self.head_dim)
        k, v = self.kv(self.norm(items)).view(b, k_max, 2, self.nhead, self.head_dim).unbind(dim=2)
        scores = torch.einsum("bhd,bkhd->bhk", q, k) / math.sqrt(self.head_dim)
        weights = self.drop(masked_softmax(scores, mask[:, None, :].expand_as(scores)))
        pooled = torch.einsum("bhk,bkhd->bhd", weights, v).reshape(b, d)
        out: Tensor = self.out(pooled)
        return out * mask.any(dim=1, keepdim=True).to(out.dtype)


class SetAggregator(nn.Module):
    """Two stacked attention-pooling blocks (ch33 BUO: simultaneous evidence).

    Block 1 pools with a learned query; block 2 re-attends to the items conditioned on the first
    summary. No positional information is used, so the result is invariant to item order and padding.
    """

    def __init__(self, d_model: int, nhead: int, blocks: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(AttentionPooling(d_model, nhead, dropout) for _ in range(blocks))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, items: Tensor, mask: Tensor) -> Tensor:
        summary: Tensor | None = None
        for block in self.blocks:
            pooled = block(items, mask, summary)
            summary = pooled if summary is None else summary + pooled
        assert summary is not None
        out: Tensor = self.norm(summary)
        return out * mask.any(dim=1, keepdim=True).to(out.dtype)


def weighted_pool(items: Tensor, weights: Tensor, mask: Tensor) -> Tensor:
    """Baseline aggregation: normalised non-negative weights over valid items (ch3 default baseline)."""
    w = weights.clamp_min(0.0) * mask.to(items.dtype)
    return (items * w.unsqueeze(-1)).sum(dim=-2) / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
