"""Entity stream: features -> 128 -> 256 tokens, then heterogeneous graph-transformer layers whose
attention is biased by typed relation embeddings (ch33: 3 layers, 8 heads, relation dim 64).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class HeteroGraphTransformerLayer(nn.Module):
    def __init__(self, d: int, heads: int, n_rel: int, rel_dim: int, ffn: int, dropout: float) -> None:
        super().__init__()
        if d % heads:
            raise ValueError(f"d_model {d} is not divisible by heads {heads}")
        self.h, self.dk = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.rel_emb = nn.Embedding(n_rel + 1, rel_dim)  # index 0 = no edge
        self.rel_bias = nn.Linear(rel_dim, heads)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))

    def forward(self, x: Tensor, rel: Tensor, mask: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        logits = logits + self.rel_bias(self.rel_emb(rel)).permute(0, 3, 1, 2)
        eye = torch.eye(n, dtype=torch.bool, device=x.device)[None]
        allowed = ((rel > 0) | eye) & mask[:, None, :] & mask[:, :, None]
        allowed = allowed | eye  # padded rows attend to themselves only (kept finite)
        logits = logits.masked_fill(~allowed[:, None], float("-inf"))
        att = self.drop(torch.softmax(logits, dim=-1))
        y = (att @ v).transpose(1, 2).reshape(b, n, d)
        x = x + self.drop(self.out(y))
        return x + self.drop(self.ffn(self.ln2(x)))


class EntityStream(nn.Module):
    def __init__(
        self,
        feat: int,
        hidden: int,
        d: int,
        layers: int,
        heads: int,
        n_rel: int,
        rel_dim: int,
        ffn: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(feat, hidden), nn.GELU(), nn.Linear(hidden, d), nn.LayerNorm(d))
        self.layers = nn.ModuleList(
            HeteroGraphTransformerLayer(d, heads, n_rel, rel_dim, ffn, dropout) for _ in range(layers)
        )

    def tokens(self, features: Tensor) -> Tensor:
        out: Tensor = self.embed(features)
        return out

    def blocks(self, x: Tensor, rel: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, rel, mask)
        return x
