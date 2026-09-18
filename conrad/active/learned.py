"""ch33 'MCBR exact implementation': learned candidate ranker (EXPERIMENTAL_CANDIDATE).

candidate = [target belief 256, uncertainty 64, hypothesis 128, viewpoint/sensor/config 64,
             observability 64, navigation cost/risk/energy 32, mission 64] -> 256
3 blocks: self-attention over candidates + cross-attention over belief/claim tokens.
heads: visibility (sigmoid), oracle value, hypothesis discrimination, cost/risk correction (2).
score = predicted value - constrained cost. Feasibility filtering still happens BEFORE this ranker.

Training targets (oracle values, visibility labels) are supplied as tensors by evaluation code; this
module never imports an oracle or truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pydantic import Field
from torch import Tensor, nn

from conrad.schemas.base import ConradModel


class MCBRRankerConfig(ConradModel):
    belief_dim: int = 256
    uncertainty_dim: int = 64
    hypothesis_dim: int = 128
    viewpoint_dim: int = 64
    observability_dim: int = 64
    cost_dim: int = 32
    mission_dim: int = 64
    context_dim: int = 256
    model_dim: int = 256
    head_hidden: int = 128
    num_blocks: int = 3
    num_heads: int = 8
    dropout: float = 0.10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    lambda_value: float = 1.0
    lambda_rank: float = 1.0
    lambda_visibility: float = 1.0
    seed: int = Field(default=2026201, ge=0)

    @property
    def candidate_dim(self) -> int:
        return (
            self.belief_dim
            + self.uncertainty_dim
            + self.hypothesis_dim
            + self.viewpoint_dim
            + self.observability_dim
            + self.cost_dim
            + self.mission_dim
        )


@dataclass
class MCBRBatch:
    candidates: Tensor  # [B, K, candidate_dim]
    candidate_mask: Tensor  # [B, K] bool (feasible only)
    context: Tensor  # [B, T, context_dim] belief / claim tokens
    context_mask: Tensor  # [B, T] bool
    oracle_value: Tensor | None = None  # [B, K]
    visible: Tensor | None = None  # [B, K] in {0, 1}


@dataclass
class MCBROutput:
    visibility: Tensor
    value: Tensor
    discrimination: Tensor
    cost_risk: Tensor  # [B, K, 2]
    score: Tensor  # [B, K], -inf where masked


class _Block(nn.Module):
    def __init__(self, cfg: MCBRRankerConfig) -> None:
        super().__init__()
        d = cfg.model_dim
        self.self_attn = nn.MultiheadAttention(d, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(4 * d, d))

    def forward(self, x: Tensor, ctx: Tensor, x_mask: Tensor, ctx_mask: Tensor) -> Tensor:
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, key_padding_mask=~x_mask, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.cross_attn(h, ctx, ctx, key_padding_mask=~ctx_mask, need_weights=False)[0]
        return x + self.ffn(self.n3(x))


def _head(d: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out))


class MCBRRanker(nn.Module):
    def __init__(self, cfg: MCBRRankerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cand_in = nn.Sequential(nn.Linear(cfg.candidate_dim, cfg.model_dim), nn.LayerNorm(cfg.model_dim))
        self.ctx_in = nn.Sequential(nn.Linear(cfg.context_dim, cfg.model_dim), nn.LayerNorm(cfg.model_dim))
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.num_blocks)])
        self.visibility = _head(cfg.model_dim, cfg.head_hidden, 1)
        self.value = _head(cfg.model_dim, cfg.head_hidden, 1)
        self.discrimination = _head(cfg.model_dim, cfg.head_hidden, 1)
        self.cost_risk = _head(cfg.model_dim, cfg.head_hidden, 2)

    def forward(self, batch: MCBRBatch) -> MCBROutput:
        x = self.cand_in(batch.candidates)
        ctx = self.ctx_in(batch.context)
        # every row needs at least one attendable key
        x_mask = batch.candidate_mask.clone()
        x_mask[x_mask.sum(-1) == 0, 0] = True
        c_mask = batch.context_mask.clone()
        c_mask[c_mask.sum(-1) == 0, 0] = True
        for block in self.blocks:
            x = block(x, ctx, x_mask, c_mask)
        vis = torch.sigmoid(self.visibility(x).squeeze(-1))
        value = self.value(x).squeeze(-1)
        cost_risk = nn.functional.softplus(self.cost_risk(x))
        score = (value - cost_risk.sum(-1)).masked_fill(~batch.candidate_mask, float("-inf"))
        return MCBROutput(
            vis, value, nn.functional.softplus(self.discrimination(x).squeeze(-1)), cost_risk, score
        )


def build_ranker(cfg: MCBRRankerConfig) -> MCBRRanker:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        return MCBRRanker(cfg)


def mcbr_loss(out: MCBROutput, batch: MCBRBatch, cfg: MCBRRankerConfig) -> tuple[Tensor, dict[str, float]]:
    """Value regression (Huber) + pairwise logistic ranking + Brier visibility, all masked."""
    m = batch.candidate_mask.float()
    denom = m.sum().clamp(min=1.0)
    value = rank = vis = torch.zeros(())
    if batch.oracle_value is not None:
        y = batch.oracle_value
        value = (nn.functional.huber_loss(out.value, y, reduction="none") * m).sum() / denom
        diff = y[:, :, None] - y[:, None, :]
        pair = (diff > 0).float() * m[:, :, None] * m[:, None, :]
        s = out.value[:, :, None] - out.value[:, None, :]
        rank = (nn.functional.softplus(-s) * pair).sum() / pair.sum().clamp(min=1.0)
    if batch.visible is not None:
        vis = (((out.visibility - batch.visible) ** 2) * m).sum() / denom
    total = cfg.lambda_value * value + cfg.lambda_rank * rank + cfg.lambda_visibility * vis
    return total, {"value": float(value), "rank": float(rank), "visibility": float(vis)}


def train_ranker(
    model: MCBRRanker, batches: list[MCBRBatch], cfg: MCBRRankerConfig, epochs: int
) -> list[float]:
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    losses: list[float] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        model.train()
        for _ in range(epochs):
            total = 0.0
            for batch in batches:
                optim.zero_grad()
                loss, _ = mcbr_loss(model(batch), batch, cfg)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                total += float(loss) / len(batches)
            losses.append(total)
    model.eval()
    return losses
