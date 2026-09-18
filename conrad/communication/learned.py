"""ch33 'BAAC exact implementation': learned value heads (EXPERIMENTAL_CANDIDATE).

Receiver state = up to 256 belief-summary tokens; sender units and receiver tokens are encoded by
separate features -> 256 MLPs, then two cross-attention blocks. Per-unit heads: receiver novelty,
expected mission value, expected information loss at F0..F4, bits, latency, deadline-violation
probability. These heads ESTIMATE values for the deterministic scheduler; they cannot bypass its
bandwidth / deadline / energy / link constraints. Oracle targets arrive as tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pydantic import Field
from torch import Tensor, nn

from conrad.schemas.base import ConradModel


class BAACHeadsConfig(ConradModel):
    unit_feature_dim: int = 64
    receiver_feature_dim: int = 64
    model_dim: int = 256
    max_receiver_tokens: int = 256
    num_blocks: int = 2
    num_heads: int = 8
    num_fidelities: int = 5
    dropout: float = 0.10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = Field(default=2026201, ge=0)


@dataclass
class BAACBatch:
    units: Tensor  # [B, U, unit_feature_dim]
    unit_mask: Tensor  # [B, U] bool
    receiver: Tensor  # [B, R, receiver_feature_dim], R <= max_receiver_tokens
    receiver_mask: Tensor  # [B, R] bool
    novelty: Tensor | None = None  # [B, U] in {0, 1}
    value: Tensor | None = None  # [B, U] oracle V_comm
    info_loss: Tensor | None = None  # [B, U, 5]
    deadline_violation: Tensor | None = None  # [B, U] in {0, 1}


@dataclass
class BAACOutput:
    novelty: Tensor
    value: Tensor
    info_loss: Tensor
    log_bits: Tensor
    latency_s: Tensor
    deadline_violation: Tensor


def _mlp(i: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d))


class BAACValueHeads(nn.Module):
    def __init__(self, cfg: BAACHeadsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.model_dim
        self.unit_in = _mlp(cfg.unit_feature_dim, d)
        self.recv_in = _mlp(cfg.receiver_feature_dim, d)
        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(d, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
                for _ in range(cfg.num_blocks)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(cfg.num_blocks)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                for _ in range(cfg.num_blocks)
            ]
        )
        self.heads = nn.Linear(d, 1 + 1 + cfg.num_fidelities + 1 + 1 + 1)

    def forward(self, batch: BAACBatch) -> BAACOutput:
        if batch.receiver.shape[1] > self.cfg.max_receiver_tokens:
            raise ValueError("receiver state exceeds max_receiver_tokens")
        x = self.unit_in(batch.units)
        r = self.recv_in(batch.receiver)
        mask = batch.receiver_mask.clone()
        mask[mask.sum(-1) == 0, 0] = True
        for attn, norm, ffn in zip(self.attn, self.norms, self.ffn, strict=True):
            x = x + attn(norm(x), r, r, key_padding_mask=~mask, need_weights=False)[0]
            x = x + ffn(x)
        h = self.heads(x)
        f = self.cfg.num_fidelities
        return BAACOutput(
            novelty=torch.sigmoid(h[..., 0]),
            value=h[..., 1],
            info_loss=torch.sigmoid(h[..., 2 : 2 + f]),
            log_bits=h[..., 2 + f],
            latency_s=nn.functional.softplus(h[..., 3 + f]),
            deadline_violation=torch.sigmoid(h[..., 4 + f]),
        )


def build_heads(cfg: BAACHeadsConfig) -> BAACValueHeads:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        return BAACValueHeads(cfg)


def baac_loss(out: BAACOutput, batch: BAACBatch) -> Tensor:
    """Huber value loss + BCE novelty/deadline + fidelity information-loss regression, masked."""
    m = batch.unit_mask.float()
    n = m.sum().clamp(min=1.0)
    total = torch.zeros(())
    if batch.value is not None:
        total = total + (nn.functional.huber_loss(out.value, batch.value, reduction="none") * m).sum() / n
    if batch.novelty is not None:
        bce = nn.functional.binary_cross_entropy(
            out.novelty.clamp(1e-6, 1 - 1e-6), batch.novelty, reduction="none"
        )
        total = total + (bce * m).sum() / n
    if batch.deadline_violation is not None:
        bce = nn.functional.binary_cross_entropy(
            out.deadline_violation.clamp(1e-6, 1 - 1e-6), batch.deadline_violation, reduction="none"
        )
        total = total + (bce * m).sum() / n
    if batch.info_loss is not None:
        total = total + (((out.info_loss - batch.info_loss) ** 2).mean(-1) * m).sum() / n
    return total


def train_heads(model: BAACValueHeads, batches: list[BAACBatch], epochs: int) -> list[float]:
    cfg = model.cfg
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    losses = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        model.train()
        for _ in range(epochs):
            total = 0.0
            for b in batches:
                optim.zero_grad()
                loss = baac_loss(model(b), b)
                loss.backward()
                optim.step()
                total += float(loss) / len(batches)
            losses.append(total)
    model.eval()
    return losses
