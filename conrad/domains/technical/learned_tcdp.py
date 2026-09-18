"""Learned TCDP (ch33): three mechanism-conditioned message layers. EXPERIMENTAL_CANDIDATE.

Per edge j -> i: message MLP concat(z_i, z_j, r_ji, mechanism, delta_t) -> d_hidden -> d_z -> d_z, gate
d_hidden -> d_gate -> 1, reliability-weighted attention over incoming edges, then the RBP update
z_i <- LN(z_i + gate_i * Wo(m_i)) applied ONLY to the mutable latent partitions (degradation / load /
history); material and geometry partitions are never rewritten. Directly observed nodes are never
updated (hard constraint, mirrors the analytic TCDP). Not on the Model2T runtime path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from conrad.domains.technical.config import LearnedTCDPConfig

MECHANISM_KINDS: tuple[str, ...] = (
    "CORROSION",
    "FATIGUE",
    "CRACK",
    "LOAD_PATH",
    "REPAIR_INTERVENTION",
    "GENERIC",
)


@dataclass
class GraphBatch:
    node_features: Tensor  # [N, d_node_in]
    edge_index: Tensor  # [2, E] long (source j, destination i)
    edge_attr: Tensor  # [E, d_r]; masked features are zero
    mechanism: Tensor  # [E] long index into MECHANISM_KINDS
    reliability: Tensor  # [E] in (0, 1]: source direct support
    delta_t: Tensor  # [E] log1p(days since source evidence)
    observed: Tensor  # [N] bool: directly observed nodes (never updated)
    target: Tensor | None = None  # [N, 2] (corrosion mm, crack mm)
    target_mask: Tensor | None = None  # [N, 2] bool, True = supervised
    misleading: Tensor | None = None  # [N] bool: healthy nodes beside a degraded neighbour


@dataclass
class TCDPOutput:
    z: Tensor
    mean: Tensor  # [N, 2]
    log_var: Tensor  # [N, 2]
    gates: list[Tensor]


def _mutable_mask(cfg: LearnedTCDPConfig) -> Tensor:
    if sum(cfg.partitions) != cfg.d_z:
        raise ValueError("latent partitions must sum to d_z")
    parts = [torch.full((n,), i in cfg.mutable_partitions) for i, n in enumerate(cfg.partitions)]
    return torch.cat(parts)


class TCDPLayer(nn.Module):
    def __init__(self, cfg: LearnedTCDPConfig) -> None:
        super().__init__()
        d_in = 2 * cfg.d_z + cfg.d_r + cfg.d_mech + 1
        self.msg_in = nn.Sequential(nn.Linear(d_in, cfg.d_hidden), nn.GELU(), nn.Dropout(cfg.dropout))
        self.msg_out = nn.Sequential(
            nn.Linear(cfg.d_hidden, cfg.d_z), nn.GELU(), nn.LayerNorm(cfg.d_z), nn.Linear(cfg.d_z, cfg.d_z)
        )
        self.gate = nn.Sequential(nn.Linear(cfg.d_hidden, cfg.d_gate), nn.GELU(), nn.Linear(cfg.d_gate, 1))
        self.attn = nn.Linear(cfg.d_hidden, 1)
        self.node_gate = nn.Sequential(
            nn.Linear(2 * cfg.d_z, cfg.d_z), nn.GELU(), nn.Linear(cfg.d_z, cfg.d_z)
        )
        self.wo = nn.Linear(cfg.d_z, cfg.d_z)
        mutable = _mutable_mask(cfg)
        self.register_buffer("mutable", mutable)
        self.mutable: Tensor
        self.norm = nn.LayerNorm(int(mutable.sum()))

    def forward(self, z: Tensor, mech: Tensor, b: GraphBatch) -> tuple[Tensor, Tensor]:
        n = z.shape[0]
        src, dst = b.edge_index[0], b.edge_index[1]
        x = torch.cat([z[dst], z[src], b.edge_attr, mech, b.delta_t.unsqueeze(-1)], dim=-1)
        h = self.msg_in(x)
        msg = self.msg_out(h)
        g = torch.sigmoid(self.gate(h))
        score = self.attn(h).squeeze(-1) + torch.log(b.reliability.clamp_min(1e-6))
        smax = torch.full((n,), -1e30, dtype=z.dtype).scatter_reduce(0, dst, score, "amax", include_self=True)
        w = torch.exp(score - smax[dst])
        denom = torch.zeros(n, dtype=z.dtype).index_add(0, dst, w)
        alpha = w / denom[dst].clamp_min(1e-12)
        agg = torch.zeros_like(z).index_add(0, dst, alpha.unsqueeze(-1) * g * msg)
        gate_i = torch.sigmoid(self.node_gate(torch.cat([z, agg], dim=-1)))
        delta = gate_i * self.wo(agg)
        update = (denom > 0) & ~b.observed
        m = self.mutable
        new_mut = self.norm(z[:, m] + delta[:, m])
        z_out = z.clone()
        z_out[:, m] = torch.where(update.unsqueeze(-1), new_mut, z[:, m])
        return z_out, g.squeeze(-1)


class LearnedTCDP(nn.Module):
    def __init__(self, cfg: LearnedTCDPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(nn.Linear(cfg.d_node_in, cfg.d_z), nn.GELU(), nn.LayerNorm(cfg.d_z))
        self.mechanisms = nn.Embedding(cfg.n_mechanisms, cfg.d_mech)
        self.layers = nn.ModuleList(TCDPLayer(cfg) for _ in range(cfg.n_layers))
        self.head = nn.Sequential(nn.Linear(cfg.d_z, cfg.d_head), nn.GELU(), nn.Linear(cfg.d_head, 4))

    def forward(self, b: GraphBatch) -> TCDPOutput:
        z = self.encoder(b.node_features)
        mech = self.mechanisms(b.mechanism)
        gates = []
        for layer in self.layers:
            z, g = layer(z, mech, b)
            gates.append(g)
        out = self.head(z)
        return TCDPOutput(z, out[:, :2], out[:, 2:].clamp(-12.0, 8.0), gates)


def tcdp_loss(out: TCDPOutput, b: GraphBatch, contamination_weight: float = 1.0) -> Tensor:
    """Masked heteroscedastic NLL + contamination penalty on misleading-edge (healthy) nodes."""
    if b.target is None or b.target_mask is None:
        raise ValueError("training batch needs target and target_mask")
    mask = b.target_mask.to(out.mean.dtype)
    nll = 0.5 * (out.log_var + (out.mean - b.target) ** 2 * torch.exp(-out.log_var))
    loss = (nll * mask).sum() / mask.sum().clamp_min(1.0)
    if b.misleading is not None and bool(b.misleading.any()):
        over = F.relu(out.mean[b.misleading] - b.target[b.misleading]) * mask[b.misleading]
        loss = loss + contamination_weight * (over**2).mean()
    return loss


def train_learned_tcdp(
    model: LearnedTCDP,
    batches: Sequence[GraphBatch],
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    contamination_weight: float = 1.0,
    seed: int = 0,
) -> list[float]:
    """Small AdamW loop; returns the mean loss per epoch."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(batches), generator=gen).tolist()
        total = 0.0
        for i in order:
            opt.zero_grad()
            loss = tcdp_loss(model(batches[i]), batches[i], contamination_weight)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        history.append(total / max(len(batches), 1))
    model.eval()
    return history
