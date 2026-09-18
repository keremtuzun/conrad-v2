"""Learned CEFD losses and a small AdamW training loop. Batches come from the caller (Twin2E in the
experiments); ``synthetic_batch`` is a SYNTHETIC_ONLY smoke generator with a planted coupling.

L = l_E * NLL(entity) + l_Phi * (NLL(field) + l_g * gradient) + l_C * BCE(mean gate, coupled label)

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from conrad.domains.ecological.learned.config import LearnedCEFDConfig
from conrad.domains.ecological.learned.model import CEFDOutput, LearnedCEFD


@dataclass
class CEFDBatch:
    entity_features: Tensor  # (B, N, F)
    relations: Tensor  # (B, N, N) long
    entity_mask: Tensor  # (B, N) bool
    positions: Tensor  # (B, N, 3) in [-1, 1]
    entity_uncertainty: Tensor  # (B, N, 4)
    field_state: Tensor  # (B, C, X, Y, Z)
    entity_target: Tensor  # (B, N, 2): cover, condition
    entity_target_mask: Tensor  # (B, N, 2) bool
    field_target: Tensor  # (B, C, X, Y, Z)
    field_target_mask: Tensor  # (B, C, X, Y, Z) bool
    coupled: Tensor | None = None  # (B,) 1 = environment drives ecology here, 0 = it does not


def gaussian_nll(mu: Tensor, logvar: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    logvar = logvar.clamp(-10.0, 10.0)
    nll = 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar))
    m = mask.to(nll.dtype)
    return (nll * m).sum() / m.sum().clamp(min=1.0)


def _gradient_loss(mu: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    terms = []
    for dim in (2, 3, 4):
        size = mu.shape[dim]
        if size < 2:
            continue
        m = (mask.narrow(dim, 1, size - 1) & mask.narrow(dim, 0, size - 1)).to(mu.dtype)
        terms.append(((mu.diff(dim=dim) - y.diff(dim=dim)) ** 2 * m).sum() / m.sum().clamp(min=1.0))
    return torch.stack(terms).mean() if terms else mu.new_zeros(())


def cefd_loss(out: CEFDOutput, batch: CEFDBatch, cfg: LearnedCEFDConfig) -> dict[str, Tensor]:
    c = cfg.field_in_channels
    e_mu = torch.sigmoid(out.entity_out[..., 0:4:2])
    e_lv = out.entity_out[..., 1:4:2]
    l_ent = gaussian_nll(e_mu, e_lv, batch.entity_target, batch.entity_target_mask)
    f_mu, f_lv = out.field_out[:, :c], out.field_out[:, c : 2 * c]
    l_fld = gaussian_nll(f_mu, f_lv, batch.field_target, batch.field_target_mask)
    l_grad = _gradient_loss(f_mu, batch.field_target, batch.field_target_mask)
    l_cpl = f_mu.new_zeros(())
    if batch.coupled is not None:
        m = batch.entity_mask.to(out.gate.dtype)
        g = (out.gate[..., 0] * m).sum(1) / m.sum(1).clamp(min=1.0)
        l_cpl = F.binary_cross_entropy(g.clamp(1e-6, 1 - 1e-6), batch.coupled.to(g.dtype))
    total = (
        cfg.lambda_entity * l_ent
        + cfg.lambda_field * (l_fld + cfg.lambda_gradient * l_grad)
        + cfg.lambda_coupling * l_cpl
    )
    return {"total": total, "entity": l_ent, "field": l_fld, "gradient": l_grad, "coupling": l_cpl}


def run_forward(model: LearnedCEFD, b: CEFDBatch) -> CEFDOutput:
    out: CEFDOutput = model(
        b.entity_features, b.relations, b.entity_mask, b.positions, b.entity_uncertainty, b.field_state
    )
    return out


def train_learned_cefd(
    model: LearnedCEFD,
    batch_fn: Callable[[torch.Generator], CEFDBatch],
    steps: int,
    generator: torch.Generator,
) -> list[float]:
    """AdamW with gradient clipping; returns the per-step total loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=model.cfg.lr, weight_decay=model.cfg.weight_decay)
    model.train()
    history = []
    for _ in range(steps):
        batch = batch_fn(generator)
        loss = cefd_loss(run_forward(model, batch), batch, model.cfg)["total"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(float(loss.detach()))
    return history


def synthetic_batch(
    cfg: LearnedCEFDConfig,
    g: torch.Generator,
    batch: int = 2,
    n: int = 3,
    grid: tuple[int, int, int] = (4, 4, 4),
) -> CEFDBatch:
    """SYNTHETIC_ONLY smoke data: when coupled, condition falls with field channel 0 at the entity."""
    c = cfg.field_in_channels
    field = torch.randn(batch, c, *grid, generator=g)
    pos = torch.rand(batch, n, 3, generator=g) * 2 - 1
    coupled = (torch.rand(batch, generator=g) > 0.5).float()
    idx = torch.round((pos + 1) * 0.5 * (torch.tensor(grid) - 1)).long()
    temp = torch.stack([field[i, 0, idx[i, :, 0], idx[i, :, 1], idx[i, :, 2]] for i in range(batch)])
    cover = torch.rand(batch, n, generator=g)
    condition = torch.where(coupled[:, None] > 0, torch.sigmoid(-temp), torch.full_like(temp, 0.8))
    feats = torch.randn(batch, n, cfg.entity_feature_dim, generator=g)
    feats[..., 0] = cover
    return CEFDBatch(
        entity_features=feats,
        relations=torch.randint(0, cfg.relation_types + 1, (batch, n, n), generator=g),
        entity_mask=torch.ones(batch, n, dtype=torch.bool),
        positions=pos,
        entity_uncertainty=torch.rand(batch, n, cfg.uncertainty_in, generator=g),
        field_state=field + 0.1 * torch.randn(batch, c, *grid, generator=g),
        entity_target=torch.stack([cover, condition], dim=-1),
        entity_target_mask=torch.ones(batch, n, 2, dtype=torch.bool),
        field_target=field,
        field_target_mask=torch.rand(batch, c, *grid, generator=g) > 0.3,
        coupled=coupled,
    )
