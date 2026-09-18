"""Conrad-owned self-supervised objective L_repr (ch33): view contrast + temporal + cross-modal +
masked reconstruction. The projection head exists only for pretraining and is discarded afterwards.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from conrad.core.config import CoreConfig
from conrad.core.ecmer.fusion import MODALITIES
from conrad.core.primitives import masked_loss, mlp


def info_nce(a: Tensor, b: Tensor, temperature: float, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Symmetric InfoNCE between row-aligned positives ``a[i] <-> b[i]``; other rows are negatives.

    ``mask`` [B] marks rows with a valid positive pair; invalid rows neither anchor nor act as negatives.
    """
    valid = torch.ones(a.shape[0], dtype=torch.bool) if mask is None else mask.to(torch.bool)
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    logits = logits.masked_fill(~valid.unsqueeze(0), torch.finfo(logits.dtype).min)
    target = torch.arange(a.shape[0])
    per_row = 0.5 * (
        F.cross_entropy(logits, target, reduction="none")
        + F.cross_entropy(logits.T, target, reduction="none")
    )
    per_row = torch.where(valid, per_row, torch.zeros_like(per_row))
    return masked_loss(per_row, valid)


class ProjectionHead(nn.Module):
    """De -> 1024 -> De. PRETRAINING ONLY; never part of deployed inference."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.net = mlp([cfg.evidence_dim, cfg.ecmer.projection_hidden, cfg.evidence_dim])

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        return out


@dataclass(frozen=True)
class ReprLoss:
    total: Tensor
    view_contrast: Tensor
    temporal: Tensor
    cross_modal: Tensor
    masked_reconstruction: Tensor
    denominators: dict[str, float]


class ReprObjective(nn.Module):
    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.projection = ProjectionHead(cfg)
        self.decoder = nn.Linear(cfg.evidence_dim, len(MODALITIES) * cfg.evidence_dim)

    def forward(
        self,
        view_a: Tensor,
        view_b: Tensor,
        temporal_next: Tensor | None = None,
        temporal_mask: Tensor | None = None,
        modality_a: Tensor | None = None,
        modality_b: Tensor | None = None,
        cross_mask: Tensor | None = None,
        event_masked: Tensor | None = None,
        target_tokens: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> ReprLoss:
        """All embeddings are [B, De]. ``target_tokens`` [B, n_mod, De] are detached pre-fusion tokens of
        modalities hidden from ``event_masked``; ``target_mask`` [B, n_mod] marks which were hidden."""
        t = self.cfg.ecmer.info_nce_temperature
        zero = view_a.sum() * 0.0
        p = self.projection
        view, n_view = info_nce(p(view_a), p(view_b), t)
        temp, n_temp = (
            (zero, torch.zeros(()))
            if temporal_next is None
            else info_nce(p(view_a), p(temporal_next), t, temporal_mask)
        )
        cross, n_cross = (
            (zero, torch.zeros(()))
            if modality_a is None or modality_b is None
            else info_nce(p(modality_a), p(modality_b), t, cross_mask)
        )
        if event_masked is not None and target_tokens is not None and target_mask is not None:
            recon = self.decoder(event_masked).view_as(target_tokens)
            err = ((recon - target_tokens.detach()) ** 2).mean(-1)
            rec, n_rec = masked_loss(err, target_mask)
        else:
            rec, n_rec = zero, torch.zeros(())
        total = view + temp + cross + rec
        dens = {
            "view_contrast": float(n_view),
            "temporal": float(n_temp),
            "cross_modal": float(n_cross),
            "masked_reconstruction": float(n_rec),
        }
        return ReprLoss(total, view, temp, cross, rec, dens)


def gaussian_blur(images: Tensor, sigma: Tensor) -> Tensor:
    """Per-image separable Gaussian blur; ``sigma`` [B] in pixels (0 = unchanged)."""
    radius = int(max(1, torch.ceil(3 * sigma.max()).item()))
    xs = torch.arange(-radius, radius + 1, dtype=images.dtype)
    k = torch.exp(-(xs[None] ** 2) / (2 * sigma.clamp_min(1e-3)[:, None] ** 2))
    k = k / k.sum(-1, keepdim=True)
    b, c, h, w = images.shape
    x = images.reshape(1, b * c, h, w)
    kh = k.repeat_interleave(c, 0).view(b * c, 1, 1, -1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), kh, groups=b * c)
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), kh.transpose(2, 3), groups=b * c)
    out = x.view(b, c, h, w)
    return torch.where((sigma > 0)[:, None, None, None], out, images)


def corrupt_images(images: Tensor, severity: Tensor, generator: torch.Generator) -> Tensor:
    """Known-severity degradation (stage E2/E3): blur + noise + turbidity haze + occlusion patch."""
    s = severity.clamp(0, 1)
    out = gaussian_blur(images, 3.0 * s)
    noise = torch.randn(images.shape, generator=generator) * (0.2 * s)[:, None, None, None]
    haze = 0.5 * s[:, None, None, None]
    out = (1 - haze) * (out + noise) + haze * 0.5
    b, _, h, w = images.shape
    side = (s * 0.5 * min(h, w)).long()
    for i in range(b):
        if int(side[i]) > 0:
            y = int(torch.randint(0, h - int(side[i]) + 1, (1,), generator=generator))
            x = int(torch.randint(0, w - int(side[i]) + 1, (1,), generator=generator))
            out[i, :, y : y + int(side[i]), x : x + int(side[i])] = 0.0
    return out.clamp(0, 1)
