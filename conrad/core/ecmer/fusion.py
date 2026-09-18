"""ECMER event fusion: modality tokens + [EVENT] token -> transformer -> e_k + quality head (ch33).

Missing modalities are normal input: an absent modality contributes masked tokens only.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from conrad.core.config import CoreConfig
from conrad.core.ecmer.encoders import (
    PoseTimeContextEncoder,
    ResNet18Encoder,
    ScalarSensorEncoder,
    VoxelPointEncoder,
)
from conrad.core.primitives import PreNormTransformerBlock, masked_mean, mlp

MODALITIES: tuple[str, ...] = ("rgb", "sonar", "points", "scalar")


@dataclass
class EventBatch:
    """One observation EVENT per row. Every modality is optional; ``*_avail`` [B] marks presence."""

    batch_size: int
    context_raw: Tensor  # [B, 14]
    sensor_idx: Tensor  # [B] long
    frame_idx: Tensor  # [B] long
    raw_quality: Tensor  # [B, Q]
    rgb: Tensor | None = None
    sonar: Tensor | None = None
    points: Tensor | None = None
    points_mask: Tensor | None = None
    scalars: Tensor | None = None
    scalar_present: Tensor | None = None
    avail: dict[str, Tensor] = field(default_factory=dict)

    def available(self, name: str) -> Tensor:
        a = self.avail.get(name)
        return torch.zeros(self.batch_size, dtype=torch.bool) if a is None else a.to(torch.bool)


@dataclass(frozen=True)
class EcmerOutput:
    event: Tensor  # [B, De] e_k
    reliability: Tensor  # [B]
    log_var_a: Tensor  # [B] clipped to [log_var_min, log_var_max]
    ood_logit: Tensor  # [B]
    usable_logit: Tensor  # [B]
    tokens: Tensor  # [B, L, De] fused tokens (index 0 = [EVENT])
    token_mask: Tensor  # [B, L]
    modality_tokens: dict[str, Tensor]  # pre-fusion global token per modality [B, De]
    availability: Tensor  # [B, n_modalities] float


class EcmerModel(nn.Module):
    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        e, de = cfg.ecmer, cfg.evidence_dim
        self.rgb = ResNet18Encoder(cfg, e.image_channels_rgb)
        self.sonar = ResNet18Encoder(cfg, e.image_channels_sonar)  # separate weights, no sharing
        self.points = VoxelPointEncoder(cfg)
        self.scalar = ScalarSensorEncoder(cfg)
        self.context = PoseTimeContextEncoder(cfg)
        self.event_token = nn.Parameter(torch.zeros(de))
        nn.init.normal_(self.event_token, std=0.02)
        self.modality_embedding = nn.Embedding(len(MODALITIES) + 1, de)
        self.token_kind = nn.Embedding(2, de)  # 0 global, 1 patch
        self.blocks = nn.ModuleList(
            PreNormTransformerBlock(de, cfg.heads, cfg.ffn_hidden, cfg.dropout)
            for _ in range(e.fusion_layers)
        )
        self.final_norm = nn.LayerNorm(de)
        q_in = de + len(MODALITIES) + e.quality_feature_dim + 1
        h1, h2 = e.quality_hidden
        self.quality_head = mlp([q_in, h1, h2, 4])

    def _encode(self, name: str, batch: EventBatch) -> Tensor | None:
        if name == "rgb" and batch.rgb is not None:
            return self.rgb(batch.rgb)
        if name == "sonar" and batch.sonar is not None:
            return self.sonar(batch.sonar)
        if name == "points" and batch.points is not None:
            pm = batch.points_mask if batch.points_mask is not None else torch.ones(batch.points.shape[:2])
            return self.points(batch.points, pm)
        if name == "scalar" and batch.scalars is not None:
            present = batch.scalar_present
            present = torch.isfinite(batch.scalars) if present is None else present
            return self.scalar(batch.scalars, present)
        return None

    def forward(self, batch: EventBatch, drop_modalities: Tensor | None = None) -> EcmerOutput:
        """``drop_modalities`` [B, n_mod] bool: modality dropout (training stage D) or ablation."""
        e = self.cfg.ecmer
        b = batch.batch_size
        ctx = self.context(batch.context_raw, batch.sensor_idx, batch.frame_idx)
        if not e.use_alignment:
            ctx = torch.zeros_like(ctx)
        tokens: list[Tensor] = [self.event_token.expand(b, 1, -1) + self.modality_embedding.weight[0]]
        masks: list[Tensor] = [torch.ones(b, 1, dtype=torch.bool)]
        globals_: dict[str, Tensor] = {}
        avail_cols: list[Tensor] = []
        for m_idx, name in enumerate(MODALITIES, start=1):
            avail = batch.available(name)
            if drop_modalities is not None:
                avail = avail & ~drop_modalities[:, m_idx - 1]
            avail_cols.append(avail.to(torch.float32))
            tok = self._encode(name, batch)
            if tok is None:
                continue
            kind = torch.ones(tok.shape[1], dtype=torch.long)
            kind[0] = 0
            tok = tok + self.token_kind(kind) + ctx.unsqueeze(1)
            if e.use_modality_id:
                tok = tok + self.modality_embedding.weight[m_idx]
            tok = tok * avail[:, None, None].to(tok.dtype)
            globals_[name] = tok[:, 0]
            tokens.append(tok)
            masks.append(avail.unsqueeze(1).expand(-1, tok.shape[1]))
        x, mask = torch.cat(tokens, dim=1), torch.cat(masks, dim=1)
        for block in self.blocks:
            x = block(x, key_padding_mask=mask)
        x = self.final_norm(x) * mask.unsqueeze(-1).to(x.dtype)
        event = x[:, 0]
        availability = torch.stack(avail_cols, dim=-1)
        disagreement = (
            self._disagreement(globals_, availability) if e.use_disagreement else event.new_zeros(b, 1)
        )
        q_in = torch.cat([event, availability, batch.raw_quality, disagreement], dim=-1)
        rel, log_var, ood, usable = self.quality_head(q_in).unbind(-1)
        reliability = torch.sigmoid(rel) if e.use_reliability else torch.ones_like(rel)
        return EcmerOutput(
            event=event,
            reliability=reliability,
            log_var_a=log_var.clamp(e.log_var_min, e.log_var_max),
            ood_logit=ood,
            usable_logit=usable,
            tokens=x,
            token_mask=mask,
            modality_tokens=globals_,
            availability=availability,
        )

    @staticmethod
    def _disagreement(globals_: dict[str, Tensor], availability: Tensor) -> Tensor:
        """Spread of per-modality global tokens around their mean (0 with < 2 modalities)."""
        if not globals_:
            return availability.new_zeros(availability.shape[0], 1)
        names = [n for n in MODALITIES if n in globals_]
        stack = torch.stack([globals_[n] for n in names], dim=1)
        mask = torch.stack([availability[:, MODALITIES.index(n)] > 0 for n in names], dim=1)
        centre = masked_mean(stack, mask)
        dev = ((stack - centre.unsqueeze(1)) ** 2).mean(-1)
        spread = masked_mean(dev.unsqueeze(-1), mask)
        return spread * (mask.sum(1, keepdim=True) >= 2).to(spread.dtype)


def sample_modality_dropout(avail: Tensor, p: float, generator: torch.Generator) -> Tensor:
    """Drop each available modality with prob ``p`` but never all of them in a row (stage D)."""
    drop = (torch.rand(avail.shape, generator=generator) < p) & avail
    all_gone = (avail & ~drop).sum(-1) == 0
    return drop & ~all_gone.unsqueeze(-1)
