"""Sonar ViT-S/14 encoder for OS-FM U1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SonarEncoderConfig:
    image_size: int = 28
    patch_size: int = 14
    in_channels: int = 1
    width: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: int = 4
    dense_tap_layers: tuple[int, ...] = (3, 6, 9, 12)
    pretrained_source: str = "none"
    initialization: str = "deterministic_random_smoke"

    @property
    def grid_size(self) -> int:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size


@dataclass(frozen=True)
class SonarEncoderOutput:
    modality_repr: torch.Tensor
    patch_tokens: torch.Tensor
    dense_taps: dict[int, torch.Tensor]
    visible_mask: torch.Tensor
    masked_token_mask: torch.Tensor


class SonarViTS14Encoder(nn.Module):
    """ViT-S/14-style single-channel rendered-sonar encoder with OS-FM D_F=384 output."""

    d_model = 384

    def __init__(self, config: SonarEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or SonarEncoderConfig()
        if self.config.width != self.d_model:
            raise ValueError("P2 freezes sonar width / D_F at 384")
        self.patch_embed = nn.Conv2d(
            self.config.in_channels,
            self.config.width,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.width))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.config.num_patches + 1, self.config.width))
        block = nn.TransformerEncoderLayer(
            d_model=self.config.width,
            nhead=self.config.heads,
            dim_feedforward=self.config.width * self.config.mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.ModuleList(
            [
                block
                if idx == 0
                else type(block)(
                    d_model=self.config.width,
                    nhead=self.config.heads,
                    dim_feedforward=self.config.width * self.config.mlp_ratio,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for idx in range(self.config.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.config.width)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)

    def forward(self, sonar: torch.Tensor, masked_token_mask: torch.Tensor | None = None) -> SonarEncoderOutput:
        if sonar.ndim != 4 or sonar.shape[1] != self.config.in_channels:
            raise ValueError("sonar input must be [B, 1, H, W]")
        if sonar.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError(f"sonar input must be {self.config.image_size}x{self.config.image_size}")
        patches = self.patch_embed(sonar).flatten(2).transpose(1, 2)
        if masked_token_mask is None:
            masked_token_mask = torch.zeros(patches.shape[:2], dtype=torch.bool, device=patches.device)
        if masked_token_mask.shape != patches.shape[:2]:
            raise ValueError("masked_token_mask must match [B, num_patches]")
        visible_mask = ~masked_token_mask
        tokens = torch.cat([self.cls_token.expand(sonar.shape[0], -1, -1), patches], dim=1)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype, device=tokens.device)
        taps: dict[int, torch.Tensor] = {}
        for layer_idx, block in enumerate(self.blocks, start=1):
            tokens = block(tokens)
            if layer_idx in self.config.dense_tap_layers:
                taps[layer_idx] = self.norm(tokens[:, 1:])
        tokens = self.norm(tokens)
        return SonarEncoderOutput(
            modality_repr=tokens[:, 0],
            patch_tokens=tokens[:, 1:],
            dense_taps=taps,
            visible_mask=visible_mask,
            masked_token_mask=masked_token_mask,
        )
