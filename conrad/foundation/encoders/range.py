"""Validity-aware metric range/depth encoder for OS-FM U1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RangeEncoderConfig:
    image_size: int = 32
    patch_size: int = 8
    in_channels: int = 1
    width: int = 256
    output_dim: int = 384
    depth: int = 6
    heads: int = 4
    mlp_ratio: int = 4
    dense_tap_layers: tuple[int, ...] = (2, 4, 6)
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
class RangeEncoderOutput:
    modality_repr: torch.Tensor
    patch_tokens: torch.Tensor
    internal_patch_tokens: torch.Tensor
    dense_taps: dict[int, torch.Tensor]
    dense_taps_projected: dict[int, torch.Tensor]
    patch_valid_fraction: torch.Tensor
    visible_mask: torch.Tensor
    masked_token_mask: torch.Tensor


class RangeViTP8Encoder(nn.Module):
    """Patch-8 range/depth ViT with internal width 256 and final D_F=384 projection."""

    d_model = 384
    internal_width = 256

    def __init__(self, config: RangeEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or RangeEncoderConfig()
        if self.config.patch_size != 8:
            raise ValueError("P2 freezes range patch size at 8")
        if self.config.width != self.internal_width:
            raise ValueError("P2 freezes range internal width at 256")
        if self.config.output_dim != self.d_model:
            raise ValueError("P2 freezes OS-FM D_F at 384")
        self.patch_embed = nn.Conv2d(
            self.config.in_channels + 1,
            self.config.width,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.width))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.config.num_patches + 1, self.config.width))
        self.validity_embed = nn.Linear(1, self.config.width)
        self.output_projection = nn.Linear(self.config.width, self.config.output_dim)
        self.output_norm = nn.LayerNorm(self.config.output_dim)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.config.width,
                    nhead=self.config.heads,
                    dim_feedforward=self.config.width * self.config.mlp_ratio,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.config.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.config.width)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.validity_embed.weight, std=0.02)
        nn.init.zeros_(self.validity_embed.bias)
        nn.init.trunc_normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)

    def _patch_valid_fraction(self, validity_mask: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.avg_pool2d(
            validity_mask.float(),
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        return pooled.flatten(2).transpose(1, 2)

    def forward(
        self,
        range_raster: torch.Tensor,
        validity_mask: torch.Tensor,
        masked_token_mask: torch.Tensor | None = None,
    ) -> RangeEncoderOutput:
        if range_raster.ndim != 4 or range_raster.shape[1] != self.config.in_channels:
            raise ValueError("range input must be [B, 1, H, W]")
        if validity_mask.shape != range_raster.shape:
            raise ValueError("validity_mask must match range input shape [B, 1, H, W]")
        if range_raster.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError(f"range input must be {self.config.image_size}x{self.config.image_size}")
        clean_range = range_raster.masked_fill(~validity_mask.bool(), 0.0)
        valid_fraction = self._patch_valid_fraction(validity_mask)
        patch_input = torch.cat([clean_range, validity_mask.float()], dim=1)
        patches = self.patch_embed(patch_input).flatten(2).transpose(1, 2)
        patches = patches + self.validity_embed(valid_fraction)
        if masked_token_mask is None:
            masked_token_mask = torch.zeros(patches.shape[:2], dtype=torch.bool, device=patches.device)
        if masked_token_mask.shape != patches.shape[:2]:
            raise ValueError("masked_token_mask must match [B, num_patches]")
        visible_mask = ~masked_token_mask
        tokens = torch.cat([self.cls_token.expand(range_raster.shape[0], -1, -1), patches], dim=1)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype, device=tokens.device)
        taps: dict[int, torch.Tensor] = {}
        taps_projected: dict[int, torch.Tensor] = {}
        for layer_idx, block in enumerate(self.blocks, start=1):
            tokens = block(tokens)
            if layer_idx in self.config.dense_tap_layers:
                internal = self.norm(tokens[:, 1:])
                taps[layer_idx] = internal
                taps_projected[layer_idx] = self.output_norm(self.output_projection(internal))
        tokens = self.norm(tokens)
        projected_tokens = self.output_norm(self.output_projection(tokens))
        return RangeEncoderOutput(
            modality_repr=projected_tokens[:, 0],
            patch_tokens=projected_tokens[:, 1:],
            internal_patch_tokens=tokens[:, 1:],
            dense_taps=taps,
            dense_taps_projected=taps_projected,
            patch_valid_fraction=valid_fraction.squeeze(-1),
            visible_mask=visible_mask,
            masked_token_mask=masked_token_mask,
        )
