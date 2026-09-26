"""Perceiver-style scene fusion for OS-FM M1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SceneFusionConfig:
    d_f: int = 384
    num_scene_latents: int = 64
    cross_attention_blocks: int = 3
    latent_self_attention_blocks: int = 6
    heads: int = 6
    mlp_ratio: int = 4
    modalities: tuple[str, ...] = ("rgb", "sonar", "range", "geometry", "context")


@dataclass(frozen=True)
class ModalityTokenSet:
    modality: str
    tokens: torch.Tensor
    valid_token_mask: torch.Tensor
    naturally_missing: torch.Tensor
    artificially_dropped: torch.Tensor
    padded: torch.Tensor


@dataclass(frozen=True)
class SceneFusionOutput:
    scene_latents: torch.Tensor
    global_repr: torch.Tensor
    fused_token_mask: torch.Tensor
    modality_presence: torch.Tensor
    natural_missing_mask: torch.Tensor
    artificial_dropout_mask: torch.Tensor
    padding_mask: torch.Tensor
    modality_names: tuple[str, ...]


class CrossAttentionBlock(nn.Module):
    def __init__(self, config: SceneFusionConfig) -> None:
        super().__init__()
        self.latent_norm = nn.LayerNorm(config.d_f)
        self.context_norm = nn.LayerNorm(config.d_f)
        self.attn = nn.MultiheadAttention(config.d_f, config.heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(config.d_f),
            nn.Linear(config.d_f, config.d_f * config.mlp_ratio),
            nn.GELU(),
            nn.Linear(config.d_f * config.mlp_ratio, config.d_f),
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        query = self.latent_norm(latents)
        key_value = self.context_norm(context)
        attended, _ = self.attn(query, key_value, key_value, key_padding_mask=key_padding_mask, need_weights=False)
        latents = latents + attended
        return latents + self.ffn(latents)


class SelfAttentionBlock(nn.Module):
    def __init__(self, config: SceneFusionConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_f)
        self.attn = nn.MultiheadAttention(config.d_f, config.heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(config.d_f),
            nn.Linear(config.d_f, config.d_f * config.mlp_ratio),
            nn.GELU(),
            nn.Linear(config.d_f * config.mlp_ratio, config.d_f),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(latents)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        latents = latents + attended
        return latents + self.ffn(latents)


class SceneFusionTransformer(nn.Module):
    """M1 scene-latent fusion with no anchor modality."""

    def __init__(self, config: SceneFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or SceneFusionConfig()
        if self.config.d_f != 384:
            raise ValueError("P2 freezes shared foundation dimension D_F=384")
        if self.config.num_scene_latents != 64:
            raise ValueError("P2 freezes M1 scene latents at 64")
        if self.config.cross_attention_blocks != 3:
            raise ValueError("P2 freezes M1 cross-attention depth at 3")
        if self.config.latent_self_attention_blocks != 6:
            raise ValueError("P2 freezes M1 latent self-attention depth at 6")
        if self.config.heads != 6:
            raise ValueError("P2 freezes M1 attention heads at 6")
        self.scene_latents = nn.Parameter(torch.zeros(1, self.config.num_scene_latents, self.config.d_f))
        self.modality_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1, 1, self.config.d_f)) for name in self.config.modalities}
        )
        self.cross_blocks = nn.ModuleList([CrossAttentionBlock(self.config) for _ in range(self.config.cross_attention_blocks)])
        self.self_blocks = nn.ModuleList(
            [SelfAttentionBlock(self.config) for _ in range(self.config.latent_self_attention_blocks)]
        )
        self.output_norm = nn.LayerNorm(self.config.d_f)
        self.global_norm = nn.LayerNorm(self.config.d_f)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.scene_latents, std=0.02)
        for param in self.modality_embeddings.values():
            nn.init.trunc_normal_(param, std=0.02)

    def forward(self, modality_sets: tuple[ModalityTokenSet, ...]) -> SceneFusionOutput:
        if not modality_sets:
            raise ValueError("at least one observed modality is required")
        batch = modality_sets[0].tokens.shape[0]
        device = modality_sets[0].tokens.device
        context_parts: list[torch.Tensor] = []
        padding_parts: list[torch.Tensor] = []
        names: list[str] = []
        presence: list[torch.Tensor] = []
        natural: list[torch.Tensor] = []
        artificial: list[torch.Tensor] = []
        padded: list[torch.Tensor] = []
        for token_set in modality_sets:
            if token_set.modality not in self.modality_embeddings:
                raise ValueError(f"unknown modality {token_set.modality!r}")
            if token_set.tokens.ndim != 3 or token_set.tokens.shape[-1] != self.config.d_f:
                raise ValueError("modality tokens must be [B, T, 384]")
            if token_set.tokens.shape[0] != batch:
                raise ValueError("all modality token sets must share batch size")
            valid = token_set.valid_token_mask.bool()
            if valid.shape != token_set.tokens.shape[:2]:
                raise ValueError("valid_token_mask must match [B, T]")
            unavailable = token_set.naturally_missing | token_set.artificially_dropped | token_set.padded
            observed = (~unavailable) & valid.any(dim=1)
            context_parts.append(token_set.tokens + self.modality_embeddings[token_set.modality])
            padding_parts.append(~valid | unavailable.unsqueeze(1))
            names.append(token_set.modality)
            presence.append(observed)
            natural.append(token_set.naturally_missing)
            artificial.append(token_set.artificially_dropped)
            padded.append(token_set.padded)
        key_padding_mask = torch.cat(padding_parts, dim=1)
        if bool(key_padding_mask.all(dim=1).any().detach().cpu()):
            raise ValueError("each sample must retain at least one valid scene modality")
        context = torch.cat(context_parts, dim=1)
        latents = self.scene_latents.expand(batch, -1, -1).to(device=device, dtype=context.dtype)
        for block in self.cross_blocks:
            latents = block(latents, context, key_padding_mask)
        for block in self.self_blocks:
            latents = block(latents)
        latents = self.output_norm(latents)
        return SceneFusionOutput(
            scene_latents=latents,
            global_repr=self.global_norm(latents.mean(dim=1)),
            fused_token_mask=~key_padding_mask,
            modality_presence=torch.stack(presence, dim=1),
            natural_missing_mask=torch.stack(natural, dim=1),
            artificial_dropout_mask=torch.stack(artificial, dim=1),
            padding_mask=torch.stack(padded, dim=1),
            modality_names=tuple(names),
        )
