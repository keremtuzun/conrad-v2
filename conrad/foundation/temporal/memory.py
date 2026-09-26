"""Short-horizon OS-FM temporal memory.

This module is the P2/P3 T1 memory layer after M1 scene fusion. It is intentionally
bounded and sequence scoped; it is not persistent Model2 state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalMemoryConfig:
    d_f: int = 384
    memory_tokens: int = 16
    transformer_blocks: int = 4
    heads: int = 6
    mlp_ratio: int = 4
    max_windows: int = 10
    gap_reset_s: float = 1.0


@dataclass(frozen=True)
class TemporalMemoryOutput:
    window_repr: torch.Tensor
    memory_states: torch.Tensor
    temporal_valid_mask: torch.Tensor
    adjacent_eligible_mask: torch.Tensor
    reset_mask: torch.Tensor
    gap_reset_mask: torch.Tensor
    history_lengths: torch.Tensor


class TemporalBlock(nn.Module):
    def __init__(self, config: TemporalMemoryConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_f)
        self.attn = nn.MultiheadAttention(config.d_f, config.heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(config.d_f),
            nn.Linear(config.d_f, config.d_f * config.mlp_ratio),
            nn.GELU(),
            nn.Linear(config.d_f * config.mlp_ratio, config.d_f),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.ffn(x)


class TemporalMemoryTransformer(nn.Module):
    """Bounded 10-window transformer memory over fused M1 global representations."""

    def __init__(self, config: TemporalMemoryConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalMemoryConfig()
        if self.config.d_f != 384:
            raise ValueError("P2 freezes T1 width D_F=384")
        if self.config.memory_tokens != 16:
            raise ValueError("P2 freezes T1 memory tokens at 16")
        if self.config.transformer_blocks != 4:
            raise ValueError("P2 freezes T1 transformer depth at 4")
        if self.config.heads != 6:
            raise ValueError("P2 freezes T1 attention heads at 6")
        if self.config.max_windows != 10:
            raise ValueError("P3 baseline freezes bounded history at 10 windows")
        self.memory_tokens = nn.Parameter(torch.zeros(1, self.config.memory_tokens, self.config.d_f))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.config.max_windows, self.config.d_f))
        self.blocks = nn.ModuleList([TemporalBlock(self.config) for _ in range(self.config.transformer_blocks)])
        self.output_norm = nn.LayerNorm(self.config.d_f)
        self.window_norm = nn.LayerNorm(self.config.d_f)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.memory_tokens, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def _masks(
        self,
        valid_mask: torch.Tensor,
        timestamps_s: torch.Tensor,
        boundary_reset_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps = valid_mask.shape
        reset = torch.zeros(batch, steps, dtype=torch.bool, device=valid_mask.device)
        gap_reset = torch.zeros_like(reset)
        eligible = torch.zeros_like(reset)
        reset[:, 0] = valid_mask[:, 0]
        reset = reset | (boundary_reset_mask & valid_mask)
        for idx in range(1, steps):
            previous_valid = valid_mask[:, idx - 1]
            current_valid = valid_mask[:, idx]
            gap = timestamps_s[:, idx] - timestamps_s[:, idx - 1]
            gap_reset[:, idx] = current_valid & previous_valid & (gap > self.config.gap_reset_s)
            reset[:, idx] = reset[:, idx] | gap_reset[:, idx] | (current_valid & ~previous_valid)
            eligible[:, idx] = current_valid & previous_valid & ~gap_reset[:, idx] & ~boundary_reset_mask[:, idx]
        return reset, gap_reset, eligible, valid_mask.bool()

    def forward(
        self,
        fused_global: torch.Tensor,
        timestamps_s: torch.Tensor,
        valid_mask: torch.Tensor,
        boundary_reset_mask: torch.Tensor | None = None,
    ) -> TemporalMemoryOutput:
        if fused_global.ndim != 3 or fused_global.shape[-1] != self.config.d_f:
            raise ValueError("fused_global must be [B, T, 384]")
        batch, steps, width = fused_global.shape
        if steps > self.config.max_windows:
            raise ValueError("T1 smoke only accepts bounded <=10-window sequences")
        if timestamps_s.shape != (batch, steps) or valid_mask.shape != (batch, steps):
            raise ValueError("timestamps_s and valid_mask must match [B, T]")
        boundary = (
            torch.zeros(batch, steps, dtype=torch.bool, device=fused_global.device)
            if boundary_reset_mask is None
            else boundary_reset_mask.to(device=fused_global.device, dtype=torch.bool)
        )
        reset, gap_reset, eligible, valid = self._masks(valid_mask.bool(), timestamps_s, boundary)
        padded_steps = self.config.max_windows - steps
        if padded_steps:
            pad = torch.zeros(batch, padded_steps, width, dtype=fused_global.dtype, device=fused_global.device)
            tokens = torch.cat([fused_global, pad], dim=1)
            token_valid = torch.cat(
                [valid, torch.zeros(batch, padded_steps, dtype=torch.bool, device=fused_global.device)], dim=1
            )
        else:
            tokens = fused_global
            token_valid = valid
        tokens = tokens + self.position_embedding[:, : self.config.max_windows].to(dtype=fused_global.dtype)
        memory = self.memory_tokens.expand(batch, -1, -1).to(dtype=fused_global.dtype, device=fused_global.device)
        x = torch.cat([memory, tokens], dim=1)
        key_padding = torch.cat(
            [
                torch.zeros(batch, self.config.memory_tokens, dtype=torch.bool, device=fused_global.device),
                ~token_valid,
            ],
            dim=1,
        )
        for block in self.blocks:
            x = block(x, key_padding)
        x = self.output_norm(x)
        memory_states = x[:, : self.config.memory_tokens].unsqueeze(1).expand(-1, steps, -1, -1).contiguous()
        window_repr = self.window_norm(x[:, self.config.memory_tokens : self.config.memory_tokens + steps])
        history_lengths = torch.zeros(batch, steps, dtype=torch.long, device=fused_global.device)
        for row in range(batch):
            current = 0
            for idx in range(steps):
                if not bool(valid[row, idx]):
                    current = 0
                elif bool(reset[row, idx]):
                    current = 1
                else:
                    current = min(current + 1, self.config.max_windows)
                history_lengths[row, idx] = current
        return TemporalMemoryOutput(
            window_repr=window_repr,
            memory_states=memory_states,
            temporal_valid_mask=valid,
            adjacent_eligible_mask=eligible,
            reset_mask=reset,
            gap_reset_mask=gap_reset,
            history_lengths=history_lengths,
        )
