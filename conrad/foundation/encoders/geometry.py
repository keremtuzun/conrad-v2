"""Deterministic grouped point-cloud geometry encoder for OS-FM U1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GeometryEncoderConfig:
    input_dim: int = 3
    width: int = 256
    output_dim: int = 384
    depth: int = 6
    heads: int = 4
    mlp_ratio: int = 4
    num_groups: int = 128
    group_size: int = 32
    dense_tap_layers: tuple[int, ...] = (2, 4, 6)
    grouping_method: str = "deterministic_fps_valid_xyz_then_nearest_neighbors"
    pretrained_source: str = "none"
    initialization: str = "deterministic_random_smoke"


@dataclass(frozen=True)
class GeometryEncoderOutput:
    modality_repr: torch.Tensor
    group_tokens: torch.Tensor
    internal_group_tokens: torch.Tensor
    dense_taps: dict[int, torch.Tensor]
    dense_taps_projected: dict[int, torch.Tensor]
    grouped_points: torch.Tensor
    group_centers: torch.Tensor
    group_valid_mask: torch.Tensor
    group_valid_fraction: torch.Tensor
    neighbor_indices: torch.Tensor
    visible_mask: torch.Tensor
    masked_group_mask: torch.Tensor


class GeometryGroupedEncoder(nn.Module):
    """Point-MAE-style local-group encoder with frozen P2 width/depth/head/projection contract."""

    d_model = 384
    internal_width = 256

    def __init__(self, config: GeometryEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or GeometryEncoderConfig()
        if self.config.input_dim != 3:
            raise ValueError("P2 geometry input is XYZ-primary; input_dim must be 3")
        if self.config.width != self.internal_width:
            raise ValueError("P2 freezes geometry internal width at 256")
        if self.config.output_dim != self.d_model:
            raise ValueError("P2 freezes OS-FM D_F at 384")
        if self.config.depth != 6:
            raise ValueError("P2 freezes geometry transformer depth at 6")
        if self.config.heads != 4:
            raise ValueError("P2 freezes geometry attention heads at 4")
        self.point_embed = nn.Sequential(
            nn.Linear(self.config.input_dim + 1, self.config.width),
            nn.GELU(),
            nn.Linear(self.config.width, self.config.width),
        )
        self.group_context = nn.Sequential(
            nn.Linear(5, self.config.width),
            nn.GELU(),
            nn.Linear(self.config.width, self.config.width),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.width))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.config.width))
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
        self.output_projection = nn.Linear(self.config.width, self.config.output_dim)
        self.output_norm = nn.LayerNorm(self.config.output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _fps_indices_single(self, points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        n_points = points.shape[0]
        valid_idx = valid.nonzero(as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            return torch.zeros(self.config.num_groups, dtype=torch.long, device=points.device)
        valid_points = points.index_select(0, valid_idx)
        first_local = torch.lexsort if False else None
        del first_local
        scores = (
            valid_points[:, 0] * 1_000_000.0
            + valid_points[:, 1] * 1_000.0
            + valid_points[:, 2]
        )
        selected_local = [int(torch.argmin(scores).detach().cpu())]
        min_dist = torch.cdist(valid_points[selected_local[-1] : selected_local[-1] + 1], valid_points).squeeze(0)
        for _ in range(1, self.config.num_groups):
            farthest = int(torch.argmax(min_dist).detach().cpu())
            selected_local.append(farthest)
            candidate_dist = torch.cdist(valid_points[farthest : farthest + 1], valid_points).squeeze(0)
            min_dist = torch.minimum(min_dist, candidate_dist)
        selected = valid_idx[torch.tensor(selected_local, dtype=torch.long, device=points.device)]
        if selected.numel() < self.config.num_groups:
            pad = selected[-1:].expand(self.config.num_groups - selected.numel())
            selected = torch.cat([selected, pad], dim=0)
        return selected[: self.config.num_groups]

    def group_points(self, points: torch.Tensor, validity_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if points.ndim != 3 or points.shape[-1] != self.config.input_dim:
            raise ValueError("geometry points must be [B, P, 3]")
        if validity_mask.shape != points.shape[:2]:
            raise ValueError("validity_mask must be [B, P]")
        clean = points.masked_fill(~validity_mask.unsqueeze(-1).bool(), 0.0)
        grouped: list[torch.Tensor] = []
        centers: list[torch.Tensor] = []
        group_valid: list[torch.Tensor] = []
        neighbor_indices: list[torch.Tensor] = []
        for batch_idx in range(points.shape[0]):
            center_idx = self._fps_indices_single(clean[batch_idx], validity_mask[batch_idx].bool())
            center = clean[batch_idx].index_select(0, center_idx)
            dist = torch.cdist(center, clean[batch_idx])
            dist = dist.masked_fill(~validity_mask[batch_idx].bool().unsqueeze(0), torch.finfo(dist.dtype).max)
            knn = torch.topk(dist, k=self.config.group_size, largest=False, dim=1).indices
            local = clean[batch_idx].index_select(0, knn.flatten()).view(
                self.config.num_groups, self.config.group_size, self.config.input_dim
            )
            local_valid = validity_mask[batch_idx].bool().index_select(0, knn.flatten()).view(
                self.config.num_groups, self.config.group_size
            )
            grouped.append(local)
            centers.append(center)
            group_valid.append(local_valid)
            neighbor_indices.append(knn)
        return (
            torch.stack(grouped, dim=0),
            torch.stack(centers, dim=0),
            torch.stack(group_valid, dim=0),
            torch.stack(neighbor_indices, dim=0),
        )

    def forward(
        self,
        points: torch.Tensor,
        validity_mask: torch.Tensor,
        masked_group_mask: torch.Tensor | None = None,
    ) -> GeometryEncoderOutput:
        grouped, centers, group_valid, neighbor_indices = self.group_points(points, validity_mask)
        if masked_group_mask is None:
            masked_group_mask = torch.zeros(
                points.shape[0], self.config.num_groups, dtype=torch.bool, device=points.device
            )
        if masked_group_mask.shape != (points.shape[0], self.config.num_groups):
            raise ValueError("masked_group_mask must match [B, num_groups]")
        group_valid_fraction = group_valid.float().mean(dim=-1)
        relative = grouped - centers.unsqueeze(2)
        point_input = torch.cat([relative, group_valid.unsqueeze(-1).float()], dim=-1)
        point_features = self.point_embed(point_input)
        group_denom = group_valid.float().sum(dim=-1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
        group_tokens = (point_features * group_valid.unsqueeze(-1).float()).sum(dim=2) / group_denom.squeeze(2)
        radial = torch.linalg.norm(centers, dim=-1, keepdim=True)
        scale = torch.linalg.norm(relative, dim=-1).amax(dim=-1, keepdim=True)
        context = torch.cat([centers, radial, group_valid_fraction.unsqueeze(-1)], dim=-1)
        group_tokens = group_tokens + self.group_context(context)
        group_tokens = torch.where(masked_group_mask.unsqueeze(-1), self.mask_token.expand_as(group_tokens), group_tokens)
        tokens = torch.cat([self.cls_token.expand(points.shape[0], -1, -1), group_tokens], dim=1)
        taps: dict[int, torch.Tensor] = {}
        taps_projected: dict[int, torch.Tensor] = {}
        for layer_idx, block in enumerate(self.blocks, start=1):
            tokens = block(tokens)
            if layer_idx in self.config.dense_tap_layers:
                internal = self.norm(tokens[:, 1:])
                taps[layer_idx] = internal
                taps_projected[layer_idx] = self.output_norm(self.output_projection(internal))
        tokens = self.norm(tokens)
        projected = self.output_norm(self.output_projection(tokens))
        return GeometryEncoderOutput(
            modality_repr=projected[:, 0],
            group_tokens=projected[:, 1:],
            internal_group_tokens=tokens[:, 1:],
            dense_taps=taps,
            dense_taps_projected=taps_projected,
            grouped_points=grouped,
            group_centers=centers,
            group_valid_mask=group_valid,
            group_valid_fraction=group_valid_fraction,
            neighbor_indices=neighbor_indices,
            visible_mask=~masked_group_mask,
            masked_group_mask=masked_group_mask,
        )
