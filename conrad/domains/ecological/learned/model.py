"""Learned CEFD (ch33 freeze, EXPERIMENTAL_CANDIDATE): entity graph-transformer stream + 3D U-Net
field stream coupled by gated cross-attention and entity splatting.

    field_to_entity = gate * cross_attention(query=entity_tokens, key_value=sampled_field_tokens)
    entity_to_field = splat_entity_messages(gate * msg(entity_tokens), positions, grid_shape)
    entity_tokens   = entity_blocks(entity_tokens + field_to_entity)
    field_state     = field_unet(field_state + entity_to_field)

gate = sigmoid(MLP([entity(d), sampled_field(d), uncertainty(u)])): with g ~ 0 there is no coupling.
Positions are normalised grid coordinates in [-1, 1] (x, y, z) for a (B, C, X, Y, Z) field.
Outputs are belief-plane estimates; they never command hardware and are never OBSERVED claims.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from conrad.domains.ecological.learned.config import LearnedCEFDConfig
from conrad.domains.ecological.learned.entity_stream import EntityStream
from conrad.domains.ecological.learned.field_unet import FieldUNet3D

# centre + six axis neighbours, in cells
_OFFSETS = torch.tensor(
    [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float32
)


@dataclass
class CEFDOutput:
    entity_tokens: Tensor  # (B, N, d)
    entity_out: Tensor  # (B, N, entity_heads_out)
    field_out: Tensor  # (B, C_out, X, Y, Z)
    gate: Tensor  # (B, N, 1)


def normalize_positions(
    pos_m: Tensor, origin_m: Tensor, spacing_m: Tensor, shape: tuple[int, int, int]
) -> Tensor:
    """Metres -> [-1, 1] voxel-centre coordinates (align_corners=True convention)."""
    size = torch.tensor(shape, dtype=pos_m.dtype, device=pos_m.device)
    idx = (pos_m - origin_m) / spacing_m - 0.5
    return 2.0 * idx / torch.clamp(size - 1, min=1) - 1.0


class LearnedCEFD(nn.Module):
    def __init__(self, cfg: LearnedCEFDConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, c1 = cfg.d_model, cfg.unet_channels[0]
        self.entities = EntityStream(
            cfg.entity_feature_dim,
            cfg.entity_hidden,
            d,
            cfg.entity_layers,
            cfg.heads,
            cfg.relation_types,
            cfg.relation_dim,
            cfg.ffn_dim,
            cfg.dropout,
        )
        c = cfg.field_in_channels
        self.field_unet = FieldUNet3D(c, c, cfg.unet_channels, cfg.gn_groups)
        self.field_lift = nn.Conv3d(c, c1, kernel_size=1)
        self.field_token = nn.Linear(c1, d)
        self.cross = nn.MultiheadAttention(d, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.dist_scale = nn.Parameter(torch.full((cfg.heads,), cfg.distance_scale_cells))
        self.unc = nn.Linear(cfg.uncertainty_in, cfg.uncertainty_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d + cfg.uncertainty_dim, cfg.gate_hidden), nn.GELU(), nn.Linear(cfg.gate_hidden, 1)
        )
        self.to_field = nn.Linear(d, c)
        self.entity_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.entity_heads_out))
        self.field_head = nn.Conv3d(c, cfg.field_out_channels, kernel_size=1)

    # ------------------------------------------------------------------ coupling pieces
    def sample_field_tokens(self, field: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        """Tokens at each entity and its axis neighbours: (B, N, K, d) plus distances (K,) in cells."""
        b, n, _ = pos.shape
        shape = torch.tensor(field.shape[-3:], dtype=pos.dtype, device=pos.device)
        step = 2.0 / torch.clamp(shape - 1, min=1)
        offs = _OFFSETS.to(pos)
        pts = pos[:, :, None, :] + offs[None, None] * step  # (B, N, K, 3) in xyz
        grid = pts.flip(-1).reshape(b, n, offs.shape[0], 1, 3)  # grid_sample wants (z, y, x) for (X, Y, Z)
        lifted = self.field_lift(field)
        s = F.grid_sample(lifted, grid, mode="bilinear", padding_mode="border", align_corners=True)
        tokens = self.field_token(s[..., 0].permute(0, 2, 3, 1))  # (B, N, K, d)
        return tokens, offs.norm(dim=-1)

    def splat(self, msg: Tensor, pos: Tensor, mask: Tensor, grid_shape: tuple[int, int, int]) -> Tensor:
        """Scatter per-entity messages (B, N, C) into their nearest voxel -> (B, C, X, Y, Z)."""
        b, n, c = msg.shape
        size = torch.tensor(grid_shape, device=pos.device)
        idx = torch.round((pos + 1.0) * 0.5 * (size - 1).to(pos)).long()
        idx = torch.minimum(torch.clamp(idx, min=0), size - 1)
        flat = (idx[..., 0] * grid_shape[1] + idx[..., 1]) * grid_shape[2] + idx[..., 2]  # (B, N)
        out = msg.new_zeros(b, c, grid_shape[0] * grid_shape[1] * grid_shape[2])
        vals = (msg * mask[..., None].to(msg.dtype)).transpose(1, 2)
        out = out.scatter_add(2, flat[:, None, :].expand(b, c, n), vals)
        return out.view(b, c, *grid_shape)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        entity_features: Tensor,
        relations: Tensor,
        entity_mask: Tensor,
        positions: Tensor,
        entity_uncertainty: Tensor,
        field_state: Tensor,
    ) -> CEFDOutput:
        b, n, _ = entity_features.shape
        grid_shape = (int(field_state.shape[2]), int(field_state.shape[3]), int(field_state.shape[4]))
        x = self.entities.tokens(entity_features)
        field = field_state
        u = self.unc(entity_uncertainty)
        m = entity_mask[..., None].to(x.dtype)
        gate = x.new_zeros(b, n, 1)
        for _ in range(self.cfg.coupling_rounds):
            keys, dist = self.sample_field_tokens(field, positions)
            k = keys.shape[2]
            bias = -(dist[None, :] * F.softplus(self.dist_scale)[:, None])  # (H, K) distance/scale bias
            attn_mask = bias[None, :, None, :].expand(b * n, -1, 1, -1).reshape(b * n * self.cfg.heads, 1, k)
            q = x.reshape(b * n, 1, -1)
            kv = keys.reshape(b * n, k, -1)
            att, _ = self.cross(q, kv, kv, attn_mask=attn_mask, need_weights=False)
            gate = torch.sigmoid(self.gate_mlp(torch.cat([x, keys.mean(dim=2), u], dim=-1))) * m
            x = self.entities.blocks(x + gate * att.reshape(b, n, -1), relations, entity_mask)
            splat = self.splat(gate * self.to_field(x), positions, entity_mask, grid_shape)
            field = self.field_unet(field + splat)
        return CEFDOutput(x, self.entity_head(x), self.field_head(field), gate)
