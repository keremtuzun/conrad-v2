"""Modality encoders written in-repo, random init only (ch33 ECMER exact implementation).

No torchvision, no pretrained weights. RGB and sonar get SEPARATE ResNet-18 instances.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from conrad.core.config import CoreConfig
from conrad.core.primitives import ResidualMLP, mlp


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.down: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.down(x))


class ResNet18Encoder(nn.Module):
    """ResNet-18 topology (2-2-2-2 BasicBlocks) through layer4, randomly initialised.

    Output: one global token + ``patch_grid**2`` patch-summary tokens, each projected to De + LayerNorm.
    """

    def __init__(self, cfg: CoreConfig, in_channels: int) -> None:
        super().__init__()
        w = cfg.ecmer.resnet_base_width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w, 7, 2, 3, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        widths = [w, 2 * w, 4 * w, 8 * w]
        layers: list[nn.Module] = []
        c_in = w
        for i, c_out in enumerate(widths):
            stride = 1 if i == 0 else 2
            layers.append(nn.Sequential(BasicBlock(c_in, c_out, stride), BasicBlock(c_out, c_out, 1)))
            c_in = c_out
        self.layers = nn.Sequential(*layers)
        self.feature_dim = c_in  # 512 at base width 64
        self.grid = cfg.ecmer.patch_grid
        self.global_proj = nn.Sequential(nn.Linear(c_in, cfg.evidence_dim), nn.LayerNorm(cfg.evidence_dim))
        self.patch_proj = nn.Sequential(nn.Linear(c_in, cfg.evidence_dim), nn.LayerNorm(cfg.evidence_dim))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def feature_map(self, image: Tensor) -> Tensor:
        out: Tensor = self.layers(self.stem(image))
        return out

    def forward(self, image: Tensor) -> Tensor:
        """``image`` [B, C, H, W] -> tokens [B, 1 + grid^2, De]."""
        fmap = self.feature_map(image)
        glob = self.global_proj(fmap.mean(dim=(2, 3))).unsqueeze(1)
        patches = F.adaptive_avg_pool2d(fmap, self.grid).flatten(2).transpose(1, 2)
        return torch.cat([glob, self.patch_proj(patches)], dim=1)


class VoxelPointEncoder(nn.Module):
    """Voxelise to G^3; three Conv3d stages (k3, s2, GroupNorm, GELU); pool -> 128 -> De. One token."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        e = cfg.ecmer
        self.grid, self.extent = e.voxel_grid, e.voxel_extent_m
        stages: list[nn.Module] = []
        c_in = 1
        for c_out in e.voxel_channels:
            stages += [
                nn.Conv3d(c_in, c_out, 3, 2, 1),
                nn.GroupNorm(min(e.voxel_groups, c_out), c_out),
                nn.GELU(),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*stages)
        self.head = mlp([c_in, 128, cfg.evidence_dim])

    def voxelize(self, points: Tensor, mask: Tensor) -> Tensor:
        """``points`` [B, P, 3] in the sensor frame (metres), ``mask`` [B, P] -> occupancy [B, 1, G, G, G]."""
        b = points.shape[0]
        g = self.grid
        idx = ((points / self.extent + 0.5) * g).floor().long()
        inside = mask.to(torch.bool) & ((idx >= 0) & (idx < g)).all(-1)
        flat = (idx[..., 0] * g + idx[..., 1]) * g + idx[..., 2]
        flat = torch.where(inside, flat, torch.zeros_like(flat))
        grid = torch.zeros(b, g * g * g, dtype=points.dtype, device=points.device)
        grid.scatter_add_(1, flat, inside.to(points.dtype))
        return grid.clamp(max=1.0).view(b, 1, g, g, g)

    def forward(self, points: Tensor, mask: Tensor) -> Tensor:
        fmap = self.conv(self.voxelize(points, mask))
        out: Tensor = self.head(fmap.mean(dim=(2, 3, 4))).unsqueeze(1)
        return out


class ScalarSensorEncoder(nn.Module):
    """M -> 128 -> 256 -> 256 with one missingness bit per scalar. Standardisation is external (train split)."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        m, h = cfg.ecmer.scalar_max_values, cfg.ecmer.scalar_hidden
        self.m = m
        self.net = nn.Sequential(
            nn.Linear(2 * m, h),
            nn.GELU(),
            nn.Linear(h, cfg.evidence_dim),
            ResidualMLP(cfg.evidence_dim, cfg.evidence_dim, cfg.evidence_dim, cfg.dropout),
        )

    def forward(self, values: Tensor, present: Tensor) -> Tensor:
        """``values`` [B, M'], ``present`` [B, M'] with M' <= M. Missing values are zeroed, never imputed."""
        b, m_in = values.shape
        if m_in > self.m:
            raise ValueError(f"{m_in} scalars exceed scalar_max_values={self.m}")
        v = torch.zeros(b, self.m, dtype=values.dtype, device=values.device)
        p = torch.zeros_like(v)
        pres = present.to(values.dtype)
        v[:, :m_in] = torch.nan_to_num(values) * pres
        p[:, :m_in] = pres
        out: Tensor = self.net(torch.cat([v, p], dim=-1)).unsqueeze(1)
        return out


class PoseTimeContextEncoder(nn.Module):
    """position(3) quat(4) cov summary(6) delta_t(1) -> 16 -> 128 -> De, plus sensor/frame embeddings."""

    RAW_DIM = 14

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        h1, h2 = cfg.ecmer.context_hidden
        self.net = mlp([self.RAW_DIM, h1, h2, cfg.evidence_dim])
        self.sensor = nn.Embedding(cfg.ecmer.max_sensors, cfg.evidence_dim)
        self.frame = nn.Embedding(cfg.ecmer.max_frames, cfg.evidence_dim)

    def forward(self, raw: Tensor, sensor_idx: Tensor, frame_idx: Tensor) -> Tensor:
        """Returns [B, De]; added to every modality token of the event."""
        out: Tensor = self.net(raw) + self.sensor(sensor_idx) + self.frame(frame_idx)
        return out
