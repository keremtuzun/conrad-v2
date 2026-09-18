"""Field stream: 3-level 3D U-Net (ch33): C -> 64 -> 128 -> 256, two Conv3d+GroupNorm+GELU per level,
decoder 256 -> 128 -> 64 -> C_out with skip connections.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def conv_block(cin: int, cout: int, groups: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel_size=3, padding=1),
        nn.GroupNorm(groups, cout),
        nn.GELU(),
        nn.Conv3d(cout, cout, kernel_size=3, padding=1),
        nn.GroupNorm(groups, cout),
        nn.GELU(),
    )


class FieldUNet3D(nn.Module):
    def __init__(self, c_in: int, c_out: int, channels: tuple[int, int, int], groups: int) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.enc1 = conv_block(c_in, c1, groups)
        self.enc2 = conv_block(c1, c2, groups)
        self.enc3 = conv_block(c2, c3, groups)
        self.dec2 = conv_block(c3 + c2, c2, groups)
        self.dec1 = conv_block(c2 + c1, c1, groups)
        self.head = nn.Conv3d(c1, c_out, kernel_size=1)

    @staticmethod
    def _down(x: Tensor) -> Tensor:
        if min(x.shape[-3:]) < 2:
            return x
        return F.max_pool3d(x, kernel_size=2, ceil_mode=True)

    @staticmethod
    def _up(x: Tensor, like: Tensor) -> Tensor:
        return F.interpolate(x, size=like.shape[-3:], mode="trilinear", align_corners=False)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self._down(e1))
        e3 = self.enc3(self._down(e2))
        d2 = self.dec2(torch.cat([self._up(e3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        out: Tensor = self.head(d1)
        return out
