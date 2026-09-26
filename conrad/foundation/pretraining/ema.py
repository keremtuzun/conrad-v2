"""EMA teacher schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EMASchedule:
    start: float = 0.996
    end: float = 0.9999
    total_steps: int = 100

    def value(self, step: int) -> float:
        t = min(max(step, 0), self.total_steps) / max(1, self.total_steps)
        cosine = 0.5 - 0.5 * math.cos(math.pi * t)
        return self.start + (self.end - self.start) * cosine


@torch.no_grad()
def update_ema_teacher(student: torch.nn.Module, teacher: torch.nn.Module, momentum: float) -> None:
    for t, s in zip(teacher.parameters(), student.parameters(), strict=True):
        t.data.mul_(momentum).add_(s.data, alpha=1.0 - momentum)

