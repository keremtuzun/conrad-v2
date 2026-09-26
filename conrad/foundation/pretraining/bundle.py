"""Tiny and architecture-faithful OS-FM training bundles."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from conrad.foundation.data.batch import FoundationBatch
from conrad.foundation.pretraining.ema import EMASchedule, update_ema_teacher
from conrad.foundation.pretraining.losses import ObjectiveResult, ObjectiveRouter


class TinyFoundationEncoder(nn.Module):
    def __init__(self, input_dim: int = 8, d_model: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.net(x)
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out


class ArchitectureFaithfulFoundationEncoder(nn.Module):
    """Minimal P2/P3 skeleton for smoke wiring: D_F=384, scene latents, 3 cross and 6 self blocks."""

    def __init__(self, input_dim: int = 8, d_model: int = 384, heads: int = 6) -> None:
        super().__init__()
        self.d_model = d_model
        self.scene_latents = nn.Parameter(torch.zeros(64, d_model))
        self.input = nn.Linear(input_dim, d_model)
        self.cross = nn.ModuleList(
            [nn.MultiheadAttention(d_model, heads, batch_first=True) for _ in range(3)]
        )
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=768, batch_first=True)
        self.self_blocks = nn.TransformerEncoder(layer, num_layers=6)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.input(x)
        latents = self.scene_latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for block in self.cross:
            latents, _ = block(latents, tokens, tokens, key_padding_mask=padding_mask, need_weights=False)
            latents = self.norm(latents)
        return self.self_blocks(latents)


@dataclass
class StepOutput:
    loss: torch.Tensor
    results: tuple[ObjectiveResult, ...]
    student_repr: torch.Tensor
    teacher_repr: torch.Tensor


class OSFMTrainingBundle(nn.Module):
    def __init__(self, student: nn.Module, *, ema_schedule: EMASchedule | None = None) -> None:
        super().__init__()
        self.student = student
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.reconstruction_head = nn.Linear(getattr(student, "d_model", 32), getattr(student, "d_model", 32))
        self.router = ObjectiveRouter()
        self.ema_schedule = ema_schedule or EMASchedule()

    def forward(self, batch: FoundationBatch) -> StepOutput:
        student_repr = self.student(batch.tokens, batch.padding_mask)
        with torch.no_grad():
            teacher_repr = self.teacher(batch.tokens, batch.padding_mask)
        if student_repr.shape[1] != batch.token_mask.shape[1]:
            valid = torch.ones(student_repr.shape[:2], dtype=torch.bool, device=student_repr.device)
            token_mask = torch.zeros_like(valid)
        else:
            valid = ~batch.padding_mask
            token_mask = batch.token_mask
        ts = self.router.teacher_student(student_repr, teacher_repr, valid)
        recon = self.router.mask_reconstruction(student_repr, teacher_repr, token_mask)
        loss = self.router.total((ts, recon))
        return StepOutput(loss, (ts, recon), student_repr, teacher_repr)

    @torch.no_grad()
    def update_teacher_after_optimizer(self, step: int) -> float:
        momentum = self.ema_schedule.value(step)
        update_ema_teacher(self.student, self.teacher, momentum)
        return momentum

