"""Trainer configuration and result records (ch33 Training configuration)."""

from __future__ import annotations

import fnmatch
import math
from typing import Any

import torch
from pydantic import Field, model_validator

from conrad.schemas.base import ConradModel
from conrad.training.checkpoint_meta import is_validation_metric


class FinalOnlyViolation(RuntimeError):
    """Test / OOD metrics were requested before training was finalized, or training resumed afterwards."""


class TrainerConfig(ConradModel):
    component: str = Field(min_length=1)
    lr_new: float = Field(default=3e-4, gt=0)
    lr_pretrained: float = Field(default=1e-4, gt=0)
    pretrained_patterns: tuple[str, ...] = ()
    weight_decay: float = Field(default=0.01, ge=0)
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_fraction: float = Field(default=0.05, ge=0, lt=1)
    min_lr_ratio: float = Field(default=0.0, ge=0, le=1)
    grad_clip_norm: float = Field(default=1.0, gt=0)
    batch_size: int = Field(default=16, ge=1)
    effective_batch_size: int = Field(default=64, ge=1)
    max_epochs: int = Field(default=200, ge=1)
    early_stopping_patience: int = Field(default=25, ge=1)
    primary_metric: str = "val_loss"
    metric_direction: str = Field(default="lower", pattern="^(lower|higher)$")
    seed: int = 2026201
    device: str = "cpu"
    deterministic: bool = True
    is_encoder: bool = False
    representation_pretraining_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> TrainerConfig:
        if self.effective_batch_size % self.batch_size != 0:
            raise ValueError("effective_batch_size must be a multiple of batch_size")
        if not is_validation_metric(self.primary_metric):
            raise ValueError(
                f"primary_metric {self.primary_metric!r} is not a validation metric; "
                "checkpoints are chosen on validation only, test and OOD are final-only"
            )
        if self.is_encoder and not self.representation_pretraining_id:
            raise ValueError("encoder training must declare representation_pretraining_id")
        return self

    @property
    def accumulation_steps(self) -> int:
        return self.effective_batch_size // self.batch_size


class RunIdentity(ConradModel):
    """Data / code identity stamped into every checkpoint."""

    manifest_digests: tuple[str, ...]
    split_hash: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    extra_compatibility: dict[str, str] = Field(default_factory=dict)


class EpochRecord(ConradModel):
    epoch: int
    step: int
    train_loss: float
    lr: float
    metrics: dict[str, float]
    improved: bool


class FitResult(ConradModel):
    epochs_run: int
    optimizer_steps: int
    best_epoch: int | None
    best_metric: float | None
    stopped_early: bool
    best_checkpoint: str | None
    last_checkpoint: str | None
    history: tuple[EpochRecord, ...]


def warmup_cosine(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    span = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0) / span, 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_param_groups(model: torch.nn.Module, config: TrainerConfig) -> list[dict[str, Any]]:
    new: list[torch.nn.Parameter] = []
    pretrained: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_pre = any(fnmatch.fnmatchcase(name, pattern) for pattern in config.pretrained_patterns)
        (pretrained if is_pre else new).append(param)
    groups: list[dict[str, Any]] = []
    if new:
        groups.append({"params": new, "lr": config.lr_new, "group_name": "new"})
    if pretrained:
        groups.append({"params": pretrained, "lr": config.lr_pretrained, "group_name": "pretrained"})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups
