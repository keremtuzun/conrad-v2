"""Progressive curriculum runner C0..C9 (ch3 Core training sequence, ch23 Stages I-VI).

Stages freeze / unfreeze parameters by name pattern. Patterns are configuration, not code: the
defaults live in ``configs/train/base_curriculum.yaml``. A pattern that matches nothing is an
error, so a renamed module cannot silently drop out of training.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import torch
import yaml
from pydantic import Field, model_validator

from conrad.schemas.base import ConradModel
from conrad.training.trainer import FitResult, Trainer

CORE_STAGE_NAMES: dict[str, str] = {
    "C0": "encoder initialization",
    "C1": "association",
    "C2": "BUO",
    "C3": "contradiction/uncertainty",
    "C4": "RBP",
    "C5": "TBD",
    "C6": "sequential BUO/TBD",
    "C7": "BUO + RBP + TBD",
    "C8": "persistent long sequences",
    "C9": "real-domain adaptation",
}


class CurriculumError(ValueError):
    pass


class StageStatus(str, Enum):
    COMPLETED = "COMPLETED"
    BLOCKED_EXTERNAL = "BLOCKED_EXTERNAL"
    SKIPPED_BY_CONFIG = "SKIPPED_BY_CONFIG"


class CurriculumStage(ConradModel):
    stage_id: str = Field(pattern="^C[0-9]$")
    name: str = Field(min_length=1)
    trainable_patterns: tuple[str, ...] = Field(min_length=1)
    max_epochs: int = Field(ge=1)
    requires_real_data: bool = False
    enabled: bool = True


class Curriculum(ConradModel):
    curriculum_id: str = Field(min_length=1)
    stages: tuple[CurriculumStage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> Curriculum:
        ids = [s.stage_id for s in self.stages]
        if ids != sorted(set(ids)):
            raise ValueError(f"curriculum stages must be unique and in C0..C9 order, got {ids}")
        return self


class FreezeReport(ConradModel):
    stage_id: str
    trainable: tuple[str, ...]
    frozen: tuple[str, ...]
    trainable_parameter_count: int


class StageResult(ConradModel):
    stage_id: str
    name: str
    status: StageStatus
    reason: str | None = None
    freeze: FreezeReport | None = None
    fit: FitResult | None = None


def load_curriculum(path: str | Path) -> Curriculum:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    section = data.get("train", data) if isinstance(data, dict) else None
    if not isinstance(section, dict) or "curriculum" not in section:
        raise CurriculumError(f"{path} must contain a 'train.curriculum' (or top-level 'curriculum') mapping")
    return Curriculum.model_validate(section["curriculum"])


def apply_stage_freezing(model: torch.nn.Module, stage: CurriculumStage) -> FreezeReport:
    names = [n for n, _ in model.named_parameters()]
    for pattern in stage.trainable_patterns:
        if not any(fnmatch.fnmatchcase(n, pattern) for n in names):
            raise CurriculumError(f"stage {stage.stage_id}: pattern {pattern!r} matches no parameter")
    trainable: list[str] = []
    frozen: list[str] = []
    count = 0
    for name, param in model.named_parameters():
        on = any(fnmatch.fnmatchcase(name, pattern) for pattern in stage.trainable_patterns)
        param.requires_grad_(on)
        if on:
            trainable.append(name)
            count += param.numel()
        else:
            frozen.append(name)
    return FreezeReport(
        stage_id=stage.stage_id,
        trainable=tuple(trainable),
        frozen=tuple(frozen),
        trainable_parameter_count=count,
    )


TrainerFactory = Callable[[CurriculumStage, torch.nn.Module], Trainer]
"""Builds the stage trainer AFTER freezing, so the optimizer holds only trainable parameters."""


class CurriculumRunner:
    def __init__(
        self,
        curriculum: Curriculum,
        trainer_factory: TrainerFactory,
        *,
        real_data_available: bool = False,
    ) -> None:
        self.curriculum = curriculum
        self.trainer_factory = trainer_factory
        self.real_data_available = real_data_available

    def run(self, model: torch.nn.Module) -> tuple[StageResult, ...]:
        results: list[StageResult] = []
        for stage in self.curriculum.stages:
            if not stage.enabled:
                results.append(
                    StageResult(
                        stage_id=stage.stage_id, name=stage.name, status=StageStatus.SKIPPED_BY_CONFIG
                    )
                )
                continue
            if stage.requires_real_data and not self.real_data_available:
                results.append(
                    StageResult(
                        stage_id=stage.stage_id,
                        name=stage.name,
                        status=StageStatus.BLOCKED_EXTERNAL,
                        reason="no verified real-data manifest is available; the stage was not run",
                    )
                )
                continue
            freeze = apply_stage_freezing(model, stage)
            trainer = self.trainer_factory(stage, model)
            fit = trainer.fit(max_epochs=stage.max_epochs)
            results.append(
                StageResult(
                    stage_id=stage.stage_id,
                    name=stage.name,
                    status=StageStatus.COMPLETED,
                    freeze=freeze,
                    fit=fit,
                )
            )
        return tuple(results)
