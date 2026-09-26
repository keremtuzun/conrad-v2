"""Objective routing with explicit denominators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F


class ObjectiveStatus(str, Enum):
    ACTIVE = "ACTIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ObjectiveResult:
    objective_id: str
    numerator: torch.Tensor
    denominator: torch.Tensor
    normalized_loss: torch.Tensor
    weight: float
    contribution: torch.Tensor
    status: ObjectiveStatus

    def as_metrics(self) -> dict[str, float | str]:
        return {
            f"{self.objective_id}/numerator": float(self.numerator.detach().cpu()),
            f"{self.objective_id}/denominator": float(self.denominator.detach().cpu()),
            f"{self.objective_id}/loss": float(self.normalized_loss.detach().cpu()),
            f"{self.objective_id}/contribution": float(self.contribution.detach().cpu()),
            f"{self.objective_id}/status": self.status.value,
        }


class ObjectiveRouter:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {"teacher_student_mse": 1.0, "mask_reconstruction": 0.25}

    def teacher_student(self, student: torch.Tensor, teacher: torch.Tensor, valid_mask: torch.Tensor) -> ObjectiveResult:
        denom = valid_mask.float().sum()
        numerator = F.mse_loss(student, teacher.detach(), reduction="none").mean(dim=-1)
        numerator = (numerator * valid_mask.float()).sum()
        return self._result("teacher_student_mse", numerator, denom)

    def mask_reconstruction(self, prediction: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor) -> ObjectiveResult:
        denom = token_mask.float().sum()
        numerator = F.mse_loss(prediction, target.detach(), reduction="none").mean(dim=-1)
        numerator = (numerator * token_mask.float()).sum()
        return self._result("mask_reconstruction", numerator, denom)

    def _result(self, objective_id: str, numerator: torch.Tensor, denominator: torch.Tensor) -> ObjectiveResult:
        weight = float(self.weights.get(objective_id, 1.0))
        if float(denominator.detach().cpu()) == 0.0:
            zero = numerator * 0.0
            return ObjectiveResult(objective_id, numerator, denominator, zero, weight, zero, ObjectiveStatus.NOT_APPLICABLE)
        if not torch.isfinite(numerator):
            zero = numerator * 0.0
            return ObjectiveResult(objective_id, numerator, denominator, zero, weight, zero, ObjectiveStatus.INVALID)
        loss = numerator / denominator
        return ObjectiveResult(objective_id, numerator, denominator, loss, weight, loss * weight, ObjectiveStatus.ACTIVE)

    def total(self, results: tuple[ObjectiveResult, ...]) -> torch.Tensor:
        active = [r.contribution for r in results if r.status is ObjectiveStatus.ACTIVE]
        if not active:
            raise RuntimeError("no active objectives; refusing to fake a zero-loss success")
        return torch.stack(active).sum()

