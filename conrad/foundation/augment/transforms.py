"""Deterministic transform registry for P3 views."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class PhysicalityClass(str, Enum):
    PHYSICS_GROUNDED = "PHYSICS_GROUNDED"
    CALIBRATED_EMPIRICAL = "CALIBRATED_EMPIRICAL"
    STRESS_TEST = "STRESS_TEST"


@dataclass(frozen=True)
class TransformTrace:
    transform_id: str
    physicality: PhysicalityClass
    seed: int
    params: dict[str, Any]


@dataclass(frozen=True)
class ViewRecipe:
    recipe_id: str
    transforms: tuple[str, ...] = ()
    modality_dropout: tuple[str, ...] = ()
    token_mask_fraction: float = 0.0
    teacher: bool = False


TransformFn = Callable[[torch.Tensor, torch.Generator, dict[str, Any]], torch.Tensor]


@dataclass
class TransformRegistry:
    _items: dict[str, tuple[PhysicalityClass, TransformFn, dict[str, Any]]] = field(default_factory=dict)

    def register(
        self, transform_id: str, physicality: PhysicalityClass, fn: TransformFn, params: dict[str, Any] | None = None
    ) -> None:
        if transform_id in self._items:
            raise ValueError(f"duplicate transform {transform_id}")
        self._items[transform_id] = (physicality, fn, dict(params or {}))

    def apply(self, transform_id: str, x: torch.Tensor, *, seed: int) -> tuple[torch.Tensor, TransformTrace]:
        if transform_id not in self._items:
            raise KeyError(transform_id)
        physicality, fn, params = self._items[transform_id]
        g = torch.Generator(device=x.device).manual_seed(seed)
        return fn(x, g, dict(params)), TransformTrace(transform_id, physicality, seed, dict(params))


def _identity(x: torch.Tensor, _g: torch.Generator, _params: dict[str, Any]) -> torch.Tensor:
    return x.clone()


def _jitter(x: torch.Tensor, g: torch.Generator, params: dict[str, Any]) -> torch.Tensor:
    sigma = float(params.get("sigma", 0.01))
    return x + torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * sigma


def _scale(x: torch.Tensor, _g: torch.Generator, params: dict[str, Any]) -> torch.Tensor:
    return x * float(params.get("scale", 0.98))


def default_registry() -> TransformRegistry:
    registry = TransformRegistry()
    registry.register("identity_recorded_observation", PhysicalityClass.PHYSICS_GROUNDED, _identity)
    registry.register("rgb_calibrated_jitter", PhysicalityClass.CALIBRATED_EMPIRICAL, _jitter, {"sigma": 0.01})
    registry.register("sonar_snr_jitter", PhysicalityClass.CALIBRATED_EMPIRICAL, _jitter, {"sigma": 0.02})
    registry.register("range_quantization_stress", PhysicalityClass.STRESS_TEST, _scale, {"scale": 0.99})
    registry.register("geometry_point_jitter", PhysicalityClass.CALIBRATED_EMPIRICAL, _jitter, {"sigma": 0.005})
    registry.register("context_calibrated_scale", PhysicalityClass.CALIBRATED_EMPIRICAL, _scale, {"scale": 0.98})
    return registry


def replay_seed(base_seed: int, recipe_id: str, lineage_id: str, transform_id: str) -> int:
    payload = f"{base_seed}:{recipe_id}:{lineage_id}:{transform_id}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) % (2**31)

