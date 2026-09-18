"""Seeding, device resolution and compute inspection. A GPU is never assumed (ch26 Compute tiers)."""

from __future__ import annotations

import json
import os
import platform
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from conrad.schemas.base import ConradModel


class ComputeReport(ConradModel):
    platform: str
    python: str
    torch_version: str
    cpu_count: int | None
    torch_threads: int
    cuda_available: bool
    cuda_visible_devices: str | None
    cuda_devices: tuple[str, ...]
    mps_available: bool
    compute_tier: str


class SeedReport(ConradModel):
    seed: int
    intended_device: str
    device: str
    device_fallback: bool
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool

    def seeds(self) -> dict[str, int]:
        return {"python": self.seed, "numpy": self.seed, "torch": self.seed}


def inspect_compute() -> ComputeReport:
    cuda = bool(torch.cuda.is_available())
    devices = tuple(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) if cuda else ()
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend is not None and mps_backend.is_available())
    return ComputeReport(
        platform=platform.platform(),
        python=platform.python_version(),
        torch_version=str(torch.__version__),
        cpu_count=os.cpu_count(),
        torch_threads=torch.get_num_threads(),
        cuda_available=cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_devices=devices,
        mps_available=mps,
        compute_tier="REMOTE_OR_LOCAL_GPU" if cuda else "LOCAL_CPU",
    )


def resolve_device(intended: str) -> tuple[str, bool]:
    """Return ``(device, device_fallback)``. A missing accelerator falls back to CPU and says so."""
    wanted = intended.lower()
    if wanted.startswith("cuda"):
        if torch.cuda.is_available():
            index = int(wanted.split(":")[1]) if ":" in wanted else 0
            if index < torch.cuda.device_count():
                return wanted, False
        return "cpu", True
    if wanted == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps", False
        return "cpu", True
    if wanted != "cpu":
        raise ValueError(f"unknown device {intended!r}")
    return "cpu", False


def seed_everything(seed: int, *, deterministic: bool = True, intended_device: str = "cpu") -> SeedReport:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    device, fallback = resolve_device(intended_device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    return SeedReport(
        seed=seed,
        intended_device=intended_device,
        device=device,
        device_fallback=fallback,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
    )


def determinism_flags() -> dict[str, Any]:
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def capture_rng_state() -> dict[str, Any]:
    """Plain containers and tensors only, so the payload stays ``weights_only`` loadable."""
    np_state: Any = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": json.dumps(random.getstate()),
        "numpy": json.dumps(
            [
                np_state[0],
                [int(v) for v in np_state[1]],
                int(np_state[2]),
                int(np_state[3]),
                float(np_state[4]),
            ]
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    version, internal, gauss = json.loads(state["python"])
    random.setstate((version, tuple(internal), gauss))
    kind, keys, pos, has_gauss, cached = json.loads(state["numpy"])
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
