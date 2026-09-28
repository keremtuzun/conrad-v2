"""Small fail-closed distributed-training contract for governed research runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as torch_dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def primary(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
            raise RuntimeError(
                f"invalid distributed environment rank={rank} local_rank={local_rank} world_size={world_size}"
            )
        context = cls(rank=rank, local_rank=local_rank, world_size=world_size)
        if context.enabled:
            if not torch.cuda.is_available():
                raise RuntimeError("distributed formal training requires CUDA")
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} exceeds visible CUDA devices={torch.cuda.device_count()}"
                )
            torch.cuda.set_device(local_rank)
            if not torch_dist.is_initialized():
                torch_dist.init_process_group(backend="nccl", init_method="env://")
        return context

    def barrier(self) -> None:
        if self.enabled:
            torch_dist.barrier()

    def broadcast_object(self, value: Any, *, source: int = 0) -> Any:
        if not self.enabled:
            return value
        values = [value]
        torch_dist.broadcast_object_list(values, src=source)
        return values[0]

    def gather_objects(self, value: Any) -> list[Any]:
        if not self.enabled:
            return [value]
        values: list[Any] = [None for _ in range(self.world_size)]
        torch_dist.all_gather_object(values, value)
        return values

    def close(self) -> None:
        if self.enabled and torch_dist.is_initialized():
            torch_dist.destroy_process_group()


def distributed_batch_plan(
    *, effective_batch_target: int, gradient_accumulation: int, world_size: int
) -> tuple[int, int, int]:
    """Return per-device batch, accumulation, and unchanged global effective batch."""
    if effective_batch_target <= 0 or gradient_accumulation <= 0 or world_size <= 0:
        raise ValueError("distributed batch values must be positive")
    divisor = gradient_accumulation * world_size
    if effective_batch_target % divisor:
        raise ValueError(
            f"effective batch {effective_batch_target} is not divisible by accumulation*world_size {divisor}"
        )
    return effective_batch_target // divisor, gradient_accumulation, effective_batch_target
