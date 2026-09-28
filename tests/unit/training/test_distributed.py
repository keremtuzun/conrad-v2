from __future__ import annotations

import pytest

from conrad.training.distributed import DistributedContext, distributed_batch_plan


def test_single_process_context_is_primary_and_disabled() -> None:
    context = DistributedContext()
    assert context.primary is True
    assert context.enabled is False


def test_two_gpu_plan_preserves_global_effective_batch() -> None:
    assert distributed_batch_plan(
        effective_batch_target=256,
        gradient_accumulation=4,
        world_size=2,
    ) == (32, 4, 256)


def test_distributed_batch_plan_fails_when_target_cannot_be_preserved() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        distributed_batch_plan(
            effective_batch_target=250,
            gradient_accumulation=4,
            world_size=2,
        )
