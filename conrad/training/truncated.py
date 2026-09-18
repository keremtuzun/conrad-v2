"""Truncated persistent training (C8 persistent long sequences).

A long sequence is processed in windows of N steps. Gradients flow inside a window only; the
belief state carried into the next window is detached, so memory stays bounded while the state
itself persists across the whole sequence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from conrad.schemas.base import ConradModel

State = Any
StepFn = Callable[[State, Any], tuple[torch.Tensor, State]]
"""``(carried_state, item) -> (loss, new_state)``."""


class TruncatedResult(ConradModel):
    window_losses: tuple[float, ...]
    optimizer_steps: int
    steps_processed: int


def detach_state(state: State) -> State:
    """Recursively detach every tensor; containers keep their type. Non-tensor leaves pass through."""
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    if isinstance(state, tuple) and hasattr(state, "_fields"):
        return type(state)(*(detach_state(v) for v in state))
    if isinstance(state, tuple):
        return tuple(detach_state(v) for v in state)
    if isinstance(state, list):
        return [detach_state(v) for v in state]
    return state


def state_requires_grad(state: State) -> bool:
    if isinstance(state, torch.Tensor):
        return state.grad_fn is not None
    if isinstance(state, dict):
        return any(state_requires_grad(v) for v in state.values())
    if isinstance(state, tuple | list):
        return any(state_requires_grad(v) for v in state)
    return False


def train_truncated_sequence(
    step_fn: StepFn,
    sequence: Sequence[Any],
    initial_state: State,
    *,
    window: int,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float = 1.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[TruncatedResult, State]:
    if window < 1:
        raise ValueError("window must be >= 1")
    state = detach_state(initial_state)
    losses: list[float] = []
    steps = 0
    for start in range(0, len(sequence), window):
        chunk = sequence[start : start + window]
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros(())
        for item in chunk:
            loss, state = step_fn(state, item)
            total = total + loss
        mean = total / len(chunk)
        if not torch.isfinite(mean):
            raise FloatingPointError(f"non-finite loss in window starting at step {start}")
        mean.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]], grad_clip_norm
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        state = detach_state(state)
        losses.append(float(mean.detach()))
        steps += 1
    return (
        TruncatedResult(window_losses=tuple(losses), optimizer_steps=steps, steps_processed=len(sequence)),
        state,
    )
