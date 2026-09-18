"""Causal-trace records of the navigation stack: goal -> trajectory -> wrench -> allocated command.

All records of one goal share its ``trace_id``. The integrator maps them onto RuntimeEvents.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


class NavRecordType:
    GOAL_ACCEPTED = "GOAL_ACCEPTED"
    GOAL_REJECTED = "GOAL_REJECTED"
    TRAJECTORY_PLANNED = "TRAJECTORY_PLANNED"
    WRENCH_REQUESTED = "WRENCH_REQUESTED"
    COMMAND_ALLOCATED = "COMMAND_ALLOCATED"
    SAFETY_DECISION = "SAFETY_DECISION"
    SAFETY_STATE_CHANGED = "SAFETY_STATE_CHANGED"
    GOAL_COMPLETED = "GOAL_COMPLETED"


@dataclass(frozen=True)
class NavTraceRecord:
    record_type: str
    trace_id: UUID
    time_ns: int
    clock_domain: str
    goal_id: UUID | None = None
    trajectory_id: UUID | None = None
    wrench_id: UUID | None = None
    command_id: UUID | None = None
    payload: dict[str, Any] = field(default_factory=dict)


NavTraceSink = Callable[[NavTraceRecord], None]
