"""Append-only runtime event contract (ch34 Runtime process model, ch36 Replay/observability).

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import Field

from conrad.schemas.base import ARCHITECTURE_ID, STACK_ID, VersionedModel
from conrad.schemas.belief import Availability
from conrad.schemas.envelope import MessageEnvelope


class EventType(str, Enum):
    RUN_STARTED = "RUN_STARTED"
    STATE_CHANGED = "STATE_CHANGED"
    OBSERVATION_RECEIVED = "OBSERVATION_RECEIVED"
    EVIDENCE_CREATED = "EVIDENCE_CREATED"
    EVIDENCE_DUPLICATE_DROPPED = "EVIDENCE_DUPLICATE_DROPPED"
    LATE_EVIDENCE_HANDLED = "LATE_EVIDENCE_HANDLED"
    BELIEF_COMMITTED = "BELIEF_COMMITTED"
    BELIEF_PUBLISHED = "BELIEF_PUBLISHED"
    DECISION_MADE = "DECISION_MADE"
    PLAN_PROPOSED = "PLAN_PROPOSED"
    ACTION_REJECTED = "ACTION_REJECTED"
    GOAL_ACCEPTED = "GOAL_ACCEPTED"
    TRAJECTORY_PLANNED = "TRAJECTORY_PLANNED"
    WRENCH_REQUESTED = "WRENCH_REQUESTED"
    COMMAND_SENT = "COMMAND_SENT"
    COMMAND_ACK = "COMMAND_ACK"
    COMMAND_REJECTED = "COMMAND_REJECTED"
    TRANSMISSION = "TRANSMISSION"
    OPERATOR_ACTION = "OPERATOR_ACTION"
    FAULT_DETECTED = "FAULT_DETECTED"
    FAULT_INJECTED = "FAULT_INJECTED"
    SAFE_HOLD_ENTERED = "SAFE_HOLD_ENTERED"
    RUN_TERMINATED = "RUN_TERMINATED"


class Severity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RuntimeEvent(VersionedModel):
    event_id: UUID
    sequence: int = Field(ge=0, description="monotonic per-run sequence; defines the replayable order")
    envelope: MessageEnvelope
    event_type: EventType
    severity: Severity = Severity.INFO
    availability: Availability = Availability.AVAILABLE
    trace_id: UUID
    module: str
    module_version: str
    payload_digest: str = Field(pattern="^[0-9a-f]{64}$")
    payload: dict[str, Any] = Field(default_factory=dict)
    architecture_id: str = ARCHITECTURE_ID
    stack_id: str = STACK_ID
