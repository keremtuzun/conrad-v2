"""Command Gateway: the sole code path allowed to call ``RobotHardwareInterface.send`` (ch34, ch36).

Accepts only a safety-approved, non-expired, typed :class:`AllocatedCommand`. Raw PWM, untyped
payloads and direct model outputs are rejected before transport. Fails closed.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any
from uuid import UUID

from conrad.robotics.hardware.interface import RobotHardwareInterface
from conrad.schemas.base import SCHEMA_VERSION, SchemaVersionError, check_schema_compatible
from conrad.schemas.events import EventType, Severity
from conrad.schemas.robot import AllocatedCommand, CommandAck, HealthLevel, RobotConfig
from conrad.settings import CommandMode, ExecutionLane, RuntimeSettings

ALLOCATOR_PRODUCER = "conrad.robotics.allocation"

Emit = Callable[..., Any]


class GatewayReason:
    NOT_TYPED = "NOT_ALLOCATED_COMMAND"
    SCHEMA = "INCOMPATIBLE_SCHEMA"
    MODE_DISABLED = "COMMAND_MODE_DISABLED"
    HARDWARE_NOT_ENABLED = "HARDWARE_MODE_NOT_ENABLED"
    HARDWARE_PREREQ = "HARDWARE_PREREQUISITES_MISSING"
    ADAPTER_LANE = "ADAPTER_NOT_PERMITTED_IN_LANE"
    UNKNOWN_MISSION = "UNKNOWN_MISSION"
    UNKNOWN_RUN = "UNKNOWN_RUN"
    UNKNOWN_PEER = "UNKNOWN_PEER"
    CLOCK_DOMAIN = "WRONG_CLOCK_DOMAIN"
    EXPIRED = "EXPIRED_DEADLINE"
    FUTURE_ISSUE = "ISSUED_IN_FUTURE"
    CONFIG_DIGEST = "WRONG_ROBOT_CONFIG_DIGEST"
    NO_AUTH = "MISSING_SAFETY_AUTHORIZATION"
    AUTH_MISMATCH = "SAFETY_AUTHORIZATION_MISMATCH"
    AUTH_STATE = "SAFETY_STATE_FORBIDS_MOTION"
    PRODUCER = "NOT_PRODUCED_BY_ALLOCATOR"
    ACTUATOR_COUNT = "INVALID_ACTUATOR_SET"
    ENVELOPE = "COMMAND_OUT_OF_ENVELOPE"
    STALE_STATE = "STALE_STATE"
    HEALTH = "INVALID_HEALTH_STATE"
    DUPLICATE = "DUPLICATE_COMMAND_ID"


MOTION_PERMITTED_STATES = frozenset({"NORMAL", "DEGRADED", "HOLD"})


class CommandGateway:
    def __init__(
        self,
        hardware: RobotHardwareInterface,
        robot_config: RobotConfig,
        settings: RuntimeSettings,
        lane: ExecutionLane,
        mission_id: UUID,
        run_id: UUID,
        emit: Emit | None = None,
        state_age_s: Callable[[], float | None] | None = None,
    ) -> None:
        self._hw = hardware
        self._config = robot_config
        self._config_digest = robot_config.content_digest()
        self._settings = settings
        self._lane = lane
        self._mission_id = mission_id
        self._run_id = run_id
        self._emit = emit
        self._state_age_s = state_age_s
        self._seen: set[UUID] = set()
        self.accepted = 0
        self.rejected = 0

    @property
    def robot_config_digest(self) -> str:
        return self._config_digest

    def _mode_reasons(self) -> list[str]:
        mode = self._settings.command_mode
        reasons: list[str] = []
        if mode is CommandMode.DISABLED:
            reasons.append(GatewayReason.MODE_DISABLED)
        if self._hw.is_physical:
            if mode is not CommandMode.HARDWARE or not self._settings.hardware_enable:
                reasons.append(GatewayReason.HARDWARE_NOT_ENABLED)
            if self._lane not in (ExecutionLane.HIL, ExecutionLane.PHYSICAL):
                reasons.append(GatewayReason.ADAPTER_LANE)
            if self._lane is ExecutionLane.PHYSICAL and (
                not self._settings.hil_evidence_ref or self._config.ungrounded_parameters()
            ):
                reasons.append(GatewayReason.HARDWARE_PREREQ)
        elif mode is CommandMode.HARDWARE:
            reasons.append(GatewayReason.ADAPTER_LANE)
        return reasons

    def validate(self, command: Any, peer: str = "127.0.0.1") -> list[str]:
        if not isinstance(command, AllocatedCommand):
            return [GatewayReason.NOT_TYPED]
        reasons = self._mode_reasons()
        try:
            check_schema_compatible(command.schema_version, SCHEMA_VERSION)
        except SchemaVersionError:
            reasons.append(GatewayReason.SCHEMA)
        if peer not in self._settings.allowed_peers:
            reasons.append(GatewayReason.UNKNOWN_PEER)
        if command.mission_id != self._mission_id:
            reasons.append(GatewayReason.UNKNOWN_MISSION)
        if command.run_id != self._run_id:
            reasons.append(GatewayReason.UNKNOWN_RUN)
        if command.clock_domain != self._hw.clock_domain():
            reasons.append(GatewayReason.CLOCK_DOMAIN)
        now = self._hw.now_ns()
        if now >= command.deadline_ns:
            reasons.append(GatewayReason.EXPIRED)
        if command.issued_time_ns > now:
            reasons.append(GatewayReason.FUTURE_ISSUE)
        if (
            command.robot_config_digest != self._config_digest
            or self._hw.robot_config_digest() != self._config_digest
        ):
            reasons.append(GatewayReason.CONFIG_DIGEST)
        if command.producer != ALLOCATOR_PRODUCER:
            reasons.append(GatewayReason.PRODUCER)
        auth = command.safety_authorization
        if auth is None:
            reasons.append(GatewayReason.NO_AUTH)
        else:
            if auth.command_id != command.command_id:
                reasons.append(GatewayReason.AUTH_MISMATCH)
            if auth.safety_state not in MOTION_PERMITTED_STATES:
                reasons.append(GatewayReason.AUTH_STATE)
        expected = {t.thruster_id for t in self._config.thrusters}
        if not expected or set(command.thruster_commands) != expected:
            reasons.append(GatewayReason.ACTUATOR_COUNT)
        if any((not math.isfinite(v)) or abs(v) > 1.0 for v in command.thruster_commands.values()):
            reasons.append(GatewayReason.ENVELOPE)
        if self._state_age_s is not None:
            age = self._state_age_s()
            limit = self._config.safety.state_stale_after_s.value
            if age is None or limit is None or age > limit:
                reasons.append(GatewayReason.STALE_STATE)
        if self._hw.get_health().overall is HealthLevel.FAULT and any(
            abs(v) > 0 for v in command.thruster_commands.values()
        ):
            reasons.append(GatewayReason.HEALTH)
        if command.command_id in self._seen:
            reasons.append(GatewayReason.DUPLICATE)
        return reasons

    def submit(self, command: Any, peer: str = "127.0.0.1") -> CommandAck:
        reasons = self.validate(command, peer)
        typed = isinstance(command, AllocatedCommand)
        trace = command.trace_id if typed else self._run_id
        if reasons:
            self.rejected += 1
            ack = CommandAck(
                command_id=command.command_id if typed else self._run_id,
                accepted=False,
                reason_codes=tuple(reasons),
                ack_time_ns=self._hw.now_ns() if not self._hw.is_physical else 0,
                adapter="command_gateway",
            )
            if self._emit:
                self._emit(
                    EventType.COMMAND_REJECTED,
                    "conrad.runtime.command_gateway",
                    trace,
                    {"command_id": str(ack.command_id), "reason_codes": list(reasons), "typed": typed},
                    severity=Severity.WARNING,
                )
            return ack
        assert isinstance(command, AllocatedCommand)
        self._seen.add(command.command_id)
        if self._emit:
            self._emit(
                EventType.COMMAND_SENT,
                "conrad.runtime.command_gateway",
                trace,
                {
                    "command_id": str(command.command_id),
                    "source_wrench_id": str(command.source_wrench_id),
                    "thruster_commands": command.thruster_commands,
                    "deadline_ns": command.deadline_ns,
                    "provenance_root": str(command.provenance_root),
                },
                measurement_time_ns=command.issued_time_ns,
            )
        ack = self._hw.send(command)
        self.accepted += int(ack.accepted)
        if self._emit:
            self._emit(
                EventType.COMMAND_ACK,
                "conrad.runtime.command_gateway",
                trace,
                {
                    "command_id": str(command.command_id),
                    "accepted": ack.accepted,
                    "reason_codes": list(ack.reason_codes),
                },
            )
        return ack
