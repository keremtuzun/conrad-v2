"""Versioned Conrad <-> Unity wire protocol: envelope, endpoint registry, encode/decode.

One envelope shape for every message::

    {"protocol_version": "1.0.0", "schema_version": "1.0.0", "kind": "STEP",
     "session_id": "...", "seq": 12, "body": {...}}

Ground-truth bodies are NOT registered on the control endpoint; they live in ``conrad.sim.unity``.
This module re-exports the whole wire vocabulary so callers need a single import.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field, ValidationError

from conrad.adapters.unity.messages import (
    CommandPacket,
    ErrorReply,
    FaultAck,
    FaultInjectionRequest,
    HandshakeAck,
    HandshakeRequest,
    MetricsReply,
    MetricsRequest,
    PollRequest,
    ResetAck,
    ResetRequest,
    StatePacket,
    StepRequest,
    WireBattery,
    WireCapabilities,
    WireCommandAck,
    WireHealth,
    WireThrusterState,
)
from conrad.adapters.unity.sensor_packet import SensorPacket, make_sensor_packet
from conrad.adapters.unity.vocabulary import (
    CAMERA_LAYOUT,
    DEPTH_LAYOUT,
    IMU_LAYOUT,
    PROTOCOL_VERSION,
    SONAR_LAYOUT,
    ClientRole,
    FaultType,
    MessageKind,
    PayloadEncoding,
    SimulationValidityLevel,
    UnityProtocolError,
    WireModel,
)
from conrad.schemas.base import SCHEMA_VERSION, ConradModel, SchemaVersionError, check_schema_compatible

__all__ = [
    "CAMERA_LAYOUT",
    "DEPTH_LAYOUT",
    "IMU_LAYOUT",
    "PROTOCOL_VERSION",
    "SONAR_LAYOUT",
    "BodyRegistry",
    "ClientRole",
    "CommandPacket",
    "ErrorReply",
    "FaultAck",
    "FaultInjectionRequest",
    "FaultType",
    "HandshakeAck",
    "HandshakeRequest",
    "MessageKind",
    "MetricsReply",
    "MetricsRequest",
    "PayloadEncoding",
    "PollRequest",
    "ResetAck",
    "ResetRequest",
    "SensorPacket",
    "SimulationValidityLevel",
    "StatePacket",
    "StepRequest",
    "UnityProtocolError",
    "WireBattery",
    "WireCapabilities",
    "WireCommandAck",
    "WireEnvelope",
    "WireHealth",
    "WireModel",
    "WireThrusterState",
    "check_protocol_compatible",
    "decode_message",
    "encode_message",
    "make_sensor_packet",
]


class WireEnvelope(ConradModel):
    protocol_version: str
    schema_version: str
    kind: MessageKind
    session_id: str = ""
    seq: int = Field(ge=0)
    body: dict[str, Any]


BodyRegistry = dict[MessageKind, type[WireModel]]

CONTROL_BODIES: BodyRegistry = {
    MessageKind.HANDSHAKE: HandshakeRequest,
    MessageKind.HANDSHAKE_ACK: HandshakeAck,
    MessageKind.RESET: ResetRequest,
    MessageKind.RESET_ACK: ResetAck,
    MessageKind.STEP: StepRequest,
    MessageKind.POLL: PollRequest,
    MessageKind.STATE: StatePacket,
    MessageKind.COMMAND: CommandPacket,
    MessageKind.COMMAND_ACK: WireCommandAck,
    MessageKind.INJECT_FAULT: FaultInjectionRequest,
    MessageKind.FAULT_ACK: FaultAck,
    MessageKind.GET_METRICS: MetricsRequest,
    MessageKind.METRICS: MetricsReply,
    MessageKind.SENSOR: SensorPacket,
    MessageKind.ERROR: ErrorReply,
}


def check_protocol_compatible(found: str, expected: str = PROTOCOL_VERSION) -> None:
    try:
        found_major = int(found.split(".")[0])
        expected_major = int(expected.split(".")[0])
    except (ValueError, IndexError) as exc:
        raise UnityProtocolError(f"unparseable protocol version {found!r}") from exc
    if found_major != expected_major:
        raise UnityProtocolError(
            f"protocol major {found_major} is incompatible with supported major {expected_major}"
        )


def encode_message(
    kind: MessageKind,
    body: WireModel,
    *,
    session_id: str,
    seq: int,
    protocol_version: str = PROTOCOL_VERSION,
    schema_version: str = SCHEMA_VERSION,
) -> bytes:
    env = WireEnvelope(
        protocol_version=protocol_version,
        schema_version=schema_version,
        kind=kind,
        session_id=session_id,
        seq=seq,
        body=body.model_dump(mode="json"),
    )
    return env.canonical_json().encode("utf-8")


def decode_message(data: bytes, registry: BodyRegistry | None = None) -> tuple[WireEnvelope, WireModel]:
    """Parse + validate one message. Any deviation raises :class:`UnityProtocolError`."""
    reg = CONTROL_BODIES if registry is None else registry
    try:
        raw = json.loads(data.decode("utf-8"))
        env = WireEnvelope.model_validate(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise UnityProtocolError(f"malformed envelope: {exc}") from exc
    check_protocol_compatible(env.protocol_version)
    try:
        check_schema_compatible(env.schema_version)
    except SchemaVersionError as exc:
        raise UnityProtocolError(str(exc)) from exc
    model = reg.get(env.kind)
    if model is None:
        raise UnityProtocolError(f"message kind {env.kind.value} is not permitted on this endpoint")
    try:
        return env, model.model_validate(env.body)
    except ValidationError as exc:
        raise UnityProtocolError(f"invalid {env.kind.value} body: {exc}") from exc
