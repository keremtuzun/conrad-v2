"""Pure wire <-> Conrad conversions and checks used by ``UnityRobotHardware``.

Every function either returns a typed Conrad contract or raises :class:`UnityProtocolError`.
Nothing here holds state or touches a socket, so all of it is unit-testable without ZeroMQ.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from typing import Literal, Protocol
from uuid import UUID, uuid5

from pydantic import Field

from conrad.adapters.unity.frames import UNITY_FRAME_CONVENTION, UnityFrameMapper
from conrad.adapters.unity.protocol import (
    CAMERA_LAYOUT,
    DEPTH_LAYOUT,
    IMU_LAYOUT,
    SONAR_LAYOUT,
    HandshakeAck,
    SensorPacket,
    SimulationValidityLevel,
    StatePacket,
    UnityProtocolError,
    WireCapabilities,
    check_protocol_compatible,
)
from conrad.adapters.unity.transport import UnityPeerError
from conrad.schemas.base import ConradModel, SchemaVersionError, check_schema_compatible
from conrad.schemas.observation import MAX_INLINE_VALUES, Modality, Observation, PayloadRef, SensorHealth
from conrad.schemas.robot import (
    AllocatedCommand,
    BatteryState,
    DepthSample,
    HealthLevel,
    ImuSample,
    RobotCapabilities,
    SystemHealth,
    ThrusterState,
)
from conrad.schemas.timebase import TimeStamp

ADAPTER_NAME = "unity_v2"
_LEVEL_ORDER = list(SimulationValidityLevel)


class PayloadStore(Protocol):
    """Structural match for ``conrad.persistence.object_store.ObjectStore``."""

    def put_bytes(
        self, data: bytes, media_type: str, shape: tuple[int, ...] = (), dtype: str | None = None
    ) -> PayloadRef: ...


class UnityBridgeConfig(ConradModel):
    transport: Literal["tcp", "zmq"] = Field(
        default="tcp", description="tcp = length-prefixed JSON (what the Unity player serves); zmq = optional"
    )
    connect_timeout_ms: int = Field(default=2000, gt=0)
    control_endpoint: str = "tcp://127.0.0.1:5591"
    stream_endpoint: str | None = None
    allowed_peers: tuple[str, ...] = ("127.0.0.1",)
    allowed_simulator_ids: tuple[str, ...] = ("conrad-unity-v2",)
    clock_domain: str = "SIM"
    lock_step: bool = True
    request_timeout_ms: int = Field(default=2000, gt=0)
    client_name: str = "conrad-python"
    require_safety_authorization: bool = True
    minimum_validity_level: SimulationValidityLevel = SimulationValidityLevel.L0_FUNCTIONAL


class AdapterReason:
    CONFIG_DIGEST = "WRONG_ROBOT_CONFIG_DIGEST"
    CLOCK_DOMAIN = "WRONG_CLOCK_DOMAIN"
    NO_AUTH = "MISSING_SAFETY_AUTHORIZATION"
    ACTUATORS = "INVALID_ACTUATOR_SET"
    ENVELOPE = "COMMAND_OUT_OF_ENVELOPE"


def health_level(value: str) -> HealthLevel:
    try:
        return HealthLevel(value)
    except ValueError as exc:
        raise UnityProtocolError(f"unknown health level {value!r}") from exc


def verify_handshake(
    ack: HandshakeAck, cfg: UnityBridgeConfig, digest: str, nonce: str, thruster_ids: set[str]
) -> None:
    """Every identity/compatibility check of the handshake. Raises on the first failure."""
    if ack.simulator_id not in cfg.allowed_simulator_ids:
        raise UnityPeerError(f"unknown simulator peer {ack.simulator_id!r}")
    if ack.nonce != nonce:
        raise UnityPeerError("handshake nonce was not echoed; reply does not belong to this session")
    check_protocol_compatible(ack.protocol_version)
    try:
        check_schema_compatible(ack.schema_version)
    except SchemaVersionError as exc:
        raise UnityProtocolError(str(exc)) from exc
    if ack.clock_domain != cfg.clock_domain:
        raise UnityProtocolError(f"clock domain mismatch: {ack.clock_domain!r} != {cfg.clock_domain!r}")
    if ack.robot_config_digest != digest:
        raise UnityProtocolError("robot_config_digest mismatch: Unity loaded a different RobotConfig")
    if ack.frame_convention != UNITY_FRAME_CONVENTION:
        raise UnityProtocolError(f"unsupported wire frame convention {ack.frame_convention!r}")
    if ack.lock_step != cfg.lock_step:
        raise UnityProtocolError("simulator stepping mode differs from the configured lock_step")
    if _LEVEL_ORDER.index(ack.validity_level) < _LEVEL_ORDER.index(cfg.minimum_validity_level):
        raise UnityProtocolError(f"simulator validity {ack.validity_level.value} is below the minimum")
    if set(ack.capabilities.thruster_ids) != thruster_ids:
        raise UnityProtocolError("simulator thruster set differs from the RobotConfig")


def precheck_command(
    command: AllocatedCommand, digest: str, clock_domain: str, thruster_ids: set[str], require_auth: bool
) -> list[str]:
    """Adapter-side refusal reasons (defence in depth behind the Command Gateway)."""
    reasons: list[str] = []
    if command.robot_config_digest != digest:
        reasons.append(AdapterReason.CONFIG_DIGEST)
    if command.clock_domain != clock_domain:
        reasons.append(AdapterReason.CLOCK_DOMAIN)
    auth = command.safety_authorization
    if require_auth and (auth is None or auth.command_id != command.command_id):
        reasons.append(AdapterReason.NO_AUTH)
    if set(command.thruster_commands) != thruster_ids:
        reasons.append(AdapterReason.ACTUATORS)
    if any(not math.isfinite(v) or abs(v) > 1.0 for v in command.thruster_commands.values()):
        reasons.append(AdapterReason.ENVELOPE)
    return reasons


def capabilities_from_wire(caps: WireCapabilities) -> RobotCapabilities:
    return RobotCapabilities(
        capability_version=caps.capability_version,
        cameras=caps.cameras,
        sonars=caps.sonars,
        imu=caps.imu,
        depth=caps.depth,
        environmental_sensors=caps.environmental_sensors,
        thruster_count=len(caps.thruster_ids),
        controllable_dof=caps.controllable_dof,
        communication_links=caps.communication_links,
        onboard_compute=caps.onboard_compute,
    )


def acquisition_stamp(packet: SensorPacket) -> TimeStamp:
    """The acquisition time travels unchanged; delivery time is kept separately in the context."""
    return TimeStamp(
        time_ns=packet.acquisition_time_ns,
        clock_domain=packet.clock_domain,
        sequence_index=packet.sequence_index,
    )


def imu_from_packet(packet: SensorPacket, mapper: UnityFrameMapper) -> ImuSample:
    if packet.layout != IMU_LAYOUT or packet.shape != (10,):
        raise UnityProtocolError(f"IMU packet layout {packet.layout!r}/{packet.shape} is not {IMU_LAYOUT}")
    v = [float(x) for x in packet.payload_array()]
    quat = mapper.quat_to_conrad((v[6], v[7], v[8], v[9])) if all(map(math.isfinite, v[6:10])) else None
    return ImuSample(
        timestamp=acquisition_stamp(packet),
        frame_id=packet.frame_id,
        linear_acceleration_mps2=mapper.vector_to_conrad((v[0], v[1], v[2])),
        angular_velocity_rps=mapper.axial_to_conrad((v[3], v[4], v[5])),
        orientation_wxyz=quat,
    )


def depth_from_packet(packet: SensorPacket) -> DepthSample:
    if packet.layout != DEPTH_LAYOUT or packet.shape != (1,) or packet.units != "m":
        raise UnityProtocolError("depth packet must be depth_v1, shape (1,), units m")
    return DepthSample(
        timestamp=acquisition_stamp(packet),
        frame_id=packet.frame_id,
        depth_m=float(packet.payload_array()[0]),
        health=health_level(packet.health),
    )


def check_image_layout(packet: SensorPacket, modality: Modality) -> None:
    if modality is Modality.RGB and (
        packet.layout != CAMERA_LAYOUT or len(packet.shape) != 3 or packet.shape[2] != 3
    ):
        raise UnityProtocolError("camera packet must be rgb8_hwc_v1 with shape (H, W, 3)")
    if modality is Modality.SONAR and (packet.layout != SONAR_LAYOUT or len(packet.shape) != 2):
        raise UnityProtocolError("sonar packet must be sonar_beam_bin_v1 with shape (beams, bins)")


def observation_from_packet(
    packet: SensorPacket,
    modality: Modality,
    *,
    robot_id: UUID,
    mission_id: UUID,
    run_id: UUID,
    observation_id: UUID,
    trace_id: UUID,
    validity_level: SimulationValidityLevel,
    store: PayloadStore | None,
) -> Observation:
    check_image_layout(packet, modality)
    raw = packet.payload_bytes()
    array = packet.payload_array()
    payload_ref: PayloadRef | None = None
    inline: tuple[float, ...] | None = None
    if store is not None:
        media = "application/x-conrad-rgb8" if modality is Modality.RGB else "application/x-conrad-sonar"
        payload_ref = store.put_bytes(raw, media, packet.shape, str(array.dtype))
        if payload_ref.digest != packet.payload_digest:
            raise UnityProtocolError("payload store digest differs from the wire digest")
    elif array.size <= MAX_INLINE_VALUES:
        inline = tuple(float(x) for x in array.reshape(-1))
    else:
        raise UnityProtocolError(f"{packet.sensor_name}: {array.size} values need a PayloadStore")
    context = dict(packet.context)
    context.update(
        {
            "delivery_time_ns": packet.delivery_time_ns,
            "layout": packet.layout,
            "shape": list(packet.shape),
            "wire_payload_digest": packet.payload_digest,
            "simulator_validity_level": validity_level.value,
            "adapter": ADAPTER_NAME,
        }
    )
    return Observation(
        observation_id=observation_id,
        mission_id=mission_id,
        run_id=run_id,
        trace_id=trace_id,
        sensor_id=uuid5(robot_id, packet.sensor_name),
        modality=modality,
        timestamp=acquisition_stamp(packet),
        sensor_frame=packet.frame_id,
        robot_pose_estimate=None,
        payload_ref=payload_ref,
        inline_values=inline,
        inline_units=None if inline is None else packet.units,
        sensor_health=SensorHealth(health_level(packet.health).value),
        calibration_ref=packet.calibration_ref,
        sensor_context=context,
    )


def thrusters_from_state(state: StatePacket) -> tuple[ThrusterState, ...]:
    return tuple(
        ThrusterState(
            thruster_id=t.thruster_id,
            command=t.command,
            estimated_thrust_n=t.estimated_thrust_n,
            rpm=t.rpm,
            current_a=t.current_a,
            temperature_c=t.temperature_c,
            health=health_level(t.health),
        )
        for t in state.thrusters
    )


def battery_from_state(state: StatePacket) -> BatteryState | None:
    b = state.battery
    if b is None:
        return None
    return BatteryState(
        timestamp=TimeStamp(time_ns=b.time_ns, clock_domain=state.clock_domain),
        remaining_fraction=b.remaining_fraction,
        voltage_v=b.voltage_v,
        current_a=b.current_a,
        energy_used_j=b.energy_used_j,
    )


def health_from_state(state: StatePacket) -> SystemHealth:
    h = state.health
    return SystemHealth(
        timestamp=TimeStamp(time_ns=h.time_ns, clock_domain=state.clock_domain),
        overall=health_level(h.overall),
        leak_detected=h.leak_detected,
        devices={k: health_level(v) for k, v in h.devices.items()},
        faults=h.faults,
    )
