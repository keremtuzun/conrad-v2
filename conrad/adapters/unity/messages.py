"""Non-sensor message bodies of the Unity bridge (handshake, reset, step, state, command, faults).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from conrad.adapters.unity.sensor_packet import SensorPacket
from conrad.adapters.unity.vocabulary import (
    DIGEST_PATTERN,
    PROTOCOL_VERSION,
    ClientRole,
    FaultType,
    SimulationValidityLevel,
    WireModel,
)
from conrad.schemas.base import SCHEMA_VERSION
from conrad.schemas.robot import AllocatedCommand

_DIGEST = DIGEST_PATTERN


class WireCapabilities(WireModel):
    capability_version: str
    cameras: tuple[str, ...] = ()
    sonars: tuple[str, ...] = ()
    range_imagers: tuple[str, ...] = ()  # DEPTH_RANGE sensors (range_f32_hw_v1); optional, older players omit it
    imu: bool = False
    depth: bool = False
    environmental_sensors: tuple[str, ...] = ()
    thruster_ids: tuple[str, ...] = ()
    controllable_dof: tuple[str, ...] = ()
    communication_links: tuple[str, ...] = ()
    onboard_compute: str | None = None


class HandshakeRequest(WireModel):
    client_name: str = Field(min_length=1)
    role: ClientRole = ClientRole.CONTROL
    protocol_version: str = PROTOCOL_VERSION
    schema_version: str = SCHEMA_VERSION
    robot_config_digest: str = Field(pattern=_DIGEST)
    clock_domain: str = Field(min_length=1)
    frame_convention: str = Field(min_length=1)
    lock_step: bool = True
    nonce: str = Field(min_length=8)


class HandshakeAck(WireModel):
    simulator_id: str = Field(min_length=1)
    simulator_version: str
    protocol_version: str
    schema_version: str
    robot_config_digest: str = Field(pattern=_DIGEST)
    clock_domain: str = Field(min_length=1)
    frame_convention: str
    lock_step: bool
    session_id: str = Field(min_length=1)
    nonce: str
    validity_level: SimulationValidityLevel
    physics_dt_ns: int = Field(gt=0)
    sim_time_ns: int = Field(ge=0)
    capabilities: WireCapabilities


class ResetRequest(WireModel):
    seed: int
    scenario_ref: str | None = None
    scenario_digest: str | None = Field(default=None, pattern=_DIGEST)
    experiment: dict[str, Any] = Field(default_factory=dict)


class ResetAck(WireModel):
    seed: int
    sim_time_ns: int = Field(ge=0)
    scenario_digest: str | None = None


class StepRequest(WireModel):
    step_index: int = Field(ge=0)
    dt_ns: int = Field(gt=0)


class PollRequest(WireModel):
    pass


class WireThrusterState(WireModel):
    thruster_id: str
    command: float = Field(ge=-1, le=1)
    estimated_thrust_n: float
    rpm: float | None = None
    current_a: float | None = None
    temperature_c: float | None = None
    health: str = "OK"


class WireBattery(WireModel):
    time_ns: int = Field(ge=0)
    remaining_fraction: float = Field(ge=0, le=1)
    voltage_v: float | None = None
    current_a: float | None = None
    energy_used_j: float = Field(default=0.0, ge=0)


class WireHealth(WireModel):
    time_ns: int = Field(ge=0)
    overall: str
    leak_detected: bool | None = None
    devices: dict[str, str] = Field(default_factory=dict)
    faults: tuple[str, ...] = ()


class StatePacket(WireModel):
    """Reply to STEP and POLL: everything the RobotHardwareInterface can report, never truth."""

    step_index: int | None = None
    sim_time_ns: int = Field(ge=0)
    clock_domain: str = Field(min_length=1)
    sensors: tuple[SensorPacket, ...] = ()
    thrusters: tuple[WireThrusterState, ...] = ()
    battery: WireBattery | None = None
    health: WireHealth
    queue_backlog: int = Field(default=0, ge=0)
    dropped_frames: int = Field(default=0, ge=0)


class CommandPacket(WireModel):
    command_id: str
    trace_id: str
    robot_config_digest: str = Field(pattern=_DIGEST)
    clock_domain: str
    issued_time_ns: int = Field(ge=0)
    deadline_ns: int = Field(ge=0)
    thruster_commands: dict[str, float]
    safety_authorization_id: str | None = None
    safety_state: str | None = None

    @classmethod
    def from_allocated(cls, command: AllocatedCommand) -> CommandPacket:
        auth = command.safety_authorization
        return cls(
            command_id=str(command.command_id),
            trace_id=str(command.trace_id),
            robot_config_digest=command.robot_config_digest,
            clock_domain=command.clock_domain,
            issued_time_ns=command.issued_time_ns,
            deadline_ns=command.deadline_ns,
            thruster_commands=dict(command.thruster_commands),
            safety_authorization_id=None if auth is None else str(auth.authorization_id),
            safety_state=None if auth is None else auth.safety_state,
        )


class WireCommandAck(WireModel):
    command_id: str
    accepted: bool
    reason_codes: tuple[str, ...] = ()
    ack_time_ns: int = Field(ge=0)
    sim_time_ns: int = Field(ge=0)


class FaultInjectionRequest(WireModel):
    fault_id: str = Field(min_length=1)
    fault_type: FaultType
    target: str | None = Field(default=None, description="thruster_id / sensor_name / link_name")
    start_time_ns: int = Field(ge=0)
    duration_ns: int | None = Field(default=None, gt=0, description="None = until reset")
    magnitude: float = Field(default=1.0, description="fault-type specific strength, SI or fraction")
    parameters: dict[str, float] = Field(default_factory=dict)


class FaultAck(WireModel):
    fault_id: str
    accepted: bool
    scheduled_time_ns: int = Field(ge=0)
    reason_codes: tuple[str, ...] = ()


class MetricsRequest(WireModel):
    pass


class MetricsReply(WireModel):
    sim_time_ns: int = Field(ge=0)
    validity_level: SimulationValidityLevel
    metrics: dict[str, float] = Field(default_factory=dict)


class SceneConfigureRequest(WireModel):
    """Static collider geometry in the Conrad WORLD frame (built by ``conrad.sim.unity.scene``).

    Primitive dicts are validated strictly on the sim side; Unity validates them again and refuses the
    whole request (leaving the previous world untouched) on the first bad primitive.
    """

    frame: str = "WORLD"
    replace: bool = True
    primitives: tuple[dict[str, Any], ...]


class SceneAck(WireModel):
    scene_digest: str = Field(pattern=_DIGEST)
    primitive_count: int = Field(ge=0)
    sim_time_ns: int = Field(ge=0)


class ProbePose(WireModel):
    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]


class FrameProbeRequest(WireModel):
    poses: tuple[ProbePose, ...] = Field(min_length=1, max_length=256)


class FrameProbeResult(WireModel):
    """Engine answers (Unity convention) plus the pose converted back to Conrad by the C# frame map."""

    unity_position: tuple[float, float, float]
    unity_rotation_wxyz: tuple[float, float, float, float]
    unity_forward: tuple[float, float, float]
    unity_right: tuple[float, float, float]
    unity_up: tuple[float, float, float]
    conrad_position: tuple[float, float, float]
    conrad_orientation_wxyz: tuple[float, float, float, float]


class FrameProbeReply(WireModel):
    results: tuple[FrameProbeResult, ...]


class ErrorReply(WireModel):
    code: str
    detail: str = ""
