"""Robot-side contracts: sourced physical parameters, RobotConfig, capabilities, state, commands.

A physical fact that was not supplied by the hardware owner stays OPEN (INV-ARCH-08). Simulation
may use explicitly labelled estimates; nothing here converts a missing value into a measurement.

implementation_status: FROZEN_CONTRACT
source_sections: [ch1 Hardware Handoff, ch20 RobotConfig/RHI, ch22, ch34 command gateway]
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import Field, model_validator

from conrad.schemas.base import VersionedModel
from conrad.schemas.belief import Availability
from conrad.schemas.frames import Pose, Quat, Vec3
from conrad.schemas.timebase import TimeStamp

T = TypeVar("T")


class SourceKind(str, Enum):
    MEASURED = "MEASURED"
    IDENTIFIED = "IDENTIFIED"
    LITERATURE_PRIOR = "LITERATURE_PRIOR"
    ENGINEERING_ESTIMATE = "ENGINEERING_ESTIMATE"
    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
    OPEN = "OPEN"


PHYSICALLY_GROUNDED = frozenset({SourceKind.MEASURED, SourceKind.IDENTIFIED})


class Sourced(VersionedModel, Generic[T]):
    """A physical parameter with units, frame, range and provenance of the number itself."""

    value: T | None
    units: str = Field(min_length=1)
    source: SourceKind
    frame_id: str | None = None
    measured_at: TimeStamp | None = None
    uncertainty_1sigma: float | None = Field(default=None, ge=0)
    valid_range: tuple[float, float] | None = None
    provenance: str | None = Field(default=None, description="raw-log / report / citation reference")

    @model_validator(mode="after")
    def _open_has_no_value(self) -> Sourced[T]:
        if self.source is SourceKind.OPEN and self.value is not None:
            raise ValueError("an OPEN parameter cannot carry a value")
        if self.source is not SourceKind.OPEN and self.value is None:
            raise ValueError(f"{self.source.value} parameter requires a value")
        if self.source in PHYSICALLY_GROUNDED and not self.provenance:
            raise ValueError("MEASURED/IDENTIFIED values require provenance (raw log or report reference)")
        return self

    @property
    def is_open(self) -> bool:
        return self.source is SourceKind.OPEN

    def require(self, name: str) -> T:
        if self.value is None:
            raise OpenParameterError(f"RobotConfig parameter {name!r} is OPEN")
        return self.value


class OpenParameterError(RuntimeError):
    """A consumer needed a physical parameter whose status is OPEN."""


class ThrusterConfig(VersionedModel):
    thruster_id: str = Field(min_length=1)
    position_body_m: Sourced[Vec3]
    direction_body: Sourced[Vec3]
    max_forward_thrust_n: Sourced[float]
    max_reverse_thrust_n: Sourced[float]
    deadzone_command: Sourced[float]
    time_constant_s: Sourced[float]
    latency_s: Sourced[float]
    thrust_coefficient: Sourced[float] = Field(description="k in T = k u|u|")


class SensorConfig(VersionedModel):
    sensor_name: str
    modality: str
    frame_id: str
    mount_pose_body: Sourced[tuple[float, ...]]
    rate_hz: Sourced[float]
    noise_std: Sourced[float]
    bias: Sourced[float]
    drift_per_s: Sourced[float]
    latency_s: Sourced[float]
    calibration_ref: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class BatteryConfig(VersionedModel):
    capacity_j: Sourced[float]
    nominal_voltage_v: Sourced[float]
    max_current_a: Sourced[float]
    reserve_fraction: Sourced[float]


class CommsLinkConfig(VersionedModel):
    link_name: str
    bandwidth_bps: Sourced[float]
    latency_s: Sourced[float]
    packet_loss: Sourced[float]
    bit_error_rate: Sourced[float]


class ComputeConfig(VersionedModel):
    description: str
    ram_bytes: Sourced[float]
    thermal_limit_c: Sourced[float]


class SafetyEnvelope(VersionedModel):
    """Mission/RobotConfig-specific safe-hold behaviour. Never hard-coded to 'surface' (ch28, ch34)."""

    safe_hold_action: Sourced[str]
    max_depth_m: Sourced[float]
    min_altitude_m: Sourced[float]
    max_speed_mps: Sourced[float]
    max_pose_sigma_m: Sourced[float]
    min_battery_fraction: Sourced[float]
    state_stale_after_s: Sourced[float]
    command_timeout_s: Sourced[float]


class RobotConfig(VersionedModel):
    robot_id: UUID
    config_name: str
    config_version: str
    mass_kg: Sourced[float]
    displaced_volume_m3: Sourced[float]
    center_of_mass_body_m: Sourced[Vec3]
    center_of_buoyancy_body_m: Sourced[Vec3]
    inertia_diag_kgm2: Sourced[Vec3]
    dimensions_m: Sourced[Vec3]
    linear_drag: Sourced[tuple[float, ...]]
    quadratic_drag: Sourced[tuple[float, ...]]
    added_mass_diag: Sourced[tuple[float, ...]]
    thrusters: tuple[ThrusterConfig, ...]
    sensors: tuple[SensorConfig, ...]
    battery: BatteryConfig
    compute: ComputeConfig
    communications: tuple[CommsLinkConfig, ...]
    safety: SafetyEnvelope

    @model_validator(mode="after")
    def _unique_thrusters(self) -> RobotConfig:
        names = [t.thruster_id for t in self.thrusters]
        if len(names) != len(set(names)):
            raise ValueError("duplicate thruster_id")
        return self

    def open_parameters(self) -> list[str]:
        """Dotted names of every parameter whose status is OPEN."""
        found: list[str] = []

        def walk(obj: Any, path: str) -> None:
            if isinstance(obj, Sourced):
                if obj.is_open:
                    found.append(path)
            elif isinstance(obj, VersionedModel):
                for name in type(obj).model_fields:
                    walk(getattr(obj, name), f"{path}.{name}" if path else name)
            elif isinstance(obj, tuple):
                for i, item in enumerate(obj):
                    walk(item, f"{path}[{i}]")

        walk(self, "")
        return found

    def ungrounded_parameters(self) -> list[str]:
        """Parameters that are not MEASURED/IDENTIFIED: insufficient for a physical lane."""
        found: list[str] = []

        def walk(obj: Any, path: str) -> None:
            if isinstance(obj, Sourced):
                if obj.source not in PHYSICALLY_GROUNDED:
                    found.append(path)
            elif isinstance(obj, VersionedModel):
                for name in type(obj).model_fields:
                    walk(getattr(obj, name), f"{path}.{name}" if path else name)
            elif isinstance(obj, tuple):
                for i, item in enumerate(obj):
                    walk(item, f"{path}[{i}]")

        walk(self, "")
        return found


class RobotCapabilities(VersionedModel):
    capability_version: str
    cameras: tuple[str, ...] = ()
    sonars: tuple[str, ...] = ()
    imu: bool = False
    depth: bool = False
    environmental_sensors: tuple[str, ...] = ()
    thruster_count: int = Field(ge=0)
    controllable_dof: tuple[str, ...] = ()
    communication_links: tuple[str, ...] = ()
    onboard_compute: str | None = None


class HealthLevel(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAULT = "FAULT"
    UNKNOWN = "UNKNOWN"


class BatteryState(VersionedModel):
    timestamp: TimeStamp
    remaining_fraction: float = Field(ge=0, le=1)
    voltage_v: float | None = None
    current_a: float | None = None
    energy_used_j: float = Field(default=0.0, ge=0)


class ThrusterState(VersionedModel):
    thruster_id: str
    command: float = Field(ge=-1, le=1)
    estimated_thrust_n: float
    rpm: float | None = None
    current_a: float | None = None
    temperature_c: float | None = None
    health: HealthLevel = HealthLevel.OK


class SystemHealth(VersionedModel):
    timestamp: TimeStamp
    overall: HealthLevel
    leak_detected: bool | None = None
    devices: dict[str, HealthLevel] = Field(default_factory=dict)
    module_availability: dict[str, Availability] = Field(default_factory=dict)
    faults: tuple[str, ...] = ()


class RobotState(VersionedModel):
    """ESTIMATED state (X-hat, P). The simulator's true state never uses this type in deployment paths."""

    timestamp: TimeStamp
    pose: Pose
    linear_velocity_body_mps: Vec3 | None = None
    angular_velocity_body_rps: Vec3 | None = None
    estimator_health: HealthLevel
    estimator_name: str
    battery: BatteryState | None = None


class ImuSample(VersionedModel):
    timestamp: TimeStamp
    frame_id: str
    linear_acceleration_mps2: Vec3
    angular_velocity_rps: Vec3
    orientation_wxyz: Quat | None = None


class DepthSample(VersionedModel):
    timestamp: TimeStamp
    frame_id: str
    depth_m: float
    health: HealthLevel = HealthLevel.OK


class WrenchCommand(VersionedModel):
    """Controller output: desired body wrench. Not executable by itself."""

    command_id: UUID
    trace_id: UUID
    timestamp: TimeStamp
    frame_id: str
    force_n: Vec3
    torque_nm: Vec3


class SafetyAuthorization(VersionedModel):
    authorization_id: UUID
    command_id: UUID
    supervisor_version: str
    issued_time_ns: int = Field(ge=0)
    safety_state: str
    reason_codes: tuple[str, ...] = ()


class AllocatedCommand(VersionedModel):
    """The ONLY object the Command Gateway accepts (ch34). Raw PWM / untyped payloads do not exist here."""

    command_id: UUID
    mission_id: UUID
    run_id: UUID
    trace_id: UUID
    belief_snapshot_id: UUID | None
    robot_config_digest: str = Field(pattern="^[0-9a-f]{64}$")
    clock_domain: str
    issued_time_ns: int = Field(ge=0)
    deadline_ns: int = Field(ge=0)
    thruster_commands: dict[str, float]
    source_wrench_id: UUID
    provenance_root: UUID
    producer: str = Field(description="module that produced the command; must be the allocator")
    safety_authorization: SafetyAuthorization | None = None

    @model_validator(mode="after")
    def _deadline(self) -> AllocatedCommand:
        if self.deadline_ns <= self.issued_time_ns:
            raise ValueError("deadline must follow issue time")
        return self


class CommandAck(VersionedModel):
    command_id: UUID
    accepted: bool
    reason_codes: tuple[str, ...] = ()
    ack_time_ns: int = Field(ge=0)
    adapter: str
