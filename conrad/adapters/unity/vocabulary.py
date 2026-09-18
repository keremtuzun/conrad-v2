"""Wire vocabulary shared by every Unity bridge message: errors, enums, layouts, base model.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from enum import Enum

from conrad.schemas.base import ConradModel

PROTOCOL_VERSION = "1.0.0"
DIGEST_PATTERN = "^[0-9a-f]{64}$"


class UnityProtocolError(ValueError):
    """The peer spoke a malformed, incompatible or unexpected message. Always fail closed."""


class MessageKind(str, Enum):
    HANDSHAKE = "HANDSHAKE"
    HANDSHAKE_ACK = "HANDSHAKE_ACK"
    RESET = "RESET"
    RESET_ACK = "RESET_ACK"
    STEP = "STEP"
    POLL = "POLL"
    STATE = "STATE"
    COMMAND = "COMMAND"
    COMMAND_ACK = "COMMAND_ACK"
    INJECT_FAULT = "INJECT_FAULT"
    FAULT_ACK = "FAULT_ACK"
    GET_METRICS = "GET_METRICS"
    METRICS = "METRICS"
    SENSOR = "SENSOR"  # PUB stream
    GET_GROUND_TRUTH = "GET_GROUND_TRUTH"  # truth endpoint only
    GROUND_TRUTH = "GROUND_TRUTH"  # truth endpoint only
    ERROR = "ERROR"


class ClientRole(str, Enum):
    CONTROL = "CONTROL"
    TRUTH = "TRUTH"


class SimulationValidityLevel(str, Enum):
    """ch20/ch21 simulation validity levels. Only L4 may be called a validated twin, and only within its envelope."""

    L0_FUNCTIONAL = "L0_FUNCTIONAL"
    L1_APPROXIMATE_PHYSICS = "L1_APPROXIMATE_PHYSICS"
    L2_CHARACTERIZED = "L2_CHARACTERIZED"
    L3_IDENTIFIED = "L3_IDENTIFIED"
    L4_VALIDATED_ENVELOPE = "L4_VALIDATED_ENVELOPE"


class FaultType(str, Enum):
    THRUSTER_FAILURE = "THRUSTER_FAILURE"
    THRUSTER_DEGRADATION = "THRUSTER_DEGRADATION"
    THRUSTER_STUCK = "THRUSTER_STUCK"
    THRUSTER_LATENCY = "THRUSTER_LATENCY"
    THRUSTER_INTERMITTENT = "THRUSTER_INTERMITTENT"
    IMU_BIAS = "IMU_BIAS"
    IMU_DROPOUT = "IMU_DROPOUT"
    DEPTH_BIAS = "DEPTH_BIAS"
    CAMERA_FAILURE = "CAMERA_FAILURE"
    CAMERA_FOULING = "CAMERA_FOULING"
    SONAR_NOISE = "SONAR_NOISE"
    SONAR_FAILURE = "SONAR_FAILURE"
    COMM_LOSS = "COMM_LOSS"
    BATTERY_DEGRADATION = "BATTERY_DEGRADATION"
    LEAK_SIGNAL = "LEAK_SIGNAL"
    LOCALIZATION_DEGRADATION = "LOCALIZATION_DEGRADATION"


class PayloadEncoding(str, Enum):
    F64LE = "f64le"
    F32LE = "f32le"
    U8 = "u8"


_NUMPY_DTYPE = {PayloadEncoding.F64LE: "<f8", PayloadEncoding.F32LE: "<f4", PayloadEncoding.U8: "u1"}

IMU_LAYOUT = "imu_v1"  # [ax, ay, az, gx, gy, gz, qw, qx, qy, qz]; q = NaN x4 when not provided
DEPTH_LAYOUT = "depth_v1"  # [depth_m]
CAMERA_LAYOUT = "rgb8_hwc_v1"
SONAR_LAYOUT = "sonar_beam_bin_v1"  # [n_beams, n_bins] intensity proxy


class WireModel(ConradModel):
    """Base of every wire body."""
