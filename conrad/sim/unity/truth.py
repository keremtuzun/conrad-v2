"""TRUTH PLANE ONLY: ground-truth client for the Unity truth endpoint (training/evaluation).

The control endpoint (``conrad.adapters.unity``) never registers these bodies, so truth can not
arrive through ``UnityRobotHardware``. This module lives under ``conrad.sim`` which the belief and
decision planes are forbidden to import (tests/leakage).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from conrad.adapters.unity.frames import UNITY_FRAME_CONVENTION, UnityFrameMapper
from conrad.adapters.unity.protocol import (
    CONTROL_BODIES,
    BodyRegistry,
    ClientRole,
    ErrorReply,
    HandshakeAck,
    HandshakeRequest,
    MessageKind,
    UnityProtocolError,
    WireModel,
    decode_message,
    encode_message,
)
from conrad.adapters.unity.transport import BridgeTransport
from conrad.schemas.frames import WORLD, Pose, Quat, Vec3

TRUTH_MARKER = "__conrad_truth__"


class GroundTruthRequest(WireModel):
    pass


class GroundTruthReply(WireModel):
    """Unity-convention truth of the vehicle; converted by :class:`UnityTruthClient`."""

    sim_time_ns: int = Field(ge=0)
    position_m: Vec3
    orientation_wxyz: Quat
    linear_velocity_mps: Vec3
    angular_velocity_rps: Vec3
    water_current_mps: Vec3 = (0.0, 0.0, 0.0)
    active_faults: tuple[str, ...] = ()


class SurfaceVisibilityRequest(WireModel):
    origin_m: Vec3
    points_m: tuple[Vec3, ...] = Field(min_length=1, max_length=4096)
    tolerance_m: float = Field(ge=0, le=0.5)


class SurfaceVisibilityReply(WireModel):
    visible: tuple[bool, ...]
    first_hit_distance_m: tuple[float, ...]


class TruthVehicleState(WireModel):
    """Conrad-convention truth. Never an input to estimation, Model 2 or Model 1."""

    sim_time_ns: int
    pose: Pose
    linear_velocity_world_mps: Vec3
    angular_velocity_world_rps: Vec3
    water_current_world_mps: Vec3
    active_faults: tuple[str, ...]


TRUTH_BODIES: BodyRegistry = {
    **{k: v for k, v in CONTROL_BODIES.items() if k in (MessageKind.HANDSHAKE_ACK, MessageKind.ERROR)},
    MessageKind.GROUND_TRUTH: GroundTruthReply,
    MessageKind.SURFACE_VISIBILITY_REPLY: SurfaceVisibilityReply,
}


class UnityTruthClient:
    def __init__(self, transport: BridgeTransport, robot_config_digest: str, clock_domain: str) -> None:
        self._t = transport
        self._digest = robot_config_digest
        self._clock = clock_domain
        self._mapper = UnityFrameMapper()
        self._seq = 0
        self._session = ""

    def _ask(self, kind: MessageKind, body: WireModel) -> WireModel:
        self._seq += 1
        env, reply = decode_message(
            self._t.request(encode_message(kind, body, session_id=self._session, seq=self._seq)), TRUTH_BODIES
        )
        if isinstance(reply, ErrorReply):
            raise UnityProtocolError(f"truth endpoint error {reply.code}: {reply.detail}")
        if env.seq != self._seq:
            raise UnityProtocolError("truth reply answers a different request")
        return reply

    def connect(self, nonce: str) -> HandshakeAck:
        req = HandshakeRequest(
            client_name="conrad-truth",
            role=ClientRole.TRUTH,
            robot_config_digest=self._digest,
            clock_domain=self._clock,
            frame_convention=UNITY_FRAME_CONVENTION,
            nonce=nonce,
        )
        ack = self._ask(MessageKind.HANDSHAKE, req)
        if not isinstance(ack, HandshakeAck) or ack.nonce != nonce:
            raise UnityProtocolError("truth handshake failed")
        if ack.robot_config_digest != self._digest or ack.clock_domain != self._clock:
            raise UnityProtocolError("truth endpoint serves a different RobotConfig or clock domain")
        self._session = ack.session_id
        return ack

    def get_ground_truth(self) -> TruthVehicleState:
        r = self._ask(MessageKind.GET_GROUND_TRUTH, GroundTruthRequest())
        if not isinstance(r, GroundTruthReply):
            raise UnityProtocolError("expected GROUND_TRUTH")
        m = self._mapper
        return TruthVehicleState(
            sim_time_ns=r.sim_time_ns,
            pose=Pose(
                frame_id=WORLD,
                position_m=m.point_to_conrad(r.position_m),
                orientation_wxyz=m.quat_to_conrad(r.orientation_wxyz),
            ),
            linear_velocity_world_mps=m.vector_to_conrad(r.linear_velocity_mps),
            angular_velocity_world_rps=m.axial_to_conrad(r.angular_velocity_rps),
            water_current_world_mps=m.vector_to_conrad(r.water_current_mps),
            active_faults=r.active_faults,
        )

    def surface_visibility(
        self, origin_m: Sequence[float], points_m: Sequence[Sequence[float]], tolerance_m: float
    ) -> tuple[bool, ...]:
        """Unity collider LOS for a truth-side simulated payload, in Conrad WORLD."""
        if len(origin_m) != 3 or not 1 <= len(points_m) <= 4096 or any(len(p) != 3 for p in points_m):
            raise ValueError("invalid structural visibility point batch")
        req = SurfaceVisibilityRequest(
            origin_m=(float(origin_m[0]), float(origin_m[1]), float(origin_m[2])),
            points_m=tuple((float(p[0]), float(p[1]), float(p[2])) for p in points_m),
            tolerance_m=tolerance_m,
        )
        reply = self._ask(MessageKind.SURFACE_VISIBILITY, req)
        if not isinstance(reply, SurfaceVisibilityReply) or len(reply.visible) != len(points_m):
            raise UnityProtocolError("structural visibility reply length mismatch")
        return reply.visible
