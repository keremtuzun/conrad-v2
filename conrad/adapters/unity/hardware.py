"""``UnityRobotHardware``: the RobotHardwareInterface client for the Unity V2 simulator.

Everything above the interface is identical for Unity, the Python kernel and the physical robot
(ch20 Physical software swap). The adapter fails closed: unknown peer, protocol/schema mismatch,
clock-domain mismatch, RobotConfig digest mismatch, corrupt payloads and timeouts all leave the
adapter FAULTED, after which every call raises :class:`UnityBridgeError`.

No truth crosses this adapter. Ground truth has its own endpoint and client in ``conrad.sim.unity``.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TypeVar
from uuid import UUID

from conrad.adapters.unity import conversion as cv
from conrad.adapters.unity.conversion import ADAPTER_NAME, PayloadStore, UnityBridgeConfig
from conrad.adapters.unity.frames import UNITY_FRAME_CONVENTION, UnityFrameMapper
from conrad.adapters.unity.protocol import (
    ClientRole,
    CommandPacket,
    ErrorReply,
    FaultAck,
    FaultInjectionRequest,
    HandshakeAck,
    HandshakeRequest,
    MessageKind,
    MetricsReply,
    MetricsRequest,
    PollRequest,
    ResetAck,
    ResetRequest,
    SensorPacket,
    SimulationValidityLevel,
    StatePacket,
    StepRequest,
    UnityProtocolError,
    WireCommandAck,
    WireModel,
    decode_message,
    encode_message,
)
from conrad.adapters.unity.transport import BridgeTransport, UnityBridgeError, ZmqBridgeTransport
from conrad.robotics.hardware.interface import RobotHardwareInterface
from conrad.schemas.frames import FrameConvention
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.robot import (
    AllocatedCommand,
    BatteryState,
    CommandAck,
    DepthSample,
    ImuSample,
    RobotCapabilities,
    RobotConfig,
    SystemHealth,
    ThrusterState,
)

W = TypeVar("W", bound=WireModel)
R = TypeVar("R")


class BridgeState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTED = "CONNECTED"
    FAULTED = "FAULTED"
    CLOSED = "CLOSED"


class UnityRobotHardware(RobotHardwareInterface):
    adapter_name = ADAPTER_NAME
    is_physical = False

    def __init__(
        self,
        config: UnityBridgeConfig,
        robot_config: RobotConfig,
        ids: IdFactory,
        mission_id: UUID,
        run_id: UUID,
        payload_store: PayloadStore | None = None,
        transport: BridgeTransport | None = None,
        frame_convention: FrameConvention | None = None,
    ) -> None:
        self._cfg = config
        self._robot = robot_config
        self._digest = robot_config.content_digest()
        self._thruster_ids = {t.thruster_id for t in robot_config.thrusters}
        self._ids, self._mission_id, self._run_id = ids, mission_id, run_id
        self._store = payload_store
        self._mapper = UnityFrameMapper(frame_convention)
        self._transport = transport
        self._state = BridgeState.DISCONNECTED
        self._fault_reason: str | None = None
        self._session_id = ""
        self._seq = 0
        self._step_index = 0
        self._ack: HandshakeAck | None = None
        self._sim_time_ns = 0
        self._latest: dict[str, SensorPacket] = {}
        self._last_state: StatePacket | None = None
        self.out_of_order_packets = 0

    # ---------------------------------------------------------------- lifecycle
    @property
    def state(self) -> BridgeState:
        return self._state

    @property
    def fault_reason(self) -> str | None:
        return self._fault_reason

    @property
    def validity_level(self) -> SimulationValidityLevel:
        return self._require_ack().validity_level

    @property
    def physics_dt_ns(self) -> int:
        return self._require_ack().physics_dt_ns

    def __enter__(self) -> UnityRobotHardware:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def connect(self) -> HandshakeAck:
        if self._state is not BridgeState.DISCONNECTED:
            raise UnityBridgeError(f"connect() in state {self._state.value}")
        try:
            if self._transport is None:
                c = self._cfg
                self._transport = ZmqBridgeTransport(
                    c.control_endpoint, c.stream_endpoint, c.allowed_peers, c.request_timeout_ms
                )
            nonce = self._ids.new().hex
            request = HandshakeRequest(
                client_name=self._cfg.client_name,
                role=ClientRole.CONTROL,
                robot_config_digest=self._digest,
                clock_domain=self._cfg.clock_domain,
                frame_convention=UNITY_FRAME_CONVENTION,
                lock_step=self._cfg.lock_step,
                nonce=nonce,
            )
            ack = self._exchange(MessageKind.HANDSHAKE, request, MessageKind.HANDSHAKE_ACK, HandshakeAck)
            cv.verify_handshake(ack, self._cfg, self._digest, nonce, self._thruster_ids)
        except Exception as exc:
            self._fail(f"handshake failed: {exc}")
            raise
        self._ack, self._session_id, self._sim_time_ns = ack, ack.session_id, ack.sim_time_ns
        self._state = BridgeState.CONNECTED
        return ack

    def close(self) -> None:
        if self._transport is not None and self._state in (BridgeState.CONNECTED, BridgeState.DISCONNECTED):
            self._transport.close()
        if self._state is not BridgeState.FAULTED:
            self._state = BridgeState.CLOSED

    def _fail(self, reason: str) -> None:
        was_open = self._state in (BridgeState.CONNECTED, BridgeState.DISCONNECTED)
        self._state, self._fault_reason = BridgeState.FAULTED, reason
        if was_open and self._transport is not None:
            self._transport.close()

    def _require_ack(self) -> HandshakeAck:
        if self._state is not BridgeState.CONNECTED or self._ack is None:
            raise UnityBridgeError(f"Unity bridge unavailable ({self._state.value}: {self._fault_reason})")
        return self._ack

    def _guard(self, what: str, fn: Callable[[], R]) -> R:
        """Run a conversion; any protocol error faults the adapter before propagating."""
        try:
            return fn()
        except UnityProtocolError as exc:
            self._fail(f"{what}: {exc}")
            raise

    # ---------------------------------------------------------------- messaging
    def _exchange(self, kind: MessageKind, body: WireModel, reply_kind: MessageKind, rtype: type[W]) -> W:
        assert self._transport is not None
        self._seq += 1
        raw = self._transport.request(encode_message(kind, body, session_id=self._session_id, seq=self._seq))
        env, reply = decode_message(raw)
        if isinstance(reply, ErrorReply):
            raise UnityProtocolError(f"simulator error {reply.code}: {reply.detail}")
        if env.kind is not reply_kind or not isinstance(reply, rtype):
            raise UnityProtocolError(f"expected {reply_kind.value}, got {env.kind.value}")
        if env.seq != self._seq:
            raise UnityProtocolError(f"reply seq {env.seq} does not answer request seq {self._seq}")
        if self._session_id and env.session_id != self._session_id:
            raise UnityProtocolError("reply carries a foreign session_id")
        return reply

    def _call(self, kind: MessageKind, body: WireModel, reply_kind: MessageKind, rtype: type[W]) -> W:
        self._require_ack()
        try:
            return self._exchange(kind, body, reply_kind, rtype)
        except Exception as exc:
            self._fail(f"{kind.value} failed: {exc}")
            raise

    # ---------------------------------------------------------------- simulator-only surface
    def reset(
        self,
        seed: int,
        scenario_ref: str | None = None,
        scenario_digest: str | None = None,
        experiment: dict[str, object] | None = None,
    ) -> ResetAck:
        req = ResetRequest(
            seed=seed, scenario_ref=scenario_ref, scenario_digest=scenario_digest, experiment=experiment or {}
        )
        ack = self._call(MessageKind.RESET, req, MessageKind.RESET_ACK, ResetAck)
        if ack.seed != seed or (scenario_digest is not None and ack.scenario_digest != scenario_digest):
            self._fail("reset acknowledged a different seed/scenario")
            raise UnityProtocolError("reset acknowledged a different seed/scenario")
        self._latest.clear()
        self._last_state, self._step_index, self._sim_time_ns = None, 0, ack.sim_time_ns
        return ack

    def step(self, dt_ns: int) -> StatePacket:
        """Lock-step advance. Not part of the RobotHardwareInterface: the experiment runner owns time."""
        if not self._cfg.lock_step:
            raise UnityBridgeError("step() is only valid in lock-step mode")
        state = self._call(
            MessageKind.STEP,
            StepRequest(step_index=self._step_index, dt_ns=dt_ns),
            MessageKind.STATE,
            StatePacket,
        )
        if state.step_index != self._step_index:
            self._fail("step index mismatch")
            raise UnityProtocolError("simulator answered a different step index")
        self._step_index += 1
        self._ingest(state)
        return state

    def poll(self) -> StatePacket:
        state = self._call(MessageKind.POLL, PollRequest(), MessageKind.STATE, StatePacket)
        self._ingest(state)
        return state

    def drain_stream(self) -> int:
        """Consume PUB sensor packets (free-running mode). Returns the number ingested."""
        self._require_ack()
        assert self._transport is not None

        def run() -> int:
            count = 0
            for raw in self._transport.poll_stream() if self._transport else []:
                env, body = decode_message(raw)
                if env.kind is not MessageKind.SENSOR or not isinstance(body, SensorPacket):
                    raise UnityProtocolError(f"unexpected {env.kind.value} on the sensor stream")
                if env.session_id != self._session_id:
                    raise UnityProtocolError("stream packet from a foreign session")
                self._ingest_sensor(body)
                count += 1
            return count

        return self._guard("sensor stream", run)

    def inject_fault(self, request: FaultInjectionRequest) -> FaultAck:
        ack = self._call(MessageKind.INJECT_FAULT, request, MessageKind.FAULT_ACK, FaultAck)
        if ack.fault_id != request.fault_id:
            self._fail("fault ack for a different fault")
            raise UnityProtocolError("fault ack for a different fault")
        return ack

    def get_metrics(self) -> MetricsReply:
        return self._call(MessageKind.GET_METRICS, MetricsRequest(), MessageKind.METRICS, MetricsReply)

    def _ingest(self, state: StatePacket) -> None:
        def run() -> None:
            if state.clock_domain != self._cfg.clock_domain:
                raise UnityProtocolError(f"state packet in clock domain {state.clock_domain!r}")
            if state.sim_time_ns < self._sim_time_ns:
                raise UnityProtocolError("simulation time moved backwards")
            for packet in state.sensors:
                self._ingest_sensor(packet)

        self._guard("state", run)
        self._sim_time_ns, self._last_state = state.sim_time_ns, state

    def _ingest_sensor(self, packet: SensorPacket) -> None:
        if packet.clock_domain != self._cfg.clock_domain:
            raise UnityProtocolError(f"sensor {packet.sensor_name} stamped in {packet.clock_domain!r}")
        packet.payload_bytes()  # verifies digest + size before the packet can ever be served
        previous = self._latest.get(packet.sensor_name)
        if previous is not None and packet.acquisition_time_ns < previous.acquisition_time_ns:
            self.out_of_order_packets += 1
            return
        self._latest[packet.sensor_name] = packet

    def _newest(self, modality: Modality, names: tuple[str, ...] | None = None) -> SensorPacket | None:
        self._require_ack()
        if not self._cfg.lock_step:
            self.drain_stream()
        found = [
            p
            for p in self._latest.values()
            if p.modality == modality.value and (names is None or p.sensor_name in names)
        ]
        return max(found, key=lambda p: p.acquisition_time_ns) if found else None

    def delivery_latency_ns(self, sensor_name: str) -> int | None:
        p = self._latest.get(sensor_name)
        return None if p is None else p.delivery_time_ns - p.acquisition_time_ns

    def _state_or_poll(self) -> StatePacket:
        self._require_ack()
        if self._last_state is None or not self._cfg.lock_step:
            return self.poll()
        return self._last_state

    # ---------------------------------------------------------------- RobotHardwareInterface
    def capabilities(self) -> RobotCapabilities:
        return cv.capabilities_from_wire(self._require_ack().capabilities)

    def robot_config_digest(self) -> str:
        return self._require_ack().robot_config_digest

    def clock_domain(self) -> str:
        return self._require_ack().clock_domain

    def now_ns(self) -> int:
        self._require_ack()
        if not self._cfg.lock_step:
            self.poll()
        return self._sim_time_ns

    def get_imu(self) -> ImuSample | None:
        packet = self._newest(Modality.IMU) if self._require_ack().capabilities.imu else None
        return (
            None if packet is None else self._guard("imu", lambda: cv.imu_from_packet(packet, self._mapper))
        )

    def get_depth(self) -> DepthSample | None:
        packet = self._newest(Modality.PRESSURE_DEPTH) if self._require_ack().capabilities.depth else None
        return None if packet is None else self._guard("depth", lambda: cv.depth_from_packet(packet))

    def _image(self, modality: Modality, names: tuple[str, ...]) -> Observation | None:
        ack = self._require_ack()
        packet = self._newest(modality, names) if names else None
        if packet is None:
            return None
        return self._guard(
            modality.value,
            lambda: cv.observation_from_packet(
                packet,
                modality,
                robot_id=self._robot.robot_id,
                mission_id=self._mission_id,
                run_id=self._run_id,
                observation_id=self._ids.new(),
                trace_id=self._ids.new(),
                validity_level=ack.validity_level,
                store=self._store,
            ),
        )

    def get_camera(self) -> Observation | None:
        return self._image(Modality.RGB, self._require_ack().capabilities.cameras)

    def get_sonar(self) -> Observation | None:
        return self._image(Modality.SONAR, self._require_ack().capabilities.sonars)

    def get_thruster_state(self) -> tuple[ThrusterState, ...]:
        state = self._state_or_poll()
        return self._guard("thrusters", lambda: cv.thrusters_from_state(state))

    def get_power_state(self) -> BatteryState | None:
        return cv.battery_from_state(self._state_or_poll())

    def get_health(self) -> SystemHealth:
        state = self._state_or_poll()
        return self._guard("health", lambda: cv.health_from_state(state))

    def send(self, command: AllocatedCommand) -> CommandAck:
        ack = self._require_ack()
        reasons = cv.precheck_command(
            command,
            self._digest,
            ack.clock_domain,
            set(ack.capabilities.thruster_ids),
            self._cfg.require_safety_authorization,
        )
        if reasons:
            return CommandAck(
                command_id=command.command_id,
                accepted=False,
                reason_codes=tuple(reasons),
                ack_time_ns=self._sim_time_ns,
                adapter=ADAPTER_NAME,
            )
        packet = CommandPacket.from_allocated(command)
        wire = self._call(MessageKind.COMMAND, packet, MessageKind.COMMAND_ACK, WireCommandAck)
        if wire.command_id != str(command.command_id):
            self._fail("command ack for a different command")
            raise UnityProtocolError("command ack for a different command")
        self._sim_time_ns = max(self._sim_time_ns, wire.sim_time_ns)
        return CommandAck(
            command_id=command.command_id,
            accepted=wire.accepted,
            reason_codes=wire.reason_codes,
            ack_time_ns=wire.ack_time_ns,
            adapter=ADAPTER_NAME,
        )
