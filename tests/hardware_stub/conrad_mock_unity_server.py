"""TEST-ONLY stand-in for the Unity V2 player. Speaks the real wire protocol over real ZeroMQ sockets.

It is not a simulator of record: the vehicle model is a one-line depth integrator so that a closed
loop has something to react to. Spatial values are hand-written in the UNITY convention (left-handed,
+Y up, +Z forward) without using the Python frame mapper, so adapter conversions are tested
against an independent encoding.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import zmq

from conrad.adapters.unity.frames import UNITY_FRAME_CONVENTION
from conrad.adapters.unity.protocol import (
    CAMERA_LAYOUT,
    DEPTH_LAYOUT,
    IMU_LAYOUT,
    PROTOCOL_VERSION,
    SONAR_LAYOUT,
    ClientRole,
    CommandPacket,
    ErrorReply,
    FaultAck,
    FaultInjectionRequest,
    FaultType,
    HandshakeAck,
    HandshakeRequest,
    MessageKind,
    MetricsReply,
    MetricsRequest,
    PayloadEncoding,
    PollRequest,
    ResetAck,
    ResetRequest,
    SensorPacket,
    SimulationValidityLevel,
    StatePacket,
    StepRequest,
    WireBattery,
    WireCapabilities,
    WireCommandAck,
    WireHealth,
    WireModel,
    WireThrusterState,
    decode_message,
    encode_message,
    make_sensor_packet,
)
from conrad.schemas.base import SCHEMA_VERSION


@dataclass
class MockBehaviour:
    simulator_id: str = "conrad-unity-v2"
    protocol_version: str = PROTOCOL_VERSION
    schema_version: str = SCHEMA_VERSION
    clock_domain: str = "SIM"
    sensor_clock_domain: str | None = None
    digest_override: str | None = None
    echo_nonce: bool = True
    silent: bool = False
    corrupt_payload: bool = False
    lock_step: bool = True
    vertical_thrusters: tuple[str, ...] = ("V1", "V2")
    imu_latency_ns: int = 10_000_000
    yaw_rate_conrad_rps: float = 0.1
    publish_stream: bool = False


@dataclass
class MockUnityServer:
    robot_config_digest: str
    thruster_ids: tuple[str, ...]
    behaviour: MockBehaviour = field(default_factory=MockBehaviour)

    def __post_init__(self) -> None:
        self._ctx = zmq.Context()
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.setsockopt(zmq.LINGER, 0)
        self._rep.setsockopt(zmq.RCVTIMEO, 1000)
        self._rep.setsockopt(zmq.SNDTIMEO, 1000)
        port = self._rep.bind_to_random_port("tcp://127.0.0.1")
        self.control_endpoint = f"tcp://127.0.0.1:{port}"
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        pub_port = self._pub.bind_to_random_port("tcp://127.0.0.1")
        self.stream_endpoint = f"tcp://127.0.0.1:{pub_port}"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="mock-unity", daemon=True)
        self.session_id = "mock-session-1"
        self.sim_time_ns = 0
        self.depth_m = 2.0
        self.vertical_speed = 0.0
        self.commands: dict[str, float] = dict.fromkeys(self.thruster_ids, 0.0)
        self.received_commands: list[CommandPacket] = []
        self.faults: list[FaultInjectionRequest] = []
        self.seq_counter = 0
        self.seed: int | None = None
        self.requests_seen: list[MessageKind] = []
        self.errors: list[str] = []

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> MockUnityServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._ctx.destroy(linger=0)

    def __enter__(self) -> MockUnityServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ serving
    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._rep, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(timeout=20)):
                continue
            try:
                raw = self._rep.recv_multipart()[0]
            except zmq.ZMQError:
                return
            if self.behaviour.silent:
                # Never answer: the client must time out and fail closed. (Socket stays in recv state.)
                self._stop.wait(timeout=5.0)
                return
            env = None
            try:
                env, body = decode_message(raw)
                self.requests_seen.append(env.kind)
                kind, reply = self._handle(env.kind, body)
            except Exception as exc:  # the mock reports, the client must fail closed
                self.errors.append(repr(exc))
                kind, reply = MessageKind.ERROR, ErrorReply(code="BAD_REQUEST", detail=str(exc))
            seq = env.seq if env is not None else 0
            try:
                self._rep.send_multipart(
                    [
                        encode_message(
                            kind,
                            reply,
                            session_id=self.session_id,
                            seq=seq,
                            protocol_version=self.behaviour.protocol_version,
                            schema_version=self.behaviour.schema_version,
                        )
                    ]
                )
            except zmq.ZMQError:
                return

    def _active(self, fault_type: FaultType, target: str | None = None) -> FaultInjectionRequest | None:
        for f in self.faults:
            if f.fault_type is not fault_type or (target is not None and f.target != target):
                continue
            end = None if f.duration_ns is None else f.start_time_ns + f.duration_ns
            if self.sim_time_ns >= f.start_time_ns and (end is None or self.sim_time_ns < end):
                return f
        return None

    def _handle(self, kind: MessageKind, body: WireModel) -> tuple[MessageKind, WireModel]:
        b = self.behaviour
        if isinstance(body, HandshakeRequest):
            if body.role is not ClientRole.CONTROL:
                return MessageKind.ERROR, ErrorReply(code="ROLE_FORBIDDEN", detail="truth is not served here")
            return MessageKind.HANDSHAKE_ACK, HandshakeAck(
                simulator_id=b.simulator_id,
                simulator_version="mock-0",
                protocol_version=b.protocol_version,
                schema_version=b.schema_version,
                robot_config_digest=b.digest_override or self.robot_config_digest,
                clock_domain=b.clock_domain,
                frame_convention=UNITY_FRAME_CONVENTION,
                lock_step=b.lock_step,
                session_id=self.session_id,
                nonce=body.nonce if b.echo_nonce else "0" * 32,
                validity_level=SimulationValidityLevel.L0_FUNCTIONAL,
                physics_dt_ns=5_000_000,
                sim_time_ns=self.sim_time_ns,
                capabilities=WireCapabilities(
                    capability_version="mock-1",
                    cameras=("front_camera",),
                    sonars=("sonar",),
                    imu=True,
                    depth=True,
                    thruster_ids=self.thruster_ids,
                    controllable_dof=("heave",),
                ),
            )
        if isinstance(body, ResetRequest):
            self.seed = body.seed
            self.sim_time_ns = 0
            self.depth_m = 2.0
            self.vertical_speed = 0.0
            self.faults.clear()
            self.seq_counter = 0
            return MessageKind.RESET_ACK, ResetAck(
                seed=body.seed, sim_time_ns=0, scenario_digest=body.scenario_digest
            )
        if isinstance(body, StepRequest):
            self._advance(body.dt_ns)
            return MessageKind.STATE, self._state(body.step_index)
        if isinstance(body, PollRequest):
            return MessageKind.STATE, self._state(None)
        if isinstance(body, CommandPacket):
            self.received_commands.append(body)
            reasons: list[str] = []
            if body.deadline_ns <= self.sim_time_ns:
                reasons.append("EXPIRED_DEADLINE")
            if not reasons:
                self.commands = dict(body.thruster_commands)
            return MessageKind.COMMAND_ACK, WireCommandAck(
                command_id=body.command_id,
                accepted=not reasons,
                reason_codes=tuple(reasons),
                ack_time_ns=self.sim_time_ns,
                sim_time_ns=self.sim_time_ns,
            )
        if isinstance(body, FaultInjectionRequest):
            self.faults.append(body)
            return MessageKind.FAULT_ACK, FaultAck(
                fault_id=body.fault_id,
                accepted=True,
                scheduled_time_ns=max(body.start_time_ns, self.sim_time_ns),
            )
        if isinstance(body, MetricsRequest):
            return MessageKind.METRICS, MetricsReply(
                sim_time_ns=self.sim_time_ns,
                validity_level=SimulationValidityLevel.L0_FUNCTIONAL,
                metrics={"commands_received": float(len(self.received_commands))},
            )
        return MessageKind.ERROR, ErrorReply(code="UNSUPPORTED", detail=kind.value)

    # ------------------------------------------------------------------ toy vehicle
    def _advance(self, dt_ns: int) -> None:
        dt = dt_ns / 1e9
        heave = 0.0
        for tid in self.behaviour.vertical_thrusters:
            u = self.commands.get(tid, 0.0)
            if self._active(FaultType.THRUSTER_FAILURE, tid) is not None:
                u = 0.0
            heave += u
        # positive command pushes the vehicle DOWN (depth increases) in this toy model
        accel = 2.0 * heave - 1.5 * self.vertical_speed
        self.vertical_speed += accel * dt
        self.depth_m += self.vertical_speed * dt
        self.sim_time_ns += dt_ns

    def _sensor_packets(self) -> tuple[SensorPacket, ...]:
        b = self.behaviour
        domain = b.sensor_clock_domain or b.clock_domain
        self.seq_counter += 1
        t = self.sim_time_ns
        acq_imu = max(0, t - b.imu_latency_ns)
        packets: list[SensorPacket] = []
        if self._active(FaultType.IMU_DROPOUT) is None:
            bias = self._active(FaultType.IMU_BIAS)
            bias_mag = 0.0 if bias is None else bias.magnitude
            # UNITY convention: specific force at rest is +Y (up); Conrad yaw-left is NEGATIVE about Unity +Y.
            imu = np.array(
                [bias_mag, 9.81, 0.0, 0.0, -b.yaw_rate_conrad_rps, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
            packets.append(
                make_sensor_packet(
                    sensor_name="imu",
                    modality="IMU",
                    frame_id="SENSOR_IMU",
                    clock_domain=domain,
                    acquisition_time_ns=acq_imu,
                    delivery_time_ns=t,
                    sequence_index=self.seq_counter,
                    layout=IMU_LAYOUT,
                    array=imu,
                    encoding=PayloadEncoding.F64LE,
                    units="m/s^2,rad/s,quat_wxyz",
                )
            )
        depth_bias = self._active(FaultType.DEPTH_BIAS)
        packets.append(
            make_sensor_packet(
                sensor_name="depth",
                modality="PRESSURE_DEPTH",
                frame_id="SENSOR_DEPTH",
                clock_domain=domain,
                acquisition_time_ns=t,
                delivery_time_ns=t,
                sequence_index=self.seq_counter,
                layout=DEPTH_LAYOUT,
                array=np.array([self.depth_m + (0.0 if depth_bias is None else depth_bias.magnitude)]),
                encoding=PayloadEncoding.F64LE,
                units="m",
            )
        )
        if self.seq_counter % 5 == 0 and self._active(FaultType.CAMERA_FAILURE) is None:
            image = (np.arange(4 * 6 * 3, dtype=np.int64) + self.seq_counter) % 256
            packets.append(
                make_sensor_packet(
                    sensor_name="front_camera",
                    modality="RGB",
                    frame_id="SENSOR_CAMERA_FRONT",
                    clock_domain=domain,
                    acquisition_time_ns=t,
                    delivery_time_ns=t,
                    sequence_index=self.seq_counter,
                    layout=CAMERA_LAYOUT,
                    array=image.reshape(4, 6, 3),
                    encoding=PayloadEncoding.U8,
                    units="dn",
                    context={"fx": 3.0, "fy": 3.0, "cx": 3.0, "cy": 2.0},
                )
            )
            packets.append(
                make_sensor_packet(
                    sensor_name="sonar",
                    modality="SONAR",
                    frame_id="SENSOR_SONAR",
                    clock_domain=domain,
                    acquisition_time_ns=t,
                    delivery_time_ns=t,
                    sequence_index=self.seq_counter,
                    layout=SONAR_LAYOUT,
                    array=np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(3, 8),
                    encoding=PayloadEncoding.F32LE,
                    units="intensity_proxy",
                    context={"range_max_m": 20.0, "horizontal_fov_rad": 0.5},
                )
            )
        if b.corrupt_payload and packets:
            first = packets[0]
            packets[0] = first.model_copy(update={"payload_digest": "0" * 64})
        if b.publish_stream:
            for p in packets:
                self._pub.send_multipart(
                    [encode_message(MessageKind.SENSOR, p, session_id=self.session_id, seq=self.seq_counter)]
                )
        return tuple(packets)

    def _state(self, step_index: int | None) -> StatePacket:
        leak = self._active(FaultType.LEAK_SIGNAL) is not None
        faults = tuple(
            sorted({f.fault_type.value for f in self.faults if self._active(f.fault_type, f.target)})
        )
        return StatePacket(
            step_index=step_index,
            sim_time_ns=self.sim_time_ns,
            clock_domain=self.behaviour.clock_domain,
            sensors=self._sensor_packets(),
            thrusters=tuple(
                WireThrusterState(
                    thruster_id=tid,
                    command=self.commands.get(tid, 0.0),
                    estimated_thrust_n=0.0
                    if self._active(FaultType.THRUSTER_FAILURE, tid)
                    else 40.0 * self.commands.get(tid, 0.0) * abs(self.commands.get(tid, 0.0)),
                    health="FAULT" if self._active(FaultType.THRUSTER_FAILURE, tid) else "OK",
                )
                for tid in self.thruster_ids
            ),
            battery=WireBattery(
                time_ns=self.sim_time_ns, remaining_fraction=0.9, voltage_v=15.8, energy_used_j=10.0
            ),
            health=WireHealth(
                time_ns=self.sim_time_ns,
                overall="FAULT" if leak else ("DEGRADED" if faults else "OK"),
                leak_detected=leak,
                faults=faults,
            ),
        )
