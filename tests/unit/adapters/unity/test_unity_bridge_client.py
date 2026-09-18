"""UnityRobotHardware exercised over real ZeroMQ sockets against the mock Unity server (127.0.0.1)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid5

import numpy as np
import pytest
from conrad_mock_unity_server import MockBehaviour, MockUnityServer
from conrad_unity_testkit import make_command, sim_robot_config

from conrad.adapters.unity import (
    AdapterReason,
    BridgeState,
    FaultInjectionRequest,
    FaultType,
    RecordingTransport,
    ReplayDivergenceError,
    ReplayTransport,
    UnityBridgeConfig,
    UnityBridgeError,
    UnityBridgeTimeout,
    UnityPeerError,
    UnityProtocolError,
    UnityRobotHardware,
    ZmqBridgeTransport,
    validate_private_endpoint,
)
from conrad.persistence.object_store import ObjectStore
from conrad.robotics.hardware.interface import HardwareUnavailableError, RobotHardwareInterface
from conrad.runtime.command_gateway import CommandGateway
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality
from conrad.schemas.robot import HealthLevel
from conrad.settings import CommandMode, ExecutionLane, RuntimeSettings

CONFIG = sim_robot_config()
THRUSTERS = tuple(t.thruster_id for t in CONFIG.thrusters)
DT = 10_000_000


def _hw(server: MockUnityServer, ids: IdFactory, store: ObjectStore | None = None, **over: object) -> UnityRobotHardware:
    cfg = UnityBridgeConfig(control_endpoint=server.control_endpoint, request_timeout_ms=1500, **over)  # type: ignore[arg-type]
    return UnityRobotHardware(cfg, CONFIG, ids, mission_id=ids.new(), run_id=ids.new(), payload_store=store)


@pytest.fixture
def server() -> Iterator[MockUnityServer]:
    with MockUnityServer(CONFIG.content_digest(), THRUSTERS) as srv:
        yield srv


def test_is_a_robot_hardware_interface_and_handshakes(server: MockUnityServer, ids: IdFactory) -> None:
    hw = _hw(server, ids)
    assert isinstance(hw, RobotHardwareInterface) and not hw.is_physical
    with pytest.raises(HardwareUnavailableError):
        hw.get_health()  # not connected yet: fail closed
    ack = hw.connect()
    assert hw.state is BridgeState.CONNECTED
    assert ack.robot_config_digest == CONFIG.content_digest() == hw.robot_config_digest()
    assert hw.clock_domain() == "SIM"
    caps = hw.capabilities()
    assert caps.thruster_count == len(THRUSTERS) and caps.imu and caps.depth
    assert hw.get_imu() is None  # no sample before the first step: never fabricated
    hw.close()


def test_lock_step_sensors_frames_and_acquisition_timestamps(server: MockUnityServer, ids: IdFactory, store: ObjectStore) -> None:
    hw = _hw(server, ids, store)
    hw.connect()
    hw.reset(seed=11)
    for _ in range(5):
        state = hw.step(DT)
    assert hw.now_ns() == 5 * DT == state.sim_time_ns
    imu = hw.get_imu()
    assert imu is not None
    # acquisition time (delivery - 10 ms latency) is preserved; it is NOT replaced by the receive time
    assert imu.timestamp.time_ns == 5 * DT - 10_000_000
    assert imu.timestamp.clock_domain == "SIM"
    assert hw.delivery_latency_ns("imu") == 10_000_000
    # Unity (0, 9.81, 0) specific force -> Conrad +Z up; Unity yaw-rate -0.1 about +Y -> Conrad +0.1 about +Z
    assert imu.linear_acceleration_mps2 == pytest.approx((0.0, 0.0, 9.81))
    assert imu.angular_velocity_rps == pytest.approx((0.0, 0.0, 0.1))
    assert imu.orientation_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
    depth = hw.get_depth()
    assert depth is not None and depth.depth_m == pytest.approx(2.0) and depth.health is HealthLevel.OK
    cam = hw.get_camera()
    assert cam is not None and cam.modality is Modality.RGB
    assert cam.timestamp.time_ns == 5 * DT and cam.robot_pose_estimate is None
    assert cam.payload_ref is not None and cam.payload_ref.shape == (4, 6, 3)
    assert cam.payload_ref.digest == cam.sensor_context["wire_payload_digest"]
    assert store.get_bytes(cam.payload_ref) == ((np.arange(72) + 5) % 256).astype(np.uint8).tobytes()
    assert cam.sensor_id == uuid5(CONFIG.robot_id, "front_camera")
    sonar = hw.get_sonar()
    assert sonar is not None and sonar.sensor_frame == "SENSOR_SONAR"
    assert len(hw.get_thruster_state()) == len(THRUSTERS)
    power = hw.get_power_state()
    assert power is not None and power.remaining_fraction == pytest.approx(0.9)
    assert hw.get_metrics().metrics["commands_received"] == 0.0
    hw.close()


def test_sonar_is_inlined_without_a_store_but_never_silently_dropped(server: MockUnityServer, ids: IdFactory) -> None:
    hw = _hw(server, ids)
    hw.connect()
    for _ in range(5):
        hw.step(DT)
    sonar = hw.get_sonar()
    assert sonar is not None and sonar.inline_values is not None and len(sonar.inline_values) == 24
    assert sonar.inline_units == "intensity_proxy"
    hw.close()


def test_command_through_the_real_gateway_moves_the_vehicle(server: MockUnityServer, ids: IdFactory) -> None:
    mission, run = ids.new(), ids.new()
    cfg = UnityBridgeConfig(control_endpoint=server.control_endpoint)
    hw = UnityRobotHardware(cfg, CONFIG, ids, mission, run)
    hw.connect()
    hw.step(DT)
    gateway = CommandGateway(
        hw, CONFIG, RuntimeSettings(command_mode=CommandMode.SIMULATED), ExecutionLane.SIMULATION, mission, run
    )
    cmd = make_command(ids, CONFIG, mission, run, hw.now_ns(), {"V1": 0.5, "V2": 0.5})
    ack = gateway.submit(cmd)
    assert ack.accepted and ack.adapter == "unity_v2", ack.reason_codes
    assert server.received_commands[-1].thruster_commands["V1"] == 0.5
    assert server.received_commands[-1].safety_state == "NORMAL"
    for _ in range(50):
        hw.step(DT)
    depth = hw.get_depth()
    assert depth is not None and depth.depth_m > 2.05
    # duplicate command id is refused by the gateway before it reaches the wire
    assert not gateway.submit(cmd).accepted
    assert len(server.received_commands) == 1
    hw.close()


def test_adapter_rejects_bad_commands_without_touching_the_wire(server: MockUnityServer, ids: IdFactory) -> None:
    mission, run = ids.new(), ids.new()
    hw = _hw(server, ids)
    hw.connect()
    bad = [
        (make_command(ids, CONFIG, mission, run, 0, digest="a" * 64), AdapterReason.CONFIG_DIGEST),
        (make_command(ids, CONFIG, mission, run, 0, clock_domain="WALL"), AdapterReason.CLOCK_DOMAIN),
        (make_command(ids, CONFIG, mission, run, 0, authorized=False), AdapterReason.NO_AUTH),
        (make_command(ids, CONFIG, mission, run, 0, values={"V1": 1.5}), AdapterReason.ENVELOPE),
        (make_command(ids, CONFIG, mission, run, 0, values={"GHOST": 0.1}), AdapterReason.ACTUATORS),
    ]
    for command, reason in bad:
        ack = hw.send(command)
        assert not ack.accepted and reason in ack.reason_codes
    assert server.received_commands == []
    assert hw.state is BridgeState.CONNECTED
    hw.close()


def test_fault_injection_round_trip(server: MockUnityServer, ids: IdFactory) -> None:
    hw = _hw(server, ids)
    hw.connect()
    hw.step(DT)
    ack = hw.inject_fault(
        FaultInjectionRequest(fault_id="f-1", fault_type=FaultType.IMU_BIAS, target="imu", start_time_ns=3 * DT, magnitude=0.4)
    )
    assert ack.accepted and ack.scheduled_time_ns == 3 * DT
    hw.inject_fault(FaultInjectionRequest(fault_id="f-2", fault_type=FaultType.LEAK_SIGNAL, start_time_ns=4 * DT))
    hw.step(DT)
    imu = hw.get_imu()
    assert imu is not None and imu.linear_acceleration_mps2[1] == pytest.approx(0.0)
    hw.step(DT)  # sim time 3*DT: bias active. Unity +X (right) bias arrives as Conrad -Y (left is +Y)
    imu = hw.get_imu()
    assert imu is not None and imu.linear_acceleration_mps2[1] == pytest.approx(-0.4)
    assert hw.get_health().overall is HealthLevel.DEGRADED
    hw.step(DT)
    health = hw.get_health()
    assert health.overall is HealthLevel.FAULT and health.leak_detected is True
    assert "LEAK_SIGNAL" in health.faults
    hw.close()


@pytest.mark.parametrize(
    ("behaviour", "error"),
    [
        (MockBehaviour(simulator_id="someone-else"), UnityPeerError),
        (MockBehaviour(echo_nonce=False), UnityPeerError),
        (MockBehaviour(protocol_version="2.0.0"), UnityProtocolError),
        (MockBehaviour(schema_version="9.1.0"), UnityProtocolError),
        (MockBehaviour(clock_domain="UNITY_WALL"), UnityProtocolError),
        (MockBehaviour(digest_override="b" * 64), UnityProtocolError),
        (MockBehaviour(lock_step=False), UnityProtocolError),
    ],
    ids=["unknown-peer", "nonce", "protocol", "schema", "clock-domain", "config-digest", "step-mode"],
)
def test_handshake_fails_closed(behaviour: MockBehaviour, error: type[Exception], ids: IdFactory) -> None:
    with MockUnityServer(CONFIG.content_digest(), THRUSTERS, behaviour) as srv:
        hw = _hw(srv, ids)
        with pytest.raises(error):
            hw.connect()
        assert hw.state is BridgeState.FAULTED and hw.fault_reason
        mission, run = ids.new(), ids.new()
        with pytest.raises(HardwareUnavailableError):
            hw.send(make_command(ids, CONFIG, mission, run, 0))
        with pytest.raises(HardwareUnavailableError):
            hw.get_imu()
        assert srv.received_commands == []


def test_thruster_set_mismatch_fails_closed(ids: IdFactory) -> None:
    with MockUnityServer(CONFIG.content_digest(), ("A", "B")) as srv:
        with pytest.raises(UnityProtocolError, match="thruster set"):
            _hw(srv, ids).connect()


def test_timeout_fails_closed(ids: IdFactory) -> None:
    with MockUnityServer(CONFIG.content_digest(), THRUSTERS, MockBehaviour(silent=True)) as srv:
        cfg = UnityBridgeConfig(control_endpoint=srv.control_endpoint, request_timeout_ms=150)
        hw = UnityRobotHardware(cfg, CONFIG, ids, ids.new(), ids.new())
        with pytest.raises(UnityBridgeTimeout):
            hw.connect()
        assert hw.state is BridgeState.FAULTED
        with pytest.raises(UnityBridgeError):
            hw.step(DT)


def test_corrupt_payload_and_sensor_clock_domain_fail_closed(ids: IdFactory) -> None:
    for behaviour in (MockBehaviour(corrupt_payload=True), MockBehaviour(sensor_clock_domain="CAMERA_CLOCK")):
        with MockUnityServer(CONFIG.content_digest(), THRUSTERS, behaviour) as srv:
            hw = _hw(srv, ids)
            hw.connect()
            with pytest.raises(UnityProtocolError):
                hw.step(DT)
            assert hw.state is BridgeState.FAULTED
            with pytest.raises(HardwareUnavailableError):
                hw.get_depth()


def test_endpoints_must_be_private_and_allow_listed() -> None:
    assert validate_private_endpoint("tcp://127.0.0.1:5591", ("127.0.0.1",)) == "127.0.0.1"
    assert validate_private_endpoint("tcp://10.0.0.7:5591", ("10.0.0.7",)) == "10.0.0.7"
    for endpoint in ("tcp://0.0.0.0:5591", "tcp://8.8.8.8:5591", "tcp://localhost:5591", "ipc:///tmp/x", "tcp://127.0.0.1"):
        with pytest.raises(UnityPeerError):
            validate_private_endpoint(endpoint, ("127.0.0.1", "8.8.8.8", "0.0.0.0", "localhost"))
    with pytest.raises(UnityPeerError, match="allowed_peers"):
        validate_private_endpoint("tcp://192.168.1.4:1", ("127.0.0.1",))
    with pytest.raises(UnityPeerError):
        ZmqBridgeTransport("tcp://8.8.4.4:1", None, ("8.8.4.4",), 100)


def test_free_running_stream_is_consumed(ids: IdFactory) -> None:
    behaviour = MockBehaviour(lock_step=False, publish_stream=True)
    with MockUnityServer(CONFIG.content_digest(), THRUSTERS, behaviour) as srv:
        cfg = UnityBridgeConfig(
            control_endpoint=srv.control_endpoint, stream_endpoint=srv.stream_endpoint, lock_step=False
        )
        hw = UnityRobotHardware(cfg, CONFIG, ids, ids.new(), ids.new())
        hw.connect()
        with pytest.raises(UnityBridgeError):
            hw.step(DT)  # not valid when the simulator owns time
        assert hw.now_ns() == 0
        depth = hw.get_depth()  # POLL replies carry the newest packets as well
        assert depth is not None and depth.depth_m == pytest.approx(2.0)
        hw.close()


def test_recorded_run_replays_with_identical_acquisition_timestamps(server: MockUnityServer, tmp_path: Path) -> None:
    log = tmp_path / "wire.jsonl"

    def run(hw: UnityRobotHardware) -> list[tuple[int, int, float]]:
        hw.connect()
        hw.reset(seed=3)
        out = []
        for _ in range(6):
            hw.step(DT)
            imu, depth = hw.get_imu(), hw.get_depth()
            assert imu is not None and depth is not None
            out.append((imu.timestamp.time_ns, depth.timestamp.time_ns, depth.depth_m))
        hw.close()
        return out

    ids_live = IdFactory(seed=99)
    live_cfg = UnityBridgeConfig(control_endpoint=server.control_endpoint)
    transport = RecordingTransport(
        ZmqBridgeTransport(server.control_endpoint, None, live_cfg.allowed_peers, 1500), log
    )
    live = run(UnityRobotHardware(live_cfg, CONFIG, ids_live, ids_live.new(), ids_live.new(), transport=transport))

    ids_replay = IdFactory(seed=99)
    replayed = run(
        UnityRobotHardware(
            live_cfg, CONFIG, ids_replay, ids_replay.new(), ids_replay.new(), transport=ReplayTransport(log)
        )
    )
    assert replayed == live
    assert [t[0] for t in live] == [max(0, (i + 1) * DT - 10_000_000) for i in range(6)]

    # a replay that issues different requests is a divergence, not a silent success
    ids_other = IdFactory(seed=99)
    hw = UnityRobotHardware(live_cfg, CONFIG, ids_other, ids_other.new(), ids_other.new(), transport=ReplayTransport(log))
    hw.connect()
    with pytest.raises(ReplayDivergenceError):
        hw.reset(seed=4)
    assert hw.state is BridgeState.FAULTED
