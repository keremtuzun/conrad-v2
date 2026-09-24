"""conrad.sim.unity: RobotConfig export, experiment determinism, validity levels, settings, truth client."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import zmq
from conrad_unity_testkit import sim_robot_config

from conrad.adapters.unity import FaultType, SimulationValidityLevel, ZmqBridgeTransport
from conrad.adapters.unity.frames import UNITY_FRAME_CONVENTION
from conrad.adapters.unity.protocol import (
    PROTOCOL_VERSION,
    HandshakeAck,
    HandshakeRequest,
    MessageKind,
    UnityProtocolError,
    WireCapabilities,
    decode_message,
    encode_message,
)
from conrad.robotics.hardware.config import load_robot_config
from conrad.schemas.base import SCHEMA_VERSION
from conrad.schemas.robot import OpenParameterError, Sourced, SourceKind
from conrad.sim.unity import (
    FaultSpec,
    assess_validity_level,
    draw_fault_schedule,
    load_unity_settings,
    robot_config_to_unity,
    write_unity_robot_config,
)
from conrad.sim.unity.truth import (
    GroundTruthReply,
    GroundTruthRequest,
    SurfaceVisibilityReply,
    SurfaceVisibilityRequest,
    UnityTruthClient,
)

CONFIG = sim_robot_config()


def test_export_embeds_digest_and_sources(tmp_path: Path) -> None:
    digest = write_unity_robot_config(CONFIG, tmp_path / "rc.json")
    doc = json.loads((tmp_path / "rc.json").read_text(encoding="utf-8"))
    assert digest == CONFIG.content_digest() == doc["robot_config_digest"]
    assert doc["mass_kg"] == {"value": 11.5, "units": "kg", "source": "SYNTHETIC_ONLY", "sigma": None}
    assert len(doc["thrusters"]) == 8 and doc["wire_convention"] == UNITY_FRAME_CONVENTION
    assert doc["robot_id"] == str(CONFIG.robot_id)


def test_export_refuses_open_parameters() -> None:
    with pytest.raises(OpenParameterError, match="OPEN"):
        robot_config_to_unity(load_robot_config("configs/robot/physical_template.yaml"))
    opened = CONFIG.model_copy(
        update={"mass_kg": Sourced[float](value=None, units="kg", source=SourceKind.OPEN)}
    )
    with pytest.raises(OpenParameterError, match="mass_kg"):
        robot_config_to_unity(opened)


def test_fault_schedule_is_deterministic_per_seed() -> None:
    specs = (
        FaultSpec(fault_type=FaultType.THRUSTER_FAILURE, target="H1", probability=0.7, start_window_s=(1, 5)),
        FaultSpec(
            fault_type=FaultType.IMU_BIAS,
            probability=0.7,
            start_window_s=(0, 2),
            duration_window_s=(1, 2),
            magnitude_window=(0.01, 0.1),
        ),
    )
    a, da = draw_fault_schedule(specs, 5)
    b, db = draw_fault_schedule(specs, 5)
    assert a == b and da == db
    digests = {draw_fault_schedule(specs, s)[1] for s in range(10)}
    assert len(digests) > 1
    assert all(f.start_time_ns >= 0 for f in a)


def _grounded(value: Sourced[object], kind: SourceKind) -> Sourced[object]:
    return value.model_copy(update={"source": kind, "provenance": "raw-log:sha256:" + "a" * 64})


def test_validity_levels_follow_provenance() -> None:
    assert assess_validity_level(CONFIG).level is SimulationValidityLevel.L1_APPROXIMATE_PHYSICS
    assert assess_validity_level(CONFIG, physics_enabled=False).level is SimulationValidityLevel.L0_FUNCTIONAL
    mass = {
        n: _grounded(getattr(CONFIG, n), SourceKind.MEASURED)
        for n in (
            "mass_kg",
            "displaced_volume_m3",
            "center_of_mass_body_m",
            "center_of_buoyancy_body_m",
            "inertia_diag_kgm2",
        )
    }
    thrusters = tuple(
        t.model_copy(
            update={
                f: _grounded(getattr(t, f), SourceKind.IDENTIFIED)
                for f in ("thrust_coefficient", "time_constant_s", "deadzone_command", "position_body_m")
            }
        )
        for t in CONFIG.thrusters
    )
    l2 = CONFIG.model_copy(update={**mass, "thrusters": thrusters})
    assert assess_validity_level(l2).level is SimulationValidityLevel.L2_CHARACTERIZED
    dyn = {
        n: _grounded(getattr(CONFIG, n), SourceKind.IDENTIFIED)
        for n in ("linear_drag", "quadratic_drag", "added_mass_diag")
    }
    l3 = l2.model_copy(update=dyn)
    assert assess_validity_level(l3).level is SimulationValidityLevel.L3_IDENTIFIED
    l4 = assess_validity_level(l3, validation_report_ref="reports/val-001.json")
    assert l4.level is SimulationValidityLevel.L4_VALIDATED_ENVELOPE and l4.validated_operating_envelope
    assert not assess_validity_level(l3).validated_operating_envelope


def test_unity_settings_load() -> None:
    settings, unity = load_unity_settings("configs/sim/unity_lockstep.yaml")
    assert unity.bridge.lock_step and unity.bridge.control_endpoint == "tcp://127.0.0.1:5591"
    assert unity.bridge.minimum_validity_level is SimulationValidityLevel.L1_APPROXIMATE_PHYSICS
    _, rt = load_unity_settings("configs/sim/unity_realtime.yaml")
    assert not rt.bridge.lock_step and rt.bridge.stream_endpoint == "tcp://127.0.0.1:5592"
    assert settings.run.lane.value == "simulation"


def _truth_server(ctx: zmq.Context, digest: str, stop: threading.Event) -> tuple[str, threading.Thread]:
    rep = ctx.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.setsockopt(zmq.RCVTIMEO, 1000)
    port = rep.bind_to_random_port("tcp://127.0.0.1")

    def serve() -> None:
        for _ in range(3):
            if stop.is_set():
                break
            try:
                raw = rep.recv()
            except zmq.Again:
                continue
            registry = {
                MessageKind.HANDSHAKE: HandshakeRequest,
                MessageKind.GET_GROUND_TRUTH: GroundTruthRequest,
                MessageKind.SURFACE_VISIBILITY: SurfaceVisibilityRequest,
            }
            env, body = decode_message(raw, registry)
            if env.kind is MessageKind.HANDSHAKE:
                reply = HandshakeAck(
                    simulator_id="conrad-unity-v2",
                    simulator_version="t",
                    protocol_version=PROTOCOL_VERSION,
                    schema_version=SCHEMA_VERSION,
                    robot_config_digest=digest,
                    clock_domain="SIM",
                    frame_convention=UNITY_FRAME_CONVENTION,
                    lock_step=True,
                    session_id="truth-1",
                    nonce=body.nonce,
                    validity_level=SimulationValidityLevel.L1_APPROXIMATE_PHYSICS,  # type: ignore[attr-defined]
                    physics_dt_ns=5_000_000,
                    sim_time_ns=0,
                    capabilities=WireCapabilities(capability_version="t"),
                )
                rep.send(encode_message(MessageKind.HANDSHAKE_ACK, reply, session_id="truth-1", seq=env.seq))
            elif env.kind is MessageKind.GET_GROUND_TRUTH:
                truth = GroundTruthReply(
                    sim_time_ns=7,
                    position_m=(-2.0, -5.0, 1.0),
                    orientation_wxyz=(1, 0, 0, 0),
                    linear_velocity_mps=(0.0, 0.0, 0.5),
                    angular_velocity_rps=(0.0, -0.2, 0.0),
                )
                rep.send(encode_message(MessageKind.GROUND_TRUTH, truth, session_id="truth-1", seq=env.seq))
            else:
                assert isinstance(body, SurfaceVisibilityRequest)
                assert body.origin_m == (1.0, 2.0, 3.0)
                reply = SurfaceVisibilityReply(visible=(True, False), first_hit_distance_m=(-1.0, 0.5))
                rep.send(
                    encode_message(
                        MessageKind.SURFACE_VISIBILITY_REPLY, reply, session_id="truth-1", seq=env.seq
                    )
                )
        rep.close(linger=0)

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    return f"tcp://127.0.0.1:{port}", t


def test_truth_client_converts_frames_on_its_own_endpoint() -> None:
    ctx = zmq.Context()
    stop = threading.Event()
    endpoint, thread = _truth_server(ctx, CONFIG.content_digest(), stop)
    transport = ZmqBridgeTransport(endpoint, None, ("127.0.0.1",), 1000)
    try:
        client = UnityTruthClient(transport, CONFIG.content_digest(), "SIM")
        client.connect(nonce="nonce-1234")
        truth = client.get_ground_truth()
        # Unity (-2, -5, 1) = right -2, up -5, forward 1  ->  Conrad (1, 2, -5)
        assert truth.pose.position_m == pytest.approx((1.0, 2.0, -5.0))
        assert truth.linear_velocity_world_mps == pytest.approx((0.5, 0.0, 0.0))
        assert truth.angular_velocity_world_rps == pytest.approx((0.0, 0.0, 0.2))
        assert client.surface_visibility((1, 2, 3), ((2, 2, 3), (3, 2, 3)), 0.05) == (True, False)
    finally:
        stop.set()
        transport.close()
        thread.join(timeout=2)
        ctx.destroy(linger=0)


def test_truth_bodies_are_not_accepted_on_the_control_endpoint() -> None:
    raw = encode_message(
        MessageKind.GROUND_TRUTH,
        GroundTruthReply(
            sim_time_ns=0,
            position_m=(0, 0, 0),
            orientation_wxyz=(1, 0, 0, 0),
            linear_velocity_mps=(0, 0, 0),
            angular_velocity_rps=(0, 0, 0),
        ),
        session_id="s",
        seq=1,
    )
    with pytest.raises(UnityProtocolError, match="not permitted"):
        decode_message(raw)
