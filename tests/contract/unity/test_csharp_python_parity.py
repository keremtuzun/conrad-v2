"""Cross-language contract: the Unity C# sources must speak exactly the Python wire protocol.

The C# cannot be compiled in this environment, so these tests read the C# source text:
* every JSON key C# writes into a message body matches the pydantic model (extra="forbid" on the Python side);
* layouts, format strings, fault vocabulary and the frame convention string are identical;
* the C# frame-conversion expressions, transliterated to Python, agree with UnityFrameMapper numerically.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from conrad.adapters.unity.frames import DEFAULT_MAPPER, UNITY_FRAME_CONVENTION
from conrad.adapters.unity.protocol import (
    CAMERA_LAYOUT,
    DEPTH_LAYOUT,
    IMU_LAYOUT,
    SONAR_LAYOUT,
    ErrorReply,
    FaultAck,
    FaultType,
    HandshakeAck,
    MessageKind,
    MetricsReply,
    ResetAck,
    SensorPacket,
    StatePacket,
    WireBattery,
    WireCapabilities,
    WireCommandAck,
    WireEnvelope,
    WireHealth,
    WireThrusterState,
)
from conrad.sim.unity.robot_export import BODY_CONVENTION, UNITY_ROBOT_CONFIG_FORMAT
from conrad.sim.unity.truth import GroundTruthReply

ROOT = Path(__file__).resolve().parents[3] / "unity" / "ConradUnityV2" / "Assets" / "Conrad"
KEY = re.compile(r'\["([a-z_0-9]+)"\]\s*=')


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _keys(text: str) -> set[str]:
    return set(KEY.findall(text))


def _fields(*models: type) -> set[str]:
    out: set[str] = set()
    for m in models:
        out |= set(m.model_fields)
    return out


def test_eight_modules_and_asmdefs_exist() -> None:
    for module in (
        "VehicleDynamics",
        "Hydrodynamics",
        "Propulsion",
        "EnvironmentInteraction",
        "SensorSimulation",
        "RobotHardwareSimulation",
        "MissionExperimentRuntime",
        "ExternalInterfaces",
    ):
        assert list((ROOT / module).rglob("*.cs")), module
    assert (ROOT / "ConradUnityV2.asmdef").is_file()
    assert (ROOT / "ExternalInterfaces" / "Transport" / "ConradUnityV2.Transport.asmdef").is_file()


def test_sensor_packet_keys_match_exactly() -> None:
    assert _keys(_src("SensorSimulation/SimSensor.cs")) == _fields(SensorPacket) - {"schema_version"}


def test_robot_hardware_server_bodies_match() -> None:
    keys = _keys(_src("RobotHardwareSimulation/RobotHardwareServer.cs"))
    models = (WireCapabilities, WireCommandAck, StatePacket, WireThrusterState, WireBattery, WireHealth)
    for m in models:
        assert set(m.model_fields) - {"schema_version"} <= keys, m.__name__
    assert keys <= _fields(*models)


def test_protocol_handler_bodies_match() -> None:
    keys = _keys(_src("ExternalInterfaces/Protocol/BridgeProtocolHandler.cs"))
    models = (WireEnvelope, HandshakeAck, ResetAck, FaultAck, MetricsReply, ErrorReply)
    for m in models:
        assert set(m.model_fields) - {"schema_version", "metrics"} <= keys | {"metrics"}, m.__name__
    metric_keys = {"steps", "battery_remaining_fraction", "power_w", "dropped_frames"}
    replay_log_keys = {"request", "ack", "reasons"}  # JSONL replay records, never sent on the wire
    assert keys <= _fields(*models) | metric_keys | replay_log_keys


def test_ground_truth_keys_match() -> None:
    src = _src("MissionExperimentRuntime/ExperimentRuntime.cs")
    body = src[src.index("public Dictionary<string, object> TruthWire") : src.index("CurrentTruthWire")]
    assert _keys(body) == set(GroundTruthReply.model_fields) - {"schema_version"}


def test_message_kinds_and_layouts_and_formats() -> None:
    handler = _src("ExternalInterfaces/Protocol/BridgeProtocolHandler.cs") + _src(
        "ExternalInterfaces/Transport/ZmqBridgeServer.cs"
    )
    used = set(re.findall(r'"([A-Z_]+)"', handler)) & {k.value for k in MessageKind}
    assert used >= {
        "HANDSHAKE",
        "HANDSHAKE_ACK",
        "RESET",
        "RESET_ACK",
        "STEP",
        "POLL",
        "STATE",
        "COMMAND",
        "COMMAND_ACK",
        "INJECT_FAULT",
        "FAULT_ACK",
        "GET_METRICS",
        "METRICS",
        "SENSOR",
        "GET_GROUND_TRUTH",
        "GROUND_TRUTH",
        "ERROR",
    }
    sensors = _src("SensorSimulation/NavigationSensors.cs") + _src("SensorSimulation/ImagingSensors.cs")
    for layout in (IMU_LAYOUT, DEPTH_LAYOUT, CAMERA_LAYOUT, SONAR_LAYOUT):
        assert f'"{layout}"' in sensors
    core = _src("Core/MathTypes.cs") + _src("Core/RobotConfigLoader.cs")
    assert f'"{UNITY_FRAME_CONVENTION}"' in core and f'"{BODY_CONVENTION}"' in core
    assert f'"{UNITY_ROBOT_CONFIG_FORMAT}"' in core


def test_every_python_fault_type_is_handled_or_explicitly_refused() -> None:
    injector = _src("MissionExperimentRuntime/FaultInjector.cs")
    start = injector.index("Supported = new HashSet<string>")
    supported_block = injector[start : injector.index("};", start)]
    supported = set(re.findall(r'"([A-Z_]+)"', supported_block))
    for fault in FaultType:
        if fault is FaultType.LOCALIZATION_DEGRADATION:
            assert fault.value not in supported and "LOCALIZATION_DEGRADATION is refused" in injector
        else:
            assert fault.value in supported, fault
            assert f'case "{fault.value}":' in injector, fault


def _transliterate(expr: str) -> str:
    return re.sub(r"\b([cu])\.([XYZW])\b", lambda m: f"{m.group(1)}['{m.group(2)}']", expr)


def _csharp_function(name: str, arg: str) -> str:
    src = _src("Core/MathTypes.cs")
    m = re.search(rf"public static \w+ {name}\(\w+ {arg}\) => new \w+\((.*?)\);", src)
    assert m is not None, name
    return m.group(1)


@pytest.mark.parametrize(
    ("name", "arg", "kind", "reference"),
    [
        ("PointToUnity", "c", "vec", DEFAULT_MAPPER.point_to_unity),
        ("PointToConrad", "u", "vec", DEFAULT_MAPPER.point_to_conrad),
        ("AxialToUnity", "c", "vec", DEFAULT_MAPPER.axial_to_unity),
        ("AxialToConrad", "u", "vec", DEFAULT_MAPPER.axial_to_conrad),
        ("QuatToUnity", "c", "quat", DEFAULT_MAPPER.quat_to_unity),
        ("QuatToConrad", "u", "quat", DEFAULT_MAPPER.quat_to_conrad),
    ],
)
def test_csharp_frame_map_equals_python_mapper(name: str, arg: str, kind: str, reference) -> None:
    expr = _transliterate(_csharp_function(name, arg))
    rng = np.random.default_rng(0)
    for _ in range(20):
        if kind == "vec":
            v = tuple(float(x) for x in rng.normal(size=3))
            env = {arg: {"X": v[0], "Y": v[1], "Z": v[2]}}
        else:
            q = rng.normal(size=4)
            q = q / np.linalg.norm(q)
            v = (float(abs(q[0])), float(q[1]), float(q[2]), float(q[3]))
            env = {arg: {"W": v[0], "X": v[1], "Y": v[2], "Z": v[3]}}
        got = eval(f"({expr})", {}, env)  # expression text comes from this repository's own C# source
        assert got == pytest.approx(reference(v), abs=1e-12), name
