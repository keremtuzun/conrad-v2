"""TCP length-prefixed framing (fail-closed edge cases) and the CONFIGURE_SCENE geometry loader."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from conrad.adapters.unity import (
    MessageKind,
    SceneConfigureRequest,
    TcpBridgeTransport,
    UnityBridgeError,
    UnityBridgeTimeout,
    UnityPeerError,
    decode_message,
    encode_frame,
    encode_message,
)
from conrad.adapters.unity.transport import FRAME_HEADER, MAX_FRAME_BYTES
from conrad.sim.unity.scene import (
    BoxPrimitive,
    CapsulePrimitive,
    HeightfieldPrimitive,
    SceneGeometry,
    load_scene_geometry,
)

PEERS = ("127.0.0.1",)


def _server(behaviour: Callable[[socket.socket], None]) -> Iterator[str]:
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]

    def run() -> None:
        conn, _ = listener.accept()
        with conn:
            behaviour(conn)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    yield f"tcp://127.0.0.1:{port}"
    t.join(timeout=2)
    listener.close()


def _read_frame(conn: socket.socket) -> bytes:
    head = conn.recv(4, socket.MSG_WAITALL)
    return conn.recv(FRAME_HEADER.unpack(head)[0], socket.MSG_WAITALL)


def test_frame_is_big_endian_length_prefixed() -> None:
    assert encode_frame(b"abc") == b"\x00\x00\x00\x03abc"
    assert FRAME_HEADER.size == 4


def test_request_reply_round_trip() -> None:
    def echo(conn: socket.socket) -> None:
        conn.sendall(encode_frame(_read_frame(conn)[::-1]))

    for endpoint in _server(echo):
        t = TcpBridgeTransport(endpoint, None, PEERS, 1000)
        assert t.request(b"hello") == b"olleh"
        t.close()
        assert t.closed
        with pytest.raises(UnityBridgeError):
            t.request(b"x")


@pytest.mark.parametrize(
    ("name", "behaviour", "error"),
    [
        (
            "oversized",
            lambda c: (_read_frame(c), c.sendall(FRAME_HEADER.pack(MAX_FRAME_BYTES + 1))),
            UnityBridgeError,
        ),
        (
            "closed-mid-frame",
            lambda c: (_read_frame(c), c.sendall(FRAME_HEADER.pack(10) + b"abc")),
            UnityBridgeError,
        ),
        ("silent", lambda c: (_read_frame(c), threading.Event().wait(0.6)), UnityBridgeTimeout),
    ],
)
def test_bad_replies_fail_closed(
    name: str, behaviour: Callable[[socket.socket], None], error: type[Exception]
) -> None:
    for endpoint in _server(behaviour):
        t = TcpBridgeTransport(endpoint, None, PEERS, 300)
        with pytest.raises(error):
            t.request(b"ping")
        assert t.closed, name


def test_refused_endpoints_never_connect() -> None:
    for endpoint in ("tcp://8.8.8.8:1", "tcp://0.0.0.0:1", "tcp://localhost:1", "udp://127.0.0.1:1"):
        with pytest.raises(UnityPeerError):
            TcpBridgeTransport(endpoint, None, ("8.8.8.8", "0.0.0.0", "localhost", "127.0.0.1"), 100)
    with pytest.raises(UnityBridgeError, match="cannot connect"):
        port = socket.create_server(("127.0.0.1", 0))
        free = port.getsockname()[1]
        port.close()
        TcpBridgeTransport(f"tcp://127.0.0.1:{free}", None, PEERS, 100, connect_timeout_ms=200)


def _scene() -> SceneGeometry:
    return SceneGeometry(
        primitives=(
            BoxPrimitive(id="wall", center_m=(5.0, 0.0, -10.0), size_m=(0.5, 4.0, 3.0)),
            CapsulePrimitive(id="pipe", p0_m=(8.0, -5.0, -29.7), p1_m=(8.0, 5.0, -29.7), radius_m=0.3),
            HeightfieldPrimitive(
                id="floor",
                origin_m=(-5.0, -5.0, -30.0),
                spacing_m=(1.0, 1.0),
                heights_m=((0.0, 0.1), (0.2, 0.3)),
            ),
        )
    )


def test_scene_geometry_serialises_to_the_wire_body(tmp_path: Path) -> None:
    body = _scene().to_json()
    assert body["frame"] == "WORLD" and body["replace"] is True
    kinds = [p["kind"] for p in body["primitives"]]
    assert kinds == ["box", "capsule", "heightfield"]
    assert all("schema_version" not in p for p in body["primitives"])
    assert body["primitives"][2]["heights_m"] == [[0.0, 0.1], [0.2, 0.3]]
    raw = encode_message(
        MessageKind.CONFIGURE_SCENE, SceneConfigureRequest.model_validate(body), session_id="s", seq=1
    )
    env, decoded = decode_message(raw)
    assert env.kind is MessageKind.CONFIGURE_SCENE and isinstance(decoded, SceneConfigureRequest)
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    assert load_scene_geometry(path).to_json() == body


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "box", "id": "b", "center_m": [0, 0, 0], "size_m": [0, 1, 1]},
        {"kind": "capsule", "id": "c", "p0_m": [0, 0, 0], "p1_m": [0, 0, 0], "radius_m": 0.1},
        {
            "kind": "heightfield",
            "id": "h",
            "origin_m": [0, 0, 0],
            "spacing_m": [1, 1],
            "heights_m": [[0, 0], [0]],
        },
        {
            "kind": "heightfield",
            "id": "h",
            "origin_m": [0, 0, 0],
            "spacing_m": [0, 1],
            "heights_m": [[0, 0], [0, 0]],
        },
        {"kind": "cone", "id": "x"},
    ],
)
def test_invalid_primitives_are_refused(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SceneGeometry.model_validate({"primitives": [bad]})


def test_duplicate_primitive_ids_are_refused() -> None:
    box = {"kind": "box", "id": "same", "center_m": [0, 0, 0], "size_m": [1, 1, 1]}
    with pytest.raises(ValidationError, match="unique"):
        SceneGeometry.model_validate({"primitives": [box, box]})
