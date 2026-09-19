"""Transports for the Unity bridge: TCP (live, default), ZeroMQ (optional), recording and deterministic replay.

The protocol is transport independent (ch20 Technology bridge). The Unity player speaks
:class:`TcpBridgeTransport`: length-prefixed JSON frames over plain TCP (``System.Net.Sockets`` on the C#
side, no third-party DLLs). :class:`ZmqBridgeTransport` stays as an optional alternative for Python-side
peers (the test mock); the Unity player does not serve ZeroMQ because NetMQ was never vendored.
Endpoints must be literal loopback/private addresses that appear in the configured allow-list.

Wire frame (both directions, both languages): ``uint32 big-endian payload length`` + ``payload`` where the
payload is one UTF-8 JSON envelope. Frames above :data:`MAX_FRAME_BYTES` are refused.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import socket
import struct
import time
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from conrad.robotics.hardware.interface import HardwareUnavailableError

FRAME_HEADER = struct.Struct(">I")
MAX_FRAME_BYTES = 64 * 1024 * 1024  # same limit as LengthPrefixedFraming.MaxFrameBytes in C#


class UnityBridgeError(HardwareUnavailableError):
    """The bridge is unusable; callers must treat the hardware as unavailable (fail closed)."""


class UnityBridgeTimeout(UnityBridgeError):
    pass


class UnityPeerError(UnityBridgeError):
    """Endpoint or simulator identity is not on the allow-list."""


def validate_private_endpoint(endpoint: str, allowed_peers: tuple[str, ...]) -> str:
    """Return the host of ``endpoint`` or raise. Only tcp://<literal private ip>:<port> is accepted."""
    parsed = urlparse(endpoint)
    if parsed.scheme != "tcp":
        raise UnityPeerError(f"unsupported bridge scheme {parsed.scheme!r}; only tcp:// is permitted")
    host = parsed.hostname
    try:
        port = parsed.port
    except ValueError as exc:
        raise UnityPeerError(f"endpoint {endpoint!r} has an invalid port") from exc
    if not host or port is None:
        raise UnityPeerError(f"endpoint {endpoint!r} must be tcp://<ip>:<port>")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError as exc:
        raise UnityPeerError(f"endpoint host {host!r} must be a literal IP address") from exc
    if ip.is_unspecified:
        raise UnityPeerError("wildcard addresses are forbidden for control endpoints (ch34 Network)")
    if not (ip.is_loopback or ip.is_private or ip.is_link_local):
        raise UnityPeerError(f"endpoint host {host} is not a loopback/private address")
    if host not in allowed_peers:
        raise UnityPeerError(f"endpoint host {host} is not in allowed_peers {list(allowed_peers)}")
    return host


def encode_frame(payload: bytes) -> bytes:
    """One wire frame: 4-byte unsigned big-endian length, then the payload."""
    if len(payload) > MAX_FRAME_BYTES:
        raise UnityBridgeError(f"frame of {len(payload)} bytes exceeds {MAX_FRAME_BYTES}")
    return FRAME_HEADER.pack(len(payload)) + payload


class BridgeTransport(Protocol):
    def request(self, payload: bytes) -> bytes: ...

    def poll_stream(self, max_messages: int = 64) -> list[bytes]: ...

    def close(self) -> None: ...


def _split_endpoint(endpoint: str, allowed_peers: tuple[str, ...]) -> tuple[str, int]:
    host = validate_private_endpoint(endpoint, allowed_peers)
    port = urlparse(endpoint).port
    if port is None:  # pragma: no cover - validate_private_endpoint already guarantees a port
        raise UnityPeerError(f"endpoint {endpoint!r} has no port")
    return host, port


class TcpBridgeTransport:
    """Length-prefixed JSON frames over TCP: one request/reply connection + an optional push stream.

    Fail closed: a timeout, a short read, an oversized frame or a peer that is not the validated
    address closes the transport, after which every call raises :class:`UnityBridgeError`.
    """

    def __init__(
        self,
        control_endpoint: str,
        stream_endpoint: str | None,
        allowed_peers: tuple[str, ...],
        request_timeout_ms: int,
        connect_timeout_ms: int | None = None,
    ) -> None:
        if request_timeout_ms <= 0:
            raise ValueError("request_timeout_ms must be positive")
        host, port = _split_endpoint(control_endpoint, allowed_peers)
        stream = None if stream_endpoint is None else _split_endpoint(stream_endpoint, allowed_peers)
        self._timeout_s = request_timeout_ms / 1000.0
        connect_s = self._timeout_s if connect_timeout_ms is None else connect_timeout_ms / 1000.0
        self._stream: socket.socket | None = None
        self._stream_buf = bytearray()
        self._sock: socket.socket | None = self._connect(host, port, connect_s)
        if stream is not None:
            try:
                self._stream = self._connect(stream[0], stream[1], connect_s)
            except UnityBridgeError:
                self.close()
                raise
            self._stream.setblocking(False)

    @staticmethod
    def _connect(host: str, port: int, timeout_s: float) -> socket.socket:
        try:
            sock = socket.create_connection((host, port), timeout=timeout_s)
        except OSError as exc:
            raise UnityBridgeError(f"cannot connect to Unity at {host}:{port}: {exc}") from exc
        peer = str(sock.getpeername()[0])
        if ipaddress.ip_address(peer) != ipaddress.ip_address(host):
            sock.close()
            raise UnityPeerError(f"connected peer {peer} is not the validated endpoint host {host}")
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    @staticmethod
    def _read_exactly(sock: socket.socket, n: int, deadline: float) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            sock.settimeout(remaining)
            chunk = sock.recv(min(n - len(buf), 1 << 20))
            if not chunk:
                raise ConnectionError("Unity closed the connection")
            buf.extend(chunk)
        return bytes(buf)

    @property
    def closed(self) -> bool:
        return self._sock is None

    def request(self, payload: bytes) -> bytes:
        sock = self._sock
        if sock is None:
            raise UnityBridgeError("bridge transport is closed")
        deadline = time.monotonic() + self._timeout_s
        try:
            sock.settimeout(self._timeout_s)
            sock.sendall(encode_frame(payload))
            (length,) = FRAME_HEADER.unpack(self._read_exactly(sock, FRAME_HEADER.size, deadline))
            if length > MAX_FRAME_BYTES:
                raise UnityBridgeError(f"Unity announced a {length}-byte frame (limit {MAX_FRAME_BYTES})")
            return self._read_exactly(sock, length, deadline)
        except TimeoutError as exc:  # socket.timeout is TimeoutError on 3.10+
            # A request/reply stream that missed its reply is out of step; never send on it again.
            self.close()
            raise UnityBridgeTimeout(f"no reply from Unity within {self._timeout_s * 1000:.0f} ms") from exc
        except UnityBridgeError:
            self.close()
            raise
        except OSError as exc:
            self.close()
            raise UnityBridgeError(f"TCP failure: {exc}") from exc

    def poll_stream(self, max_messages: int = 64) -> list[bytes]:
        stream = self._stream
        if stream is None:
            return []
        try:
            while True:
                chunk = stream.recv(1 << 20)
                if not chunk:
                    raise ConnectionError("Unity closed the sensor stream")
                self._stream_buf.extend(chunk)
        except BlockingIOError:
            pass
        except OSError as exc:
            self.close()
            raise UnityBridgeError(f"sensor stream failure: {exc}") from exc
        out: list[bytes] = []
        while len(out) < max_messages and len(self._stream_buf) >= FRAME_HEADER.size:
            (length,) = FRAME_HEADER.unpack_from(self._stream_buf, 0)
            if length > MAX_FRAME_BYTES:
                self.close()
                raise UnityBridgeError(f"stream frame of {length} bytes exceeds the limit")
            end = FRAME_HEADER.size + length
            if len(self._stream_buf) < end:
                break
            out.append(bytes(self._stream_buf[FRAME_HEADER.size : end]))
            del self._stream_buf[:end]
        return out

    def close(self) -> None:
        for sock in (self._sock, self._stream):
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()
        self._sock = None
        self._stream = None


class ZmqBridgeTransport:
    """Optional: REQ socket for control + optional SUB socket for a PUB stream (Python peers only)."""

    def __init__(
        self,
        control_endpoint: str,
        stream_endpoint: str | None,
        allowed_peers: tuple[str, ...],
        request_timeout_ms: int,
    ) -> None:
        validate_private_endpoint(control_endpoint, allowed_peers)
        if stream_endpoint is not None:
            validate_private_endpoint(stream_endpoint, allowed_peers)
        if request_timeout_ms <= 0:
            raise ValueError("request_timeout_ms must be positive")
        import zmq  # optional: the default TCP transport needs nothing beyond the standard library

        self._zmq: Any = zmq
        self._timeout_ms = request_timeout_ms
        self._ctx: Any = zmq.Context()
        self._req: Any = self._ctx.socket(zmq.REQ)
        self._req.setsockopt(zmq.LINGER, 0)
        self._req.setsockopt(zmq.RCVTIMEO, request_timeout_ms)
        self._req.setsockopt(zmq.SNDTIMEO, request_timeout_ms)
        self._req.connect(control_endpoint)
        self._sub: Any = None
        if stream_endpoint is not None:
            self._sub = self._ctx.socket(zmq.SUB)
            self._sub.setsockopt(zmq.LINGER, 0)
            self._sub.setsockopt(zmq.RCVHWM, 256)
            self._sub.connect(stream_endpoint)
            self._sub.subscribe(b"")

    def request(self, payload: bytes) -> bytes:
        zmq = self._zmq
        if self._req is None:
            raise UnityBridgeError("bridge transport is closed")
        try:
            self._req.send_multipart([payload])
            parts = self._req.recv_multipart()
        except zmq.Again as exc:
            # A REQ socket that missed its reply is unusable; close so nothing can be sent blind.
            self.close()
            raise UnityBridgeTimeout(f"no reply from Unity within {self._timeout_ms} ms") from exc
        except zmq.ZMQError as exc:
            self.close()
            raise UnityBridgeError(f"ZeroMQ failure: {exc}") from exc
        if len(parts) != 1:
            self.close()
            raise UnityBridgeError(f"expected a single-frame reply, got {len(parts)} frames")
        return bytes(parts[0])

    def poll_stream(self, max_messages: int = 64) -> list[bytes]:
        zmq = self._zmq
        if self._sub is None:
            return []
        out: list[bytes] = []
        while len(out) < max_messages:
            try:
                parts = self._sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            out.append(bytes(parts[-1]))
        return out

    def close(self) -> None:
        if self._ctx.closed:
            return
        for sock in (self._req, self._sub):
            if sock is not None:
                sock.close(linger=0)
        self._req = None
        self._sub = None
        # destroy() never blocks on queued messages, unlike term(); a close can never hang.
        self._ctx.destroy(linger=0)


class RecordingTransport:
    """Writes every exchanged message verbatim to a JSONL wire log (input to :class:`ReplayTransport`)."""

    def __init__(self, inner: BridgeTransport, log_path: str | Path) -> None:
        self._inner = inner
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", encoding="utf-8", newline="\n")

    def _write(self, direction: str, data: bytes) -> None:
        self._handle.write(json.dumps({"dir": direction, "data": data.decode("utf-8")}) + "\n")
        self._handle.flush()

    def request(self, payload: bytes) -> bytes:
        self._write("req", payload)
        reply = self._inner.request(payload)
        self._write("rep", reply)
        return reply

    def poll_stream(self, max_messages: int = 64) -> list[bytes]:
        messages = self._inner.poll_stream(max_messages)
        for m in messages:
            self._write("pub", m)
        return messages

    def close(self) -> None:
        self._handle.close()
        self._inner.close()


class ReplayDivergenceError(UnityBridgeError):
    pass


class ReplayTransport:
    """Serves a recorded wire log. Requests must match the recording byte-for-byte."""

    def __init__(self, log_path: str | Path) -> None:
        self._records: list[tuple[str, bytes]] = []
        for line in Path(log_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                self._records.append((rec["dir"], rec["data"].encode("utf-8")))
        self._cursor = 0

    def request(self, payload: bytes) -> bytes:
        while self._cursor < len(self._records) and self._records[self._cursor][0] == "pub":
            self._cursor += 1
        if self._cursor + 1 >= len(self._records):
            raise ReplayDivergenceError("replay log exhausted")
        direction, recorded = self._records[self._cursor]
        if direction != "req" or recorded != payload:
            raise ReplayDivergenceError(f"request {self._cursor} diverges from the recorded run")
        reply_dir, reply = self._records[self._cursor + 1]
        if reply_dir != "rep":
            raise ReplayDivergenceError("wire log is corrupt: request without reply")
        self._cursor += 2
        return reply

    def poll_stream(self, max_messages: int = 64) -> list[bytes]:
        out: list[bytes] = []
        while (
            self._cursor < len(self._records)
            and self._records[self._cursor][0] == "pub"
            and len(out) < max_messages
        ):
            out.append(self._records[self._cursor][1])
            self._cursor += 1
        return out

    def close(self) -> None:
        self._cursor = len(self._records)
