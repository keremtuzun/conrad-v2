"""Transports for the Unity bridge: ZeroMQ (live), recording wrapper and deterministic replay.

The protocol is transport independent (ch20 Technology bridge); ZeroMQ is the first concrete choice.
Endpoints must be literal loopback/private addresses that appear in the configured allow-list.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import zmq

from conrad.robotics.hardware.interface import HardwareUnavailableError


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


class BridgeTransport(Protocol):
    def request(self, payload: bytes) -> bytes: ...

    def poll_stream(self, max_messages: int = 64) -> list[bytes]: ...

    def close(self) -> None: ...


class ZmqBridgeTransport:
    """REQ socket for control + optional SUB socket for the sensor PUB stream."""

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
        self._timeout_ms = request_timeout_ms
        self._ctx = zmq.Context()
        self._req: zmq.Socket[bytes] | None = self._ctx.socket(zmq.REQ)
        self._req.setsockopt(zmq.LINGER, 0)
        self._req.setsockopt(zmq.RCVTIMEO, request_timeout_ms)
        self._req.setsockopt(zmq.SNDTIMEO, request_timeout_ms)
        self._req.connect(control_endpoint)
        self._sub: zmq.Socket[bytes] | None = None
        if stream_endpoint is not None:
            self._sub = self._ctx.socket(zmq.SUB)
            self._sub.setsockopt(zmq.LINGER, 0)
            self._sub.setsockopt(zmq.RCVHWM, 256)
            self._sub.connect(stream_endpoint)
            self._sub.subscribe(b"")

    def request(self, payload: bytes) -> bytes:
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
        return parts[0]

    def poll_stream(self, max_messages: int = 64) -> list[bytes]:
        if self._sub is None:
            return []
        out: list[bytes] = []
        while len(out) < max_messages:
            try:
                parts = self._sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            out.append(parts[-1])
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
