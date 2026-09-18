"""Sensor packet: acquisition-stamped, framed, digest-verified binary payload (base64 on the wire).

Acquisition time is distinct from delivery time and is never rewritten by any hop.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import numpy as np
from pydantic import Field, model_validator

from conrad.adapters.unity.vocabulary import (
    _NUMPY_DTYPE,
    DIGEST_PATTERN,
    PayloadEncoding,
    UnityProtocolError,
    WireModel,
)

_DIGEST = DIGEST_PATTERN


class SensorPacket(WireModel):
    sensor_name: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)
    clock_domain: str = Field(min_length=1)
    acquisition_time_ns: int = Field(ge=0)
    delivery_time_ns: int = Field(ge=0)
    sequence_index: int = Field(ge=0)
    health: str = "OK"
    layout: str
    encoding: PayloadEncoding
    shape: tuple[int, ...]
    units: str = Field(min_length=1)
    payload_b64: str
    payload_digest: str = Field(pattern=_DIGEST)
    calibration_ref: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _causal(self) -> SensorPacket:
        if self.delivery_time_ns < self.acquisition_time_ns:
            raise ValueError("sensor packet delivered before it was acquired")
        return self

    def payload_bytes(self) -> bytes:
        """Decode and verify. A digest or size mismatch is a protocol error, never a warning."""
        try:
            raw = base64.b64decode(self.payload_b64, validate=True)
        except ValueError as exc:
            raise UnityProtocolError(f"sensor {self.sensor_name}: payload is not valid base64") from exc
        actual = hashlib.sha256(raw).hexdigest()
        if actual != self.payload_digest:
            raise UnityProtocolError(
                f"sensor {self.sensor_name}: payload digest mismatch ({actual} != {self.payload_digest})"
            )
        itemsize = np.dtype(_NUMPY_DTYPE[self.encoding]).itemsize
        expected = int(np.prod(self.shape, dtype=np.int64)) * itemsize if self.shape else 0
        if expected != len(raw):
            raise UnityProtocolError(
                f"sensor {self.sensor_name}: payload is {len(raw)} bytes, shape/encoding imply {expected}"
            )
        return raw

    def payload_array(self) -> np.ndarray:
        raw = self.payload_bytes()
        return np.frombuffer(raw, dtype=_NUMPY_DTYPE[self.encoding]).reshape(self.shape)


def make_sensor_packet(
    *,
    sensor_name: str,
    modality: str,
    frame_id: str,
    clock_domain: str,
    acquisition_time_ns: int,
    delivery_time_ns: int,
    sequence_index: int,
    layout: str,
    array: np.ndarray,
    encoding: PayloadEncoding,
    units: str,
    health: str = "OK",
    calibration_ref: str | None = None,
    context: dict[str, Any] | None = None,
) -> SensorPacket:
    """Reference encoder used by simulators written in Python (and by the mock Unity server)."""
    data = np.ascontiguousarray(array, dtype=_NUMPY_DTYPE[encoding]).tobytes()
    return SensorPacket(
        sensor_name=sensor_name,
        modality=modality,
        frame_id=frame_id,
        clock_domain=clock_domain,
        acquisition_time_ns=acquisition_time_ns,
        delivery_time_ns=delivery_time_ns,
        sequence_index=sequence_index,
        health=health,
        layout=layout,
        encoding=encoding,
        shape=tuple(int(s) for s in array.shape),
        units=units,
        payload_b64=base64.b64encode(data).decode("ascii"),
        payload_digest=hashlib.sha256(data).hexdigest(),
        calibration_ref=calibration_ref,
        context=context or {},
    )
