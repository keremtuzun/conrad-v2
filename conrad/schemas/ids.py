"""Global identifiers. IDs are machine identifiers; human-readable names are metadata (ch2).

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

import os
import random
import threading
import time
from uuid import UUID

ScenarioId = UUID
MissionId = UUID
WorldEntityId = UUID
BeliefId = UUID
ObservationId = UUID
EvidenceId = UUID
SensorId = UUID
RobotId = UUID
EventId = UUID
RelationshipId = UUID
RunId = UUID
TraceId = UUID
CommandId = UUID
ProvenanceId = UUID
MessageId = UUID


def _uuid7(unix_ms: int, rand_a: int, rand_b: int) -> UUID:
    value = (unix_ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= (rand_a & 0xFFF) << 64
    value |= 0b10 << 62
    value |= rand_b & ((1 << 62) - 1)
    return UUID(int=value)


class IdFactory:
    """UUIDv7 generator.

    With ``seed=None`` it uses wall clock and OS entropy. With a seed it is fully deterministic
    (logical millisecond counter + seeded PRNG) so that replays reproduce identical identifiers.
    """

    def __init__(self, seed: int | None = None, start_ms: int = 1_700_000_000_000) -> None:
        self._seed = seed
        self._rng = random.Random(seed) if seed is not None else None
        self._logical_ms = start_ms
        self._lock = threading.Lock()
        self._issued: int = 0

    @property
    def deterministic(self) -> bool:
        return self._rng is not None

    @property
    def issued(self) -> int:
        return self._issued

    def new(self) -> UUID:
        with self._lock:
            self._issued += 1
            if self._rng is None:
                raw = int.from_bytes(os.urandom(10), "big")
                return _uuid7(time.time_ns() // 1_000_000, raw >> 62, raw)
            self._logical_ms += 1
            return _uuid7(self._logical_ms, self._rng.getrandbits(12), self._rng.getrandbits(62))


_default_factory = IdFactory()


def new_id() -> UUID:
    """Non-deterministic UUIDv7 from the process default factory."""
    return _default_factory.new()
