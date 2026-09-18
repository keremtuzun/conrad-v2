"""Common simulator contract for every domain twin (ch2 Twin interface). TRUTH PLANE.

A twin returns sensor-shaped :class:`Observation` objects for the observation plane and keeps
supervision in a physically separate channel (:class:`TwinSample.supervision`) that only training
and evaluation code may read.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Observation
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.truth import SupervisionLabel, TruthState
from conrad.schemas.world import Domain, Scenario, ScenarioEvent, SensorSpec


@dataclass(frozen=True)
class SensingContext:
    """Everything a twin needs to emit a contract-valid Observation. ``estimated_pose`` is what the
    robot believes; ``true_pose`` is where the sensor really is and never leaves the truth plane."""

    mission_id: UUID
    run_id: UUID
    trace_id: UUID
    sensor: SensorSpec
    true_pose: Pose
    estimated_pose: Pose | None
    timestamp: TimeStamp
    degradation: dict[str, float]


@dataclass(frozen=True)
class TwinSample:
    observation: Observation
    supervision: SupervisionLabel | None


class Twin(ABC):
    domain: Domain
    twin_version: str

    def __init__(self, ids: IdFactory, store: ObjectStore) -> None:
        self.ids = ids
        self.store = store

    @abstractmethod
    def initialize(self, scenario: Scenario) -> None:
        """Read only this twin's scenario section plus explicitly shared context."""

    @abstractmethod
    def step(self, dt_s: float, events: Sequence[ScenarioEvent] = ()) -> None: ...

    @abstractmethod
    def get_truth(self, timestamp: TimeStamp) -> list[TruthState]: ...

    @abstractmethod
    def generate_observation(self, ctx: SensingContext) -> list[TwinSample]:
        """Zero or more observations. An empty list means nothing was sensed (not fabricated)."""

    @abstractmethod
    def export_domain_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def reset(self, seed: int) -> None: ...
