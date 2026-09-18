"""Shared world: one WorldEntity identity, several domain states (ch2).

This module contains NO hidden state. Hidden truth lives in ``conrad.schemas.truth`` which the
belief and decision planes are forbidden to import (tests/leakage).

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from conrad.schemas.base import VersionedModel
from conrad.schemas.frames import FrameConvention, Pose
from conrad.schemas.timebase import TimeStamp


class Domain(str, Enum):
    TECHNICAL = "TECHNICAL"
    ECOLOGICAL = "ECOLOGICAL"
    SPATIAL = "SPATIAL"


class DomainOwnership(VersionedModel):
    """Which twin owns which aspect of one physical entity."""

    spatial: bool = True
    technical: bool = False
    ecological: bool = False


class WorldEntity(VersionedModel):
    id: UUID
    entity_type: str = Field(min_length=1)
    parent_id: UUID | None = None
    geometry_ref: str | None = None
    reference_frame: str = Field(min_length=1)
    created_at: TimeStamp
    retired_at: TimeStamp | None = None
    domain_ownership: DomainOwnership = DomainOwnership()
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="aliases and other non-identity metadata"
    )


class CrossDomainRelationType(str, Enum):
    OCCUPIES = "OCCUPIES"
    AFFECTS_OBSERVABILITY_OF = "AFFECTS_OBSERVABILITY_OF"
    ATTACHED_TO = "ATTACHED_TO"
    CONTEXT_FOR = "CONTEXT_FOR"


class MissionSpec(VersionedModel):
    mission_id: UUID
    mission_type: str
    target_entity_ids: tuple[UUID, ...] = ()
    boundary_min_m: tuple[float, float, float] | None = None
    boundary_max_m: tuple[float, float, float] | None = None
    time_budget_s: float | None = Field(default=None, gt=0)
    energy_budget_j: float | None = Field(default=None, gt=0)
    parameters: dict[str, Any] = Field(default_factory=dict)


class SensorSpec(VersionedModel):
    sensor_id: UUID
    modality: str
    frame_id: str
    mount_pose: Pose
    rate_hz: float = Field(gt=0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    calibration_ref: str | None = None


class RobotSpec(VersionedModel):
    robot_id: UUID
    robot_config_ref: str
    initial_pose: Pose
    sensors: tuple[SensorSpec, ...] = ()


class ScenarioEvent(VersionedModel):
    event_id: UUID
    time_s: float = Field(ge=0)
    event_type: str
    target_entity_id: UUID | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class Scenario(VersionedModel):
    """One scenario definition feeds all twins; twins read only their own section plus shared context."""

    scenario_id: UUID
    scenario_version: str
    seed: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    coordinate_system: FrameConvention = FrameConvention()
    world_entities: tuple[WorldEntity, ...]
    environment: dict[str, Any] = Field(default_factory=dict)
    structural_state: dict[str, Any] = Field(default_factory=dict)
    ecological_state: dict[str, Any] = Field(default_factory=dict)
    spatial_state: dict[str, Any] = Field(default_factory=dict)
    events: tuple[ScenarioEvent, ...] = ()
    robots: tuple[RobotSpec, ...] = ()
    mission: MissionSpec | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Scenario:
        ids = [e.id for e in self.world_entities]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate world_entity id in scenario")
        known = set(ids)
        for e in self.world_entities:
            if e.parent_id is not None and e.parent_id not in known:
                raise ValueError(f"entity {e.id} has unknown parent {e.parent_id}")
        # ch2 Shared scenario consistency: a domain cannot attach state to an entity that does not exist.
        for section_name in ("structural_state", "ecological_state", "spatial_state"):
            section = getattr(self, section_name)
            for key in section.get("entities", {}):
                if UUID(str(key)) not in known:
                    raise ValueError(f"{section_name} references unknown world entity {key}")
        for ev in self.events:
            if ev.target_entity_id is not None and ev.target_entity_id not in known:
                raise ValueError(f"event {ev.event_id} targets unknown entity")
        return self
