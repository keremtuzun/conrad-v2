"""TRUTH PLANE ONLY. Twin hidden state and supervision labels.

Importing this module from ``conrad.core``, ``conrad.domains``, ``conrad.decision``,
``conrad.active``, ``conrad.communication``, ``conrad.robotics`` or ``conrad.runtime`` is a CI
failure (tests/leakage/test_static_imports.py, INV-ARCH-01, CC-07). It is deliberately not
re-exported from ``conrad.schemas``.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from conrad.schemas.base import VersionedModel
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain

TRUTH_MARKER = "__conrad_truth__"


class TruthState(VersionedModel):
    scenario_id: UUID
    world_entity_id: UUID
    domain: Domain
    timestamp: TimeStamp
    state: dict[str, Any]
    relationships: tuple[UUID, ...] = ()
    event_context: dict[str, Any] = Field(default_factory=dict)
    ground_truth_quality: str = "SYNTHETIC_EXACT"
    generator_version: str


class SupervisionLabel(VersionedModel):
    """Training/evaluation target kept separate from inference inputs (ch28: truth identity is supervision only)."""

    observation_id: UUID | None = None
    evidence_id: UUID | None = None
    true_world_entity_id: UUID | None = None
    domain: Domain
    targets: dict[str, Any]
    target_masks: dict[str, bool] = Field(default_factory=dict)
    lineage: str = Field(description="split lineage key: scenario/world family/seed")
