"""OS-FM curriculum IDs and promotion records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OSFMStageID(str, Enum):
    U1_RGB = "U1-RGB"
    U1_SONAR = "U1-SONAR"
    U1_RANGE = "U1-RANGE"
    U1_GEOMETRY = "U1-GEOMETRY"
    M1 = "M1"
    T1 = "T1"
    J1 = "J1"


@dataclass(frozen=True)
class PromotionRecord:
    stage_id: OSFMStageID
    parent_checkpoint_id: str | None
    checkpoint_id: str
    run_id: str
    promoted: bool
    reason: str

