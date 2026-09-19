"""Decision-plane configuration. Every threshold lives here (WORKSTREAM_BRIEF rule 11).

The defaults are ENGINEERING_ESTIMATE values for simulation; none of them is a validated mission
threshold (ch28 Acceptance Records: unfilled thresholds are NOT EVALUABLE, never PASS).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from conrad.schemas.base import ConradModel


class UncertaintyThresholds(ConradModel):
    aleatoric: float = Field(default=0.4, ge=0)
    epistemic: float = Field(default=0.4, ge=0)
    contradiction: float = Field(default=0.4, ge=0)
    observational: float = Field(default=0.4, ge=0)


class ConstraintConfig(ConradModel):
    engine_version: str = "constraint-engine-0.3"
    battery_reserve_fraction: float = Field(default=0.2, ge=0, le=1)
    max_robot_state_age_s: float = Field(default=2.0, gt=0)
    max_belief_age_s: float = Field(default=30.0, gt=0)
    risk_limit: float = Field(default=0.6, ge=0, le=1)
    max_pose_sigma_m: float | None = Field(default=None, gt=0)
    time_reserve_s: float = Field(
        default=120.0,
        ge=0,
        description="ENGINEERING_ESTIMATE: below this mission time remaining only retreat/hold actions are "
        "permitted; inactive when ResourceState.time_remaining_s is unknown (None)",
    )


class UtilityWeights(ConradModel):
    """Trade-offs over the consequence vector. Risk and uncertainty stay separate terms."""

    mission: float = 1.0
    information: float = 1.0
    time: float = 0.2
    energy: float = 0.2
    risk: float = 1.5
    communication: float = 0.1
    compute: float = 0.05


class DecisionConfig(ConradModel):
    model_version: str = "egdc-structured-0.2"
    thresholds: UncertaintyThresholds = UncertaintyThresholds()
    constraints: ConstraintConfig = ConstraintConfig()
    utility: UtilityWeights = UtilityWeights()
    max_claim_nodes: int = Field(default=128, gt=8)
    consequence_matters_above: float = Field(
        default=0.3, ge=0, le=1, description="below this consequence, uncertainty does not drive sensing"
    )
    escalate_consequence_above: float = Field(default=0.8, ge=0, le=1)
    uncalibrated_epistemic_floor: float = Field(
        default=0.45, ge=0, description="effective U_E assumed for uncalibrated beliefs on critical items"
    )
    min_spatial_coverage: float = Field(default=0.5, ge=0, le=1)
    max_information_attempts: int = Field(default=3, ge=1)
    enforce_grounding: bool = Field(
        default=True, description="False ONLY for the naive baseline arm of M1-UIR-E001"
    )
    navigation_position_tolerance_m: float = Field(default=0.5, gt=0)
    navigation_orientation_tolerance_rad: float = Field(default=0.35, gt=0)


def load_config(path: str | Path, model: type[ConradModel], section: str | None = None) -> Any:
    """Load a YAML file (optionally one top-level section) into a strict config model."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if section is not None:
        data = data.get(section, {})
    return model.model_validate(data)
