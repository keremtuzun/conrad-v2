"""Decision-plane contracts: mission, claims, actions, information needs, plans, navigation (ch16-18, ch20).

Proposal != permission. Model1 proposes inside a constrained vocabulary; the deterministic
constraint engine accepts or rejects; navigation executes; MCBR never moves the robot.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from conrad.schemas.base import VersionedModel
from conrad.schemas.frames import Pose, SpatialSupport
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.uncertainty import Uncertainty


class MissionPhase(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    TRANSIT = "TRANSIT"
    INSPECTING = "INSPECTING"
    RESOLVING = "RESOLVING"
    RETURNING = "RETURNING"
    HOLDING = "HOLDING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


class MissionState(VersionedModel):
    mission_id: UUID
    phase: MissionPhase
    timestamp: TimeStamp
    objectives_total: int = Field(ge=0)
    objectives_done: int = Field(ge=0)
    active_need_ids: tuple[UUID, ...] = ()
    notes: dict[str, Any] = Field(default_factory=dict)


class ResourceState(VersionedModel):
    timestamp: TimeStamp
    battery_fraction: float | None = Field(default=None, ge=0, le=1)
    energy_remaining_j: float | None = Field(default=None, ge=0)
    time_remaining_s: float | None = None
    storage_free_bytes: int | None = Field(default=None, ge=0)
    compute_load: float | None = Field(default=None, ge=0)


class ClaimType(str, Enum):
    BELIEF_CLAIM = "BELIEF_CLAIM"
    MISSION_REQUIREMENT = "MISSION_REQUIREMENT"
    RESOURCE_STATE = "RESOURCE_STATE"
    ROBOT_STATE = "ROBOT_STATE"
    CONSTRAINT = "CONSTRAINT"
    INFORMATION_NEED = "INFORMATION_NEED"
    CANDIDATE_ACTION = "CANDIDATE_ACTION"
    EXPECTED_OUTCOME = "EXPECTED_OUTCOME"


WORLD_DEPENDENT_CLAIMS = frozenset({ClaimType.BELIEF_CLAIM})


class ClaimEdgeType(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    REQUIRES = "REQUIRES"
    BLOCKS = "BLOCKS"
    AFFECTS = "AFFECTS"
    RESOLVES = "RESOLVES"
    DEPENDS_ON = "DEPENDS_ON"


class GroundingStatus(str, Enum):
    GROUNDED = "GROUNDED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_WORLD_DEPENDENT = "NOT_WORLD_DEPENDENT"


class DecisionClaim(VersionedModel):
    claim_id: UUID
    claim_type: ClaimType
    statement: str
    structured_value: dict[str, Any] = Field(default_factory=dict)
    content_embedding: tuple[float, ...] = ()
    source_belief_ids: tuple[UUID, ...] = ()
    source_belief_revisions: tuple[int, ...] = ()
    evidence_refs: tuple[UUID, ...] = ()
    uncertainty: Uncertainty | None = None
    timestamp: TimeStamp
    grounding: GroundingStatus

    @model_validator(mode="after")
    def _grounding_is_mechanical(self) -> DecisionClaim:
        if self.claim_type in WORLD_DEPENDENT_CLAIMS:
            has_path = bool(self.source_belief_ids)
            if self.grounding is GroundingStatus.GROUNDED and not has_path:
                raise ValueError("world-dependent claim marked GROUNDED without a belief path")
            if self.grounding is GroundingStatus.NOT_WORLD_DEPENDENT:
                raise ValueError("BELIEF_CLAIM cannot be NOT_WORLD_DEPENDENT")
        return self


class ClaimEdge(VersionedModel):
    source_claim_id: UUID
    target_claim_id: UUID
    edge_type: ClaimEdgeType


class ActionType(str, Enum):
    CONTINUE_MISSION = "CONTINUE_MISSION"
    QUERY_BELIEF = "QUERY_BELIEF"
    REQUEST_INFORMATION = "REQUEST_INFORMATION"
    REPLAN = "REPLAN"
    CHANGE_SENSOR_MODE = "CHANGE_SENSOR_MODE"
    REVISIT_REGION = "REVISIT_REGION"
    WAIT = "WAIT"
    TRANSMIT_INFORMATION = "TRANSMIT_INFORMATION"
    STORE_AND_FORWARD = "STORE_AND_FORWARD"
    ESCALATE_TO_OPERATOR = "ESCALATE_TO_OPERATOR"
    RETURN_TO_SAFE_STATE = "RETURN_TO_SAFE_STATE"
    ABORT_MISSION = "ABORT_MISSION"


class ActionProposal(VersionedModel):
    action_id: UUID
    action_type: ActionType
    target_belief_ids: tuple[UUID, ...] = ()
    target_region: SpatialSupport | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    supporting_claims: tuple[UUID, ...] = ()
    expected_outcome: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    priority: float = Field(default=0.5, ge=0, le=1)


class ConstraintDecision(VersionedModel):
    action_id: UUID
    accepted: bool
    reason_codes: tuple[str, ...] = ()
    engine_version: str


class QuestionType(str, Enum):
    CONFIRM_CONDITION = "CONFIRM_CONDITION"
    RESOLVE_CONTRADICTION = "RESOLVE_CONTRADICTION"
    EXTEND_COVERAGE = "EXTEND_COVERAGE"
    DISCRIMINATE_HYPOTHESES = "DISCRIMINATE_HYPOTHESES"
    IMPROVE_MEASUREMENT = "IMPROVE_MEASUREMENT"


class UncertaintyType(str, Enum):
    ALEATORIC = "ALEATORIC"
    EPISTEMIC = "EPISTEMIC"
    CONTRADICTION = "CONTRADICTION"
    OBSERVATIONAL = "OBSERVATIONAL"


class InformationNeed(VersionedModel):
    need_id: UUID
    trace_id: UUID
    target_belief_ids: tuple[UUID, ...] = Field(min_length=1)
    question_type: QuestionType
    target_properties: tuple[str, ...]
    priority: float = Field(ge=0, le=1)
    desired_uncertainty_reduction: dict[str, float] = Field(default_factory=dict)
    deadline_ns: int | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    originating_claim_ids: tuple[UUID, ...] = ()


class ResourceCost(VersionedModel):
    time_s: float = Field(ge=0)
    energy_j: float = Field(ge=0)
    risk: float = Field(ge=0, le=1)
    travel_m: float = Field(ge=0)


class ObservationAction(VersionedModel):
    action_id: UUID
    pose: Pose
    sensor_id: UUID
    sensor_configuration: dict[str, Any] = Field(default_factory=dict)
    target_region: SpatialSupport
    duration_s: float = Field(gt=0)
    expected_cost: ResourceCost
    predicted_visibility: float = Field(ge=0, le=1)
    expected_information_gain: float
    hypothesis_discrimination: float = Field(default=0.0, ge=0)


class RejectedCandidate(VersionedModel):
    action: ObservationAction
    reason_codes: tuple[str, ...] = Field(min_length=1)


class PlanStatus(str, Enum):
    PLAN = "PLAN"
    NEED_SATISFIED = "NEED_SATISFIED"
    NO_FEASIBLE_OBSERVATION = "NO_FEASIBLE_OBSERVATION"
    NOT_WORTH_COST = "NOT_WORTH_COST"


class ObservationPlan(VersionedModel):
    plan_id: UUID
    need_id: UUID
    trace_id: UUID
    status: PlanStatus
    target_beliefs: tuple[UUID, ...]
    primary_action: ObservationAction | None
    alternatives: tuple[ObservationAction, ...] = ()
    rejected: tuple[RejectedCandidate, ...] = ()
    expected_information_gain: float = 0.0
    expected_mission_gain: float = 0.0
    targeted_uncertainty: tuple[UncertaintyType, ...] = ()
    expected_cost: ResourceCost | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    provenance: UUID

    @model_validator(mode="after")
    def _plan_has_action(self) -> ObservationPlan:
        if (self.status is PlanStatus.PLAN) != (self.primary_action is not None):
            raise ValueError("primary_action must be present exactly when status is PLAN")
        return self


class NavigationGoal(VersionedModel):
    goal_id: UUID
    trace_id: UUID
    target_pose: Pose | None = None
    target_region: SpatialSupport | None = None
    position_tolerance_m: float = Field(gt=0)
    orientation_tolerance_rad: float = Field(gt=0)
    observation_constraints: dict[str, Any] = Field(default_factory=dict)
    deadline_ns: int | None = None
    risk_limit: float = Field(ge=0, le=1)
    energy_budget_j: float | None = Field(default=None, gt=0)
    source_plan_id: UUID | None = None
    source_action_id: UUID | None = None

    @model_validator(mode="after")
    def _has_target(self) -> NavigationGoal:
        if self.target_pose is None and self.target_region is None:
            raise ValueError("navigation goal needs a target pose or region")
        return self


class TrajectoryPoint(VersionedModel):
    t_s: float = Field(ge=0)
    pose: Pose
    linear_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)


class Trajectory(VersionedModel):
    trajectory_id: UUID
    goal_id: UUID
    trace_id: UUID
    frame_id: str
    points: tuple[TrajectoryPoint, ...] = Field(min_length=1)
    max_speed_mps: float = Field(gt=0)
    planner: str


class DecisionRecord(VersionedModel):
    decision_id: UUID
    trace_id: UUID
    mission_id: UUID
    timestamp: TimeStamp
    belief_snapshot_id: UUID
    claims: tuple[DecisionClaim, ...]
    edges: tuple[ClaimEdge, ...] = ()
    candidates: tuple[ActionProposal, ...]
    chosen: ActionProposal | None
    constraint_decisions: tuple[ConstraintDecision, ...]
    unsupported_claim_ids: tuple[UUID, ...] = ()
    abstained: bool = False
    rationale: str
    provenance_id: UUID
    model_version: str
