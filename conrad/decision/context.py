"""Model 1 input contract (ch16 'Model 1 input contract', ch17 'Model 1's working state').

``DecisionContext`` is the complete working reality of the decision plane. It contains beliefs,
estimated robot state and mission context. It never contains twin truth.

implementation_status: FROZEN_CONTRACT (I/O boundary)
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from conrad.schemas.base import ConradModel
from conrad.schemas.belief import Availability, BeliefMessage, BeliefSnapshot
from conrad.schemas.comms import LinkState
from conrad.schemas.decision import ActionType, InformationNeed, MissionState, ResourceState
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.robot import RobotState, SystemHealth
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain, MissionSpec


class MissionRequirement(ConradModel):
    """One thing the mission needs to know (Q_t). ``consequence`` is RISK, not uncertainty (ch17)."""

    requirement_id: UUID
    description: str
    domain: Domain
    target_belief_ids: tuple[UUID, ...] = ()
    target_entity_ids: tuple[UUID, ...] = Field(
        default=(), description="asset-registry identities from mission context (never simulator IDs)"
    )
    region: SpatialSupport | None = None
    properties: tuple[str, ...] = ()
    consequence: float = Field(ge=0, le=1, description="cost of acting on a wrong belief; 1 = catastrophic")
    context_domains: tuple[Domain, ...] = Field(
        default=(), description="other domains whose beliefs in the same region matter (e.g. 2S coverage)"
    )
    max_age_s: float | None = Field(default=None, gt=0)
    deadline_ns: int | None = None


class DecisionSummary(ConradModel):
    """Compact decision history H_t."""

    decision_id: UUID
    time_ns: int
    action_type: ActionType | None
    target_belief_ids: tuple[UUID, ...] = ()
    abstained: bool = False
    outcome_ok: bool | None = None
    executed: bool | None = Field(
        default=None,
        description="False when the runtime did not carry the chosen action out (deferred or dropped by "
        "routing); None = not reported, counted as carried out",
    )
    target_revisions: tuple[int, ...] = Field(
        default=(),
        description="revision of each target belief in the decision's snapshot (aligned with "
        "target_belief_ids); empty = not recorded",
    )


class DecisionContext(ConradModel):
    timestamp: TimeStamp
    trace_id: UUID
    mission: MissionState
    mission_spec: MissionSpec | None = None
    requirements: tuple[MissionRequirement, ...]
    snapshot: BeliefSnapshot
    robot_state: RobotState | None = None
    resource_state: ResourceState | None = None
    link_state: LinkState | None = None
    system_health: SystemHealth | None = None
    previous_decisions: tuple[DecisionSummary, ...] = ()
    active_information_needs: tuple[InformationNeed, ...] = ()
    available_modalities: tuple[str, ...] = Field(
        default=(), description="sensing modalities currently usable (from capabilities/health)"
    )
    motion_permitted: bool = Field(
        default=True, description="Safety Supervisor's current permission; Model 1 never sets this"
    )
    operator_reachable: bool | None = None

    def beliefs(self, domain: Domain | None = None) -> tuple[BeliefMessage, ...]:
        if domain is None:
            return self.snapshot.messages
        return tuple(m for m in self.snapshot.messages if m.domain is domain)

    def availability(self, domain: Domain) -> Availability:
        return self.snapshot.domain_availability.get(domain.value, Availability.UNAVAILABLE)

    def attempts_on(self, belief_ids: tuple[UUID, ...], action_type: ActionType) -> int:
        """How often the recent history already tried ``action_type`` on these beliefs.

        Only actions the runtime carried out count: a choice that routing deferred or dropped
        (``executed=False``) was never an attempt, so it must not use up the attempt budget.
        """
        wanted = set(belief_ids)
        return sum(
            1
            for d in self.previous_decisions
            if d.action_type is action_type
            and d.executed is not False
            and wanted.intersection(d.target_belief_ids)
        )
