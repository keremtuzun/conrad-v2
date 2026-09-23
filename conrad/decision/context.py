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
    answered_information_requests: dict[UUID, int] = Field(
        default_factory=dict,
        description="per belief, the NEWEST revision that an information request the runtime actually "
        "carried out saw, over the whole mission. previous_decisions is the short history window H_t and is "
        "the right basis for counting RECENT attempts; that an independent look was requested and taken is a "
        "fact about the mission which does not expire with the window, so it is reported separately. Empty = "
        "not reported (then only the window is available).",
    )
    active_information_needs: tuple[InformationNeed, ...] = ()
    unavailable_information_targets: dict[UUID, str] = Field(
        default_factory=dict,
        description="Runtime-reported terminal acquisition status by belief target. This reports planner/"
        "execution availability only; it never reports world truth or whether the requirement is satisfied.",
    )
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

    def information_in_flight(self, belief_ids: tuple[UUID, ...]) -> bool:
        """Whether an accepted, active information need is acquiring any requested belief.

        The runtime supplies only needs attached to its currently active acquisition goal. This is
        execution state, not a claim that the acquisition succeeded, and therefore only postpones the
        attempts-exhausted fallback while that goal remains active.
        """
        wanted = set(belief_ids)
        return bool(wanted) and any(
            wanted.intersection(need.target_belief_ids) for need in self.active_information_needs
        )

    def information_unavailable(self, belief_ids: tuple[UUID, ...]) -> bool:
        """Whether the runtime explicitly exhausted acquisition for any requested belief."""
        return any(belief_id in self.unavailable_information_targets for belief_id in belief_ids)

    def request_answered(self, belief_id: UUID, revision: int) -> bool:
        """Has an executed information request on this belief been answered by a newer revision?

        True when some carried-out ``REQUEST_INFORMATION`` in the history window saw an older revision of the
        belief, or when the runtime's window-independent ledger
        (``answered_information_requests``) records one. The ledger keeps the NEWEST such revision, so a
        request that has not been answered yet does not report as answered, and an answer already given does
        not expire when the request scrolls out of H_t.
        """
        for d in self.previous_decisions:
            if d.action_type is not ActionType.REQUEST_INFORMATION or d.executed is False:
                continue
            seen = dict(zip(d.target_belief_ids, d.target_revisions, strict=False))
            if belief_id in seen and seen[belief_id] < revision:
                return True
        ledger = self.answered_information_requests.get(belief_id)
        return ledger is not None and ledger < revision
