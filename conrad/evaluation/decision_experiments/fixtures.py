"""BELIEF-level synthetic fixtures (no perception, no truth). Everything here is SYNTHETIC_ONLY.

These builders produce schema-valid ``BeliefMessage`` / ``DecisionContext`` objects for the decision,
active and communication experiments and for unit tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from conrad.decision.context import DecisionContext, DecisionSummary, MissionRequirement
from conrad.schemas.belief import (
    Availability,
    BeliefMessage,
    BeliefSnapshot,
    KnowledgeStatus,
    Lifecycle,
    PropertyClaim,
    SpatialPayload,
    TechnicalPayload,
)
from conrad.schemas.comms import LinkState, LinkStatus
from conrad.schemas.decision import MissionPhase, MissionState, ResourceState
from conrad.schemas.frames import WORLD, Pose, SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import HealthLevel, RobotState, SystemHealth
from conrad.schemas.timebase import TimeStamp, stamp
from conrad.schemas.uncertainty import CalibrationMetadata, Uncertainty
from conrad.schemas.world import Domain, MissionSpec

CLOCK = "SIM"


def unc(
    ua: float = 0.05, ue: float = 0.05, uc: float = 0.0, uo: float = 0.05, calibrated: bool = True
) -> Uncertainty:
    return Uncertainty(
        aleatoric=ua,
        epistemic=ue,
        contradiction=uc,
        observational=uo,
        calibration_metadata=CalibrationMetadata(
            calibrated=calibrated, method="synthetic" if calibrated else None
        ),
    )


def region(center: tuple[float, float, float] = (5.0, 5.0, 0.0), half: float = 0.5) -> SpatialSupport:
    return SpatialSupport(frame_id=WORLD, center_m=center, half_extent_m=(half, half, half))


def make_belief(
    ids: IdFactory,
    *,
    domain: Domain = Domain.TECHNICAL,
    belief_id: UUID | None = None,
    revision: int = 0,
    time_s: float = 100.0,
    uncertainty: Uncertainty | None = None,
    status: KnowledgeStatus = KnowledgeStatus.OBSERVED,
    properties: dict[str, float | str | bool | None] | None = None,
    world_entity_id: UUID | None = None,
    n_evidence: int = 2,
    n_conflicts: int = 0,
    support: SpatialSupport | None = None,
    coverage: float = 0.9,
    lifecycle: Lifecycle = Lifecycle.ACTIVE,
    embedding_dim: int = 8,
    severity: float | None = None,
    relationships: Sequence[UUID] = (),
    publisher_availability: Availability = Availability.AVAILABLE,
) -> BeliefMessage:
    u = uncertainty or unc()
    prov = ids.new()
    props = {"condition": "NOMINAL"} if properties is None else properties
    claims = tuple(
        PropertyClaim(
            name=k,
            value=None if status is KnowledgeStatus.UNKNOWN else v,
            status=status,
            uncertainty=u,
            provenance_id=None if status is KnowledgeStatus.UNKNOWN else prov,
        )
        for k, v in props.items()
    )
    direct = status is KnowledgeStatus.OBSERVED
    bid = belief_id or ids.new()
    seed = (bid.int % 997) / 997.0
    return BeliefMessage(
        message_id=ids.new(),
        belief_id=bid,
        revision=revision,
        independent_observation_count=n_evidence if direct else 0,
        world_entity_id=world_entity_id,
        domain=domain,
        timestamp=stamp(time_s, CLOCK),
        state_summary=claims,
        state_embedding=tuple(round(seed + 0.01 * i + 0.1 * revision, 6) for i in range(embedding_dim)),
        knowledge_status=status,
        uncertainty=u,
        evidence_support=tuple(ids.new() for _ in range(n_evidence)) if direct else (),
        evidence_conflicts=tuple(ids.new() for _ in range(n_conflicts)),
        relationships=tuple(relationships),
        provenance_refs=(prov,),
        spatial_support=support or region(),
        lifecycle=lifecycle,
        model_version="synthetic-fixture-0.2",
        publisher_availability=publisher_availability,
        technical=TechnicalPayload(severity=severity, direct_support=1.0 if direct else 0.0)
        if domain is Domain.TECHNICAL
        else None,
        spatial=SpatialPayload(coverage=coverage, observation_count=n_evidence)
        if domain is Domain.SPATIAL
        else None,
    )


def make_snapshot(
    ids: IdFactory,
    messages: Sequence[BeliefMessage],
    now_s: float,
    availability: dict[Domain, Availability] | None = None,
    stale_ids: Sequence[UUID] = (),
) -> BeliefSnapshot:
    avail = {d.value: Availability.AVAILABLE for d in Domain}
    for d, a in (availability or {}).items():
        avail[d.value] = a
    return BeliefSnapshot(
        snapshot_id=ids.new(),
        created_time_ns=stamp(now_s, CLOCK).time_ns,
        messages=tuple(messages),
        domain_availability=avail,
        provenance={
            "revisions": {str(m.belief_id): m.revision for m in messages},
            "stale_belief_ids": [str(s) for s in stale_ids],
        },
    )


def make_requirement(
    ids: IdFactory,
    *,
    belief_ids: Sequence[UUID] = (),
    entity_ids: Sequence[UUID] = (),
    consequence: float = 0.9,
    domain: Domain = Domain.TECHNICAL,
    properties: Sequence[str] = ("condition",),
    context_domains: Sequence[Domain] = (),
    target_region: SpatialSupport | None = None,
    max_age_s: float | None = None,
) -> MissionRequirement:
    return MissionRequirement(
        requirement_id=ids.new(),
        description="determine structural condition of target",
        domain=domain,
        target_belief_ids=tuple(belief_ids),
        target_entity_ids=tuple(entity_ids),
        region=target_region or region(),
        properties=tuple(properties),
        consequence=consequence,
        context_domains=tuple(context_domains),
        max_age_s=max_age_s,
    )


def make_context(
    ids: IdFactory,
    messages: Sequence[BeliefMessage],
    requirements: Sequence[MissionRequirement],
    *,
    now_s: float = 101.0,
    battery: float | None = 0.8,
    health: HealthLevel = HealthLevel.OK,
    motion_permitted: bool = True,
    link_status: LinkStatus = LinkStatus.UP,
    availability: dict[Domain, Availability] | None = None,
    modalities: Sequence[str] = ("RGB", "SONAR"),
    previous: Sequence[DecisionSummary] = (),
    notes: dict[str, Any] | None = None,
    robot_state_age_s: float = 0.1,
    boundary: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    operator_reachable: bool | None = True,
    stale_ids: Sequence[UUID] = (),
) -> DecisionContext:
    now: TimeStamp = stamp(now_s, CLOCK)
    mission_id = ids.new()
    return DecisionContext(
        timestamp=now,
        trace_id=ids.new(),
        mission=MissionState(
            mission_id=mission_id,
            phase=MissionPhase.INSPECTING,
            timestamp=now,
            objectives_total=len(requirements),
            objectives_done=0,
            notes=notes or {},
        ),
        mission_spec=None
        if boundary is None
        else MissionSpec(
            mission_id=mission_id,
            mission_type="INSPECTION",
            boundary_min_m=boundary[0],
            boundary_max_m=boundary[1],
        ),
        requirements=tuple(requirements),
        snapshot=make_snapshot(ids, messages, now_s, availability, stale_ids),
        robot_state=RobotState(
            timestamp=stamp(now_s - robot_state_age_s, CLOCK),
            pose=Pose(frame_id=WORLD, position_m=(0.0, 0.0, 0.0)),
            estimator_health=HealthLevel.OK,
            estimator_name="synthetic",
        ),
        resource_state=ResourceState(timestamp=now, battery_fraction=battery),
        link_state=LinkState(
            link_name="acoustic",
            timestamp=now,
            status=link_status,
            bandwidth_bps=0.0 if link_status is LinkStatus.DOWN else 2000.0,
            latency_s=1.0,
            packet_loss=0.05,
            bit_error_rate=1e-5,
        ),
        system_health=SystemHealth(timestamp=now, overall=health),
        previous_decisions=tuple(previous),
        available_modalities=tuple(modalities),
        motion_permitted=motion_permitted,
        operator_reachable=operator_reachable,
    )
