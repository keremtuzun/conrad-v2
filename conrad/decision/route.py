"""Route-blocking beliefs: when a grounded obstacle belief sits on the planned route, Model 1 REPLANs.

(ch17 'Replanning' L7740-7746: plans are hypotheses; new beliefs -> Plan_{t+1} = Replan(Plan_t, B_{t+1}, ...).)
Model 1 does not plan the path (Navigation does, ch16 'Decision horizons'); it only notices that a belief it
can ground contradicts the current plan and asks the mission executive for a new one.

The planned route is mission context: ``MissionState.notes["planned_route"]`` is a list of ``SpatialSupport``
dicts (the corridor legs still ahead). No route in the context -> nothing can be blocked.

implementation_status: EXPERIMENTAL_CANDIDATE (template rule, deterministic)
"""

from __future__ import annotations

from typing import Any

from conrad.decision.config import DecisionConfig
from conrad.decision.context import DecisionContext
from conrad.schemas.belief import BeliefMessage, KnowledgeStatus
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.timebase import NS_PER_S
from conrad.schemas.world import Domain

ROUTE_NOTE = "planned_route"
OCCUPANCY_PROPERTY = "occupied"


def planned_route(ctx: DecisionContext) -> list[SpatialSupport]:
    raw: Any = ctx.mission.notes.get(ROUTE_NOTE) or []
    return [s if isinstance(s, SpatialSupport) else SpatialSupport.model_validate(s) for s in raw]


def overlaps(a: SpatialSupport, b: SpatialSupport) -> bool:
    """Axis-aligned box intersection in a shared frame (different frames never overlap: no silent transform)."""
    if a.frame_id != b.frame_id:
        return False
    return all(
        abs(a.center_m[i] - b.center_m[i]) <= a.half_extent_m[i] + b.half_extent_m[i] + 1e-9 for i in range(3)
    )


def route_blockers(ctx: DecisionContext, config: DecisionConfig) -> list[BeliefMessage]:
    """Fresh, evidence-backed, well-observed SPATIAL beliefs that claim occupancy on a planned route leg.

    UNKNOWN is never occupied (brief rule 6), an uncertain obstacle (U_O over threshold) is a coverage question,
    not a route fact, and a belief without an evidence path cannot justify a replan.
    """
    route = planned_route(ctx)
    if not route:
        return []
    stale_ids = set(ctx.snapshot.provenance.get("stale_belief_ids", []))
    out = []
    for m in ctx.beliefs(Domain.SPATIAL):
        if m.knowledge_status not in (KnowledgeStatus.OBSERVED, KnowledgeStatus.INFERRED):
            continue
        prop = next((c for c in m.state_summary if c.name == OCCUPANCY_PROPERTY), None)
        if prop is None or prop.value is not True or prop.status is KnowledgeStatus.UNKNOWN:
            continue
        if not m.evidence_support or m.evidence_conflicts:
            continue
        age_s = (ctx.timestamp.time_ns - m.timestamp.time_ns) / NS_PER_S
        if age_s > config.constraints.max_belief_age_s or str(m.belief_id) in stale_ids:
            continue
        if prop.uncertainty.observational >= config.thresholds.observational:
            continue
        if m.spatial_support is not None and any(overlaps(m.spatial_support, leg) for leg in route):
            out.append(m)
    return out
