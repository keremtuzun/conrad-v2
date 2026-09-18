"""Deterministic hard constraint layer (ch16/ch17 'Hard constraint layer', ch33 EGDC).

proposal != permission. This engine is plain code with reason codes. It has no learnable parameter,
it is not part of any loss and no policy score can change its verdict.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from conrad.decision.claims import ClaimGraph
from conrad.decision.config import DecisionConfig
from conrad.decision.consequence import ConsequenceVector
from conrad.decision.context import DecisionContext
from conrad.schemas.belief import Availability
from conrad.schemas.comms import LinkStatus
from conrad.schemas.decision import (
    WORLD_DEPENDENT_CLAIMS,
    ActionProposal,
    ActionType,
    ConstraintDecision,
    GroundingStatus,
    QuestionType,
)
from conrad.schemas.robot import HealthLevel
from conrad.schemas.timebase import NS_PER_S
from conrad.schemas.world import Domain

R_UNKNOWN_ACTION = "UNKNOWN_ACTION_TYPE"
R_INVALID_ARGUMENT = "INVALID_ARGUMENT"
R_MISSING_TARGET = "MISSING_TARGET"
R_UNKNOWN_CLAIM = "UNKNOWN_SUPPORTING_CLAIM"
R_UNSUPPORTED_AS_FACT = "UNSUPPORTED_CLAIM_AS_FACT"
R_STALE_BELIEF = "STALE_BELIEF"
R_STALE_ROBOT_STATE = "STALE_ROBOT_STATE"
R_NO_ROBOT_STATE = "ROBOT_STATE_UNAVAILABLE"
R_ESTIMATOR_FAULT = "ESTIMATOR_FAULT"
R_POSE_UNCERTAIN = "POSE_UNCERTAINTY_EXCEEDED"
R_HEALTH_FAULT = "SYSTEM_HEALTH_FAULT"
R_LEAK = "LEAK_DETECTED"
R_BATTERY = "BATTERY_BELOW_RESERVE"
R_ENERGY = "INSUFFICIENT_ENERGY"
R_RISK = "RISK_LIMIT_EXCEEDED"
R_MOTION = "MOTION_NOT_PERMITTED"
R_BOUNDARY = "OUTSIDE_MISSION_BOUNDARY"
R_LINK_DOWN = "LINK_DOWN"
R_DOMAIN_UNAVAILABLE = "DOMAIN_UNAVAILABLE"
R_MISSION_INACTIVE = "MISSION_NOT_ACTIVE"

MOTION_ACTIONS = frozenset(
    {
        ActionType.CONTINUE_MISSION,
        ActionType.REQUEST_INFORMATION,
        ActionType.REVISIT_REGION,
        ActionType.REPLAN,
    }
)
ALWAYS_PERMITTED = frozenset({ActionType.WAIT, ActionType.ESCALATE_TO_OPERATOR, ActionType.STORE_AND_FORWARD})
_PARAM_TYPES: dict[ActionType, dict[str, type | tuple[type, ...]]] = {
    ActionType.WAIT: {"duration_s": (int, float)},
    ActionType.QUERY_BELIEF: {"domain": str},
    ActionType.REQUEST_INFORMATION: {"question_type": str, "cause": str},
}


class ConstraintEngine:
    def __init__(self, config: DecisionConfig) -> None:
        self.config = config
        self.c = config.constraints

    def check(
        self,
        action: ActionProposal,
        graph: ClaimGraph,
        ctx: DecisionContext,
        consequence: ConsequenceVector | None = None,
    ) -> ConstraintDecision:
        reasons: list[str] = []
        kind: Any = action.action_type
        if not isinstance(kind, ActionType):
            return self._decision(action, [R_UNKNOWN_ACTION])
        reasons += self._arguments(action)
        reasons += self._claims(action, graph, ctx)
        if kind not in ALWAYS_PERMITTED:
            reasons += self._health_power(action, ctx)
        if kind in MOTION_ACTIONS:
            reasons += self._motion(action, ctx)
        if kind is ActionType.TRANSMIT_INFORMATION and (
            ctx.link_state is None or ctx.link_state.status is LinkStatus.DOWN
        ):
            reasons.append(R_LINK_DOWN)
        if kind is ActionType.QUERY_BELIEF:
            domain = action.parameters.get("domain")
            if (
                isinstance(domain, str)
                and domain in Domain.__members__
                and ctx.availability(Domain(domain)) is Availability.UNAVAILABLE
            ):
                reasons.append(R_DOMAIN_UNAVAILABLE)
        if consequence is not None and consequence.risk > self.c.risk_limit:
            reasons.append(R_RISK)
        return self._decision(action, reasons)

    # ------------------------------------------------------------------ groups
    def _arguments(self, action: ActionProposal) -> list[str]:
        out: list[str] = []
        for key, expected in _PARAM_TYPES.get(action.action_type, {}).items():
            value = action.parameters.get(key)
            if value is None or not isinstance(value, expected) or isinstance(value, bool):
                out.append(f"{R_INVALID_ARGUMENT}:{key}")
        kind = action.action_type
        if kind is ActionType.WAIT:
            d = action.parameters.get("duration_s")
            if isinstance(d, int | float) and not isinstance(d, bool) and not (0 < d <= 3600):
                out.append(f"{R_INVALID_ARGUMENT}:duration_s")
        if kind is ActionType.REQUEST_INFORMATION:
            if not action.target_belief_ids:
                out.append(R_MISSING_TARGET)
            q = action.parameters.get("question_type")
            if isinstance(q, str) and q not in QuestionType.__members__:
                out.append(f"{R_INVALID_ARGUMENT}:question_type")
        if kind is ActionType.REVISIT_REGION and action.target_region is None:
            out.append(R_MISSING_TARGET)
        if kind is ActionType.QUERY_BELIEF:
            d = action.parameters.get("domain")
            if isinstance(d, str) and d not in Domain.__members__:
                out.append(f"{R_INVALID_ARGUMENT}:domain")
        if (
            kind in (ActionType.TRANSMIT_INFORMATION, ActionType.STORE_AND_FORWARD)
            and not action.target_belief_ids
        ):
            out.append(R_MISSING_TARGET)
        return out

    def _claims(self, action: ActionProposal, graph: ClaimGraph, ctx: DecisionContext) -> list[str]:
        """validate_claim_support (ch28): an unsupported world claim can never be relied on as fact."""
        out: list[str] = []
        if not self.config.enforce_grounding:
            return out  # naive baseline arm of M1-UIR-E001 only; never a runtime configuration
        for claim_id in action.supporting_claims:
            claim = graph.get(claim_id)
            if claim is None:
                out.append(R_UNKNOWN_CLAIM)
                continue
            if claim.claim_type not in WORLD_DEPENDENT_CLAIMS:
                continue
            if claim.grounding is not GroundingStatus.GROUNDED:
                out.append(R_UNSUPPORTED_AS_FACT)
            age_s = (ctx.timestamp.time_ns - claim.timestamp.time_ns) / NS_PER_S
            relies_on_freshness = action.action_type is ActionType.CONTINUE_MISSION
            if relies_on_freshness and (
                age_s > self.c.max_belief_age_s or claim.structured_value.get("stale")
            ):
                out.append(R_STALE_BELIEF)
        return sorted(set(out))

    def _health_power(self, action: ActionProposal, ctx: DecisionContext) -> list[str]:
        out: list[str] = []
        retreat = action.action_type in (ActionType.RETURN_TO_SAFE_STATE, ActionType.ABORT_MISSION)
        health = ctx.system_health
        if health is not None and not retreat:
            if health.overall is HealthLevel.FAULT:
                out.append(R_HEALTH_FAULT)
            if health.leak_detected:
                out.append(R_LEAK)
        battery = None if ctx.resource_state is None else ctx.resource_state.battery_fraction
        if battery is not None and battery < self.c.battery_reserve_fraction and not retreat:
            out.append(R_BATTERY)
        needed = action.parameters.get("estimated_energy_j")
        remaining = None if ctx.resource_state is None else ctx.resource_state.energy_remaining_j
        if isinstance(needed, int | float) and remaining is not None and needed > remaining:
            out.append(R_ENERGY)
        return out

    def _motion(self, action: ActionProposal, ctx: DecisionContext) -> list[str]:
        out: list[str] = []
        if not ctx.motion_permitted:
            out.append(R_MOTION)
        state = ctx.robot_state
        if state is None:
            out.append(R_NO_ROBOT_STATE)
        else:
            if state.estimator_health is HealthLevel.FAULT:
                out.append(R_ESTIMATOR_FAULT)
            age_s = (ctx.timestamp.time_ns - state.timestamp.time_ns) / NS_PER_S
            if age_s > self.c.max_robot_state_age_s:
                out.append(R_STALE_ROBOT_STATE)
            sigma = state.pose.position_sigma_m()
            if self.c.max_pose_sigma_m is not None and sigma is not None and sigma > self.c.max_pose_sigma_m:
                out.append(R_POSE_UNCERTAIN)
        region = action.target_region
        spec = ctx.mission_spec
        if region is not None and spec is not None and spec.boundary_min_m and spec.boundary_max_m:
            for i in range(3):
                lo = region.center_m[i] - region.half_extent_m[i]
                hi = region.center_m[i] + region.half_extent_m[i]
                if lo < spec.boundary_min_m[i] - 1e-9 or hi > spec.boundary_max_m[i] + 1e-9:
                    out.append(R_BOUNDARY)
                    break
        return out

    def _decision(self, action: ActionProposal, reasons: list[str]) -> ConstraintDecision:
        return ConstraintDecision(
            action_id=action.action_id,
            accepted=not reasons,
            reason_codes=tuple(reasons),
            engine_version=self.c.engine_version,
        )


def validate_claim_support(action: ActionProposal, graph: ClaimGraph) -> tuple[UUID, ...]:
    """IDs of supporting claims that are world-dependent but not GROUNDED (must be empty to execute)."""
    bad = []
    for claim_id in action.supporting_claims:
        claim = graph.get(claim_id)
        if claim is None or (
            claim.claim_type in WORLD_DEPENDENT_CLAIMS and claim.grounding is not GroundingStatus.GROUNDED
        ):
            bad.append(claim_id)
    return tuple(bad)
