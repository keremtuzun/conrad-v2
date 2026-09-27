"""P9 host-neutral inspection intent boundary."""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import Field, model_validator

from conrad.schemas.base import VersionedModel
from conrad.schemas.decision import ObservationPlan, PlanStatus


class HostCapability(str, Enum):
    CHANGE_SENSOR_MODE = "CHANGE_SENSOR_MODE"
    HOLD_POSITION = "HOLD_POSITION"
    REVISIT_REGION = "REVISIT_REGION"
    STORE_AND_FORWARD = "STORE_AND_FORWARD"
    OPERATOR_ESCALATION = "OPERATOR_ESCALATION"


class HostCapabilityProfile(VersionedModel):
    host_id: str = Field(min_length=1)
    capabilities: tuple[HostCapability, ...]
    degraded: bool = False
    degradation_reasons: tuple[str, ...] = ()


class InspectionIntent(VersionedModel):
    intent_id: UUID
    trace_id: UUID
    source_plan_id: UUID
    requested_capability: HostCapability
    parameters: dict[str, object] = Field(default_factory=dict)
    safety_owner: str = "third-party-host"
    low_level_control_included: bool = False

    @model_validator(mode="after")
    def _host_keeps_control(self) -> InspectionIntent:
        if self.low_level_control_included:
            raise ValueError("OceanSense InspectionIntent must not include low-level control")
        if self.safety_owner != "third-party-host":
            raise ValueError("third-party host owns low-level safety and execution authority")
        return self


class GatewayDecision(VersionedModel):
    accepted: bool
    intent: InspectionIntent | None
    reason_codes: tuple[str, ...]


class HostIntentGateway:
    version = "p9-host-intent-gateway-v0"

    def negotiate(self, requested: tuple[HostCapability, ...], profile: HostCapabilityProfile) -> tuple[HostCapability, ...]:
        available = set(profile.capabilities)
        return tuple(cap for cap in requested if cap in available)

    def from_plan(self, plan: ObservationPlan, profile: HostCapabilityProfile, intent_id: UUID) -> GatewayDecision:
        if plan.status is not PlanStatus.PLAN or plan.primary_action is None:
            return GatewayDecision(accepted=False, intent=None, reason_codes=("NO_PLAN_ACTION",))
        requested = HostCapability.REVISIT_REGION
        if requested not in profile.capabilities:
            fallback = HostCapability.OPERATOR_ESCALATION if HostCapability.OPERATOR_ESCALATION in profile.capabilities else None
            if fallback is None:
                return GatewayDecision(accepted=False, intent=None, reason_codes=("HOST_CAPABILITY_UNAVAILABLE",))
            requested = fallback
        reasons = ("HOST_DEGRADED", *profile.degradation_reasons) if profile.degraded else ()
        intent = InspectionIntent(
            intent_id=intent_id,
            trace_id=plan.trace_id,
            source_plan_id=plan.plan_id,
            requested_capability=requested,
            parameters={
                "target_region": plan.primary_action.target_region.model_dump(mode="json"),
                "sensor_id": str(plan.primary_action.sensor_id),
                "duration_s": plan.primary_action.duration_s,
            },
        )
        return GatewayDecision(accepted=True, intent=intent, reason_codes=reasons)
