"""P8 inspection-intelligence helpers around InformationNeed and MCBR contracts."""

from __future__ import annotations

from conrad.active.planner import MCBRPlanner, PlanningRequest
from conrad.schemas.decision import InformationNeed, ObservationPlan, UncertaintyType


def value_of_information(need: InformationNeed) -> float:
    """Contract hook for uncertainty-reduction planning.

    It is deterministic and schema-level: useful for integration tests, not a
    final scientific superiority claim.
    """

    weighted = {
        "ALEATORIC": 0.5,
        "EPISTEMIC": 1.0,
        "CONTRADICTION": 1.25,
        "OBSERVATIONAL": 0.8,
    }
    total = 0.0
    for key, value in need.desired_uncertainty_reduction.items():
        total += weighted.get(key.upper(), 1.0) * max(0.0, value)
    return min(1.0, total * need.priority)


class AdaptiveInspectionPlanner:
    """Thin contract wrapper over EGDC/MCBR-era planning surfaces."""

    version = "p8-adaptive-inspection-v0"

    def __init__(self, mcbr: MCBRPlanner) -> None:
        self.mcbr = mcbr

    def plan(self, request: PlanningRequest) -> ObservationPlan:
        result = self.mcbr.plan(request)
        plan = result.plan
        targeted = tuple(
            t for t in plan.targeted_uncertainty if t in set(UncertaintyType)
        ) or tuple(
            UncertaintyType[k.upper()]
            for k in request.need.desired_uncertainty_reduction
            if k.upper() in UncertaintyType.__members__
        )
        return plan.model_copy(
            update={
                "expected_mission_gain": max(plan.expected_mission_gain, value_of_information(request.need)),
                "targeted_uncertainty": targeted,
            }
        )
