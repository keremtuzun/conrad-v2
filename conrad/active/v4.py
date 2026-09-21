"""MCBR V4: a planner that accounts for whether the view it picks will actually be flown.

DECISION PLANE. Same interface as every other planner (``PlanningRequest`` in, ``ObservationPlan`` out) and
the same pipeline (gap -> candidates -> feasibility filter -> rank), so MCBR still only proposes
information acquisition and navigation still moves the robot.

V4 deliberately does NOT re-rank views. ``ACTIVE-MCBR-E006`` measured the ceiling of view selection on
``ACTIVE_INSPECTION_OCCLUDED_V1`` with a truth-seeing oracle: the whole remaining ranking headroom above the
frozen V3 rule is +0.0168 [-0.0821, +0.1269], one world in forty, a fifth of the noise floor of the run that
would have to certify it. V4 therefore keeps V3's ranking rule and changes what reaches the ranker:

* **Execution feasibility.** A candidate whose approach the mission already abandoned (the route was blocked,
  the navigation stack refused the goal, the trajectory was lost) is REFUSED, with a reason code, BEFORE
  ranking. Without this the same unreachable view keeps winning the ranking and keeps not being flown.
* **Budget feasibility.** A candidate that cannot be reached, dwelled on and still leave the mission's
  remaining time or energy is refused before ranking, so the planner stops spending views it cannot finish.

Both are additional entries in the same rejection-reason vocabulary as the shared ``FeasibilityFilter``, and
both run before any score is computed: filter-before-rank is unchanged.

The abandoned-view evidence is belief plane: it is the runtime's own record of goals it accepted and did not
fly (``conrad.orchestration.view_execution``), taken from the EKF estimate and the commanded pose. No twin
truth reaches it.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field

from conrad.active.candidates import RawCandidate
from conrad.active.config import MCBRConfig
from conrad.active.gap import AbandonedView, KnowledgeGap
from conrad.active.planner import MCBRPlanner, PlanningRequest
from conrad.active.rankers import PredictiveRanker, RankerConfig
from conrad.schemas.base import ConradModel
from conrad.schemas.decision import ResourceCost
from conrad.schemas.ids import IdFactory

V4 = "V4"
R_APPROACH_ABANDONED = "APPROACH_ABANDONED"
"""The mission already accepted a view at this pose and did not reach it."""
R_TIME_BUDGET = "TIME_BUDGET_EXCEEDED"
"""Reaching this view and dwelling on it does not fit in the mission time that is left."""
R_ENERGY_BUDGET = "ENERGY_BUDGET_EXCEEDED"

# Abandonment reasons that say something about the APPROACH to a pose rather than about the mission ending.
# A view dropped because the mission clock ran out says nothing about whether that pose is reachable.
APPROACH_REASONS = (
    "ABANDONED_ROUTE_BLOCKED",
    "ABANDONED_NAV_REJECTED",
    "ABANDONED_TRAJECTORY_LOST",
    "ABANDONED_TIMEOUT",
)


class V4Config(ConradModel):
    """Everything V4 adds on top of the frozen V3 ranking rule."""

    ranker: RankerConfig = RankerConfig(kind="bayes_eig", cost_mode="ratio", cost_weight=1.0)
    abandon_filter: bool = Field(
        default=True, description="refuse candidates whose approach the mission already abandoned"
    )
    abandon_radius_m: float = Field(
        default=1.0, gt=0, description="a candidate this close to an abandoned view shares its approach"
    )
    abandon_reasons: tuple[str, ...] = APPROACH_REASONS
    abandon_min_closest_m: float = Field(
        default=0.6,
        ge=0,
        description="only an abandonment that ended this far from the commanded pose counts as a failed "
        "approach; a view abandoned on the spot was reached and lost for another reason",
    )
    budget_filter: bool = Field(
        default=True, description="refuse candidates that do not fit the remaining time / energy"
    )
    budget_reserve_s: float = Field(
        default=0.0, ge=0, description="mission time held back beyond transit plus dwell"
    )
    with_stop: bool = False


class ExecutionFilter:
    """Extra feasibility reasons, evaluated before ranking, for one ``V4Config``."""

    def __init__(self, cfg: V4Config) -> None:
        self.cfg = cfg

    def __call__(
        self,
        raw: RawCandidate,
        visibility: float,
        cost: ResourceCost | None,
        request: PlanningRequest,
        gap: KnowledgeGap,
    ) -> tuple[str, ...]:
        out: list[str] = []
        c = self.cfg
        if c.abandon_filter and self._abandoned(raw, request.abandoned_views):
            out.append(R_APPROACH_ABANDONED)
        if c.budget_filter and cost is not None:
            need_s = cost.time_s + raw.sensor.duration_s + c.budget_reserve_s
            if request.time_remaining_s is not None and need_s > request.time_remaining_s:
                out.append(R_TIME_BUDGET)
            need_j = cost.energy_j + raw.sensor.power_w * raw.sensor.duration_s
            if request.energy_remaining_j is not None and need_j > request.energy_remaining_j:
                out.append(R_ENERGY_BUDGET)
        return tuple(out)

    def _abandoned(self, raw: RawCandidate, views: tuple[AbandonedView, ...]) -> bool:
        if not views:
            return False
        p = np.asarray(raw.pose.position_m, dtype=np.float64)
        r2 = self.cfg.abandon_radius_m**2
        for v in views:
            if v.reason not in self.cfg.abandon_reasons:
                continue
            if math.isfinite(v.closest_approach_m) and v.closest_approach_m < self.cfg.abandon_min_closest_m:
                continue  # that view WAS reached; its pose is not the problem
            if float(np.sum((p - np.asarray(v.position_m, dtype=np.float64)) ** 2)) <= r2:
                return True
        return False


def v4_planner(ids: IdFactory, cfg: MCBRConfig, v4: V4Config, name: str = V4) -> MCBRPlanner:
    """V3's ranking rule behind V4's execution-aware feasibility filter."""
    ranker = PredictiveRanker(v4.ranker)
    return MCBRPlanner(
        ids,
        cfg,
        ranker,
        name=name,
        value_gate=False,
        stop=ranker.stop if v4.with_stop else None,
        extra_filter=ExecutionFilter(v4),
    )


__all__ = [
    "APPROACH_REASONS",
    "R_APPROACH_ABANDONED",
    "R_ENERGY_BUDGET",
    "R_TIME_BUDGET",
    "V4",
    "ExecutionFilter",
    "V4Config",
    "v4_planner",
]
