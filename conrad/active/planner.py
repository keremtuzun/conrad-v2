"""MCBR planner: gap -> candidates -> feasibility filter -> score -> ObservationPlan (ch16, ch18, ch33).

Filtering happens before ranking. Stop conditions produce an explicit status:
NEED_SATISFIED, NO_FEASIBLE_OBSERVATION (includes a missing target belief) or NOT_WORTH_COST.
MCBR proposes information acquisition only; navigation turns the plan into motion.

implementation_status: EXPERIMENTAL_CANDIDATE (scoring) inside FROZEN_CONTRACT output
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from conrad.active.candidates import (
    FeasibilityFilter,
    IsFree,
    MissionBounds,
    NavigationCost,
    PredictedVisibility,
    RawCandidate,
    SensorOption,
    ViewpointGenerator,
)
from conrad.active.config import MCBRConfig
from conrad.active.eig import GainBreakdown, InformationGainEstimator
from conrad.active.gap import (
    AbandonedView,
    KnowledgeGap,
    PriorView,
    build_knowledge_gaps,
    need_satisfied,
)
from conrad.active.predictive import PredictiveBelief, predicted_coverage
from conrad.schemas.belief import BeliefMessage
from conrad.schemas.decision import (
    InformationNeed,
    ObservationAction,
    ObservationPlan,
    PlanStatus,
    RejectedCandidate,
    ResourceCost,
)
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp


@dataclass
class PlanningRequest:
    """Everything a planner may use. Callables are computed from BELIEF, never from twin truth."""

    need: InformationNeed
    beliefs: Sequence[BeliefMessage]
    robot_pose: Pose
    sensors: tuple[SensorOption, ...]
    is_free: IsFree
    predicted_visibility: PredictedVisibility
    navigation_cost: NavigationCost
    now: TimeStamp
    prior_views: tuple[PriorView, ...] = ()
    bounds: MissionBounds | None = None
    rng: np.random.Generator | None = None
    # Optional belief-side predictive model (conrad.active.predictive). Planners that need it fall back to
    # the analytic channel estimate when it is absent; it is computed from BELIEF, never from twin truth.
    predictive: PredictiveBelief | None = None
    # Views this mission accepted and did not fly, with the runtime's reason. Belief plane. Empty by
    # default, so every existing planner and baseline sees exactly the request it saw before.
    abandoned_views: tuple[AbandonedView, ...] = ()
    # What the mission has left when the plan is made. ``None`` = the caller does not declare a budget and
    # the feasibility filter does not test one (the pre-V4 behaviour).
    time_remaining_s: float | None = None
    energy_remaining_j: float | None = None


@dataclass
class ScoredCandidate:
    raw: RawCandidate
    action: ObservationAction
    gain: GainBreakdown
    mission_value: float
    cost: float
    score: float


@dataclass
class PlanResult:
    plan: ObservationPlan
    provenance: ProvenanceRecord
    table: list[dict[str, Any]] = field(default_factory=list)


class ObservationPlanner(Protocol):
    name: str

    def plan(self, request: PlanningRequest) -> PlanResult: ...


class CandidateScorer(Protocol):
    """Ranks FEASIBLE candidates. Baselines differ only here."""

    def __call__(self, gap: KnowledgeGap, c: ScoredCandidate, req: PlanningRequest) -> float: ...


class StopRule(Protocol):
    """True = the best feasible candidate is not worth taking (NOT_WORTH_COST)."""

    def __call__(self, best: ScoredCandidate, req: PlanningRequest) -> bool: ...


class ExtraFilter(Protocol):
    """Additional feasibility reasons, evaluated with the shared filter and BEFORE any ranking.

    Returning a non-empty tuple refuses the candidate, exactly as ``FeasibilityFilter`` does, and the
    reason codes are kept in the plan's rejected list. It cannot promote a candidate the shared filter
    refused, so filter-before-rank still holds for every planner that supplies one.
    """

    def __call__(
        self,
        raw: RawCandidate,
        visibility: float,
        cost: ResourceCost | None,
        request: PlanningRequest,
        gap: KnowledgeGap,
    ) -> tuple[str, ...]: ...


class MCBRPlanner:
    """Shared pipeline; ``scorer`` is the ranking rule. Default = full MCBR (mission value - cost).

    Stop rule: ``stop`` if given; otherwise the legacy MCBR gate (value < cost or gain < min) when
    ``value_gate`` is True; otherwise the planner never refuses a feasible candidate.
    """

    def __init__(
        self,
        id_factory: IdFactory,
        config: MCBRConfig | None = None,
        scorer: CandidateScorer | None = None,
        name: str = "mcbr_full",
        value_gate: bool = True,
        stop: StopRule | None = None,
        extra_filter: ExtraFilter | None = None,
    ) -> None:
        self._ids = id_factory
        self.config = config or MCBRConfig()
        self.name = name
        self.value_gate = value_gate
        self.stop = stop
        self.extra_filter = extra_filter
        self.generator = ViewpointGenerator(self.config)
        self.filter = FeasibilityFilter(self.config)
        self.eig = InformationGainEstimator(self.config)
        self.scorer: CandidateScorer = scorer or (lambda gap, c, req: c.mission_value - c.cost)

    def plan(self, request: PlanningRequest) -> PlanResult:
        need = request.need
        gaps = build_knowledge_gaps(need, request.beliefs, self.config, request.prior_views)
        targeted = tuple(dict.fromkeys(t for g in gaps for t in g.targeted))
        if not gaps:
            return self._finish(request, PlanStatus.NO_FEASIBLE_OBSERVATION, [], [], targeted, ())
        if need_satisfied(need, gaps, self.config):
            return self._finish(request, PlanStatus.NEED_SATISFIED, [], [], targeted, gaps)
        gap = gaps[0]
        raws = self.generator.generate(gap.target_region, request.sensors)
        calibration_check = bool(need.constraints.get("calibration_check"))
        if calibration_check:
            raws.extend(
                self.generator.calibration_at_pose(
                    gap.target_region, request.sensors, request.robot_pose, len(raws)
                )
            )
        free = (
            np.asarray(request.is_free(np.array([r.pose.position_m for r in raws])), dtype=bool)
            if raws
            else np.zeros(0, dtype=bool)
        )
        feasible: list[ScoredCandidate] = []
        rejected: list[RejectedCandidate] = []
        for raw, ok in zip(raws, free, strict=True):
            vis = float(np.clip(request.predicted_visibility(raw.pose, gap.target_region), 0.0, 1.0))
            if request.predictive is not None and self.config.sensor_aware_visibility:
                vis = min(vis, predicted_coverage(request.predictive, raw.pose, raw.sensor))
            cost = request.navigation_cost(request.robot_pose, raw.pose)
            reasons = self.filter.reasons(raw, bool(ok), vis, cost, need, request.bounds)
            if self.extra_filter is not None:
                reasons = reasons + self.extra_filter(raw, vis, cost, request, gap)
            scored = self._score(gap, raw, vis, cost, need)
            if reasons:
                rejected.append(RejectedCandidate(action=scored.action, reason_codes=reasons))
            else:
                feasible.append(scored)
        if not feasible:
            return self._finish(request, PlanStatus.NO_FEASIBLE_OBSERVATION, [], rejected, targeted, gaps)
        for c in feasible:
            c.score = float(self.scorer(gap, c, request))
        if calibration_check:
            # The open requirement is a fresh DIRECT revision, not a lower raw uncertainty value. Among
            # safe, feasible observations, close that freshness gap with the shortest completion time.
            feasible.sort(
                key=lambda c: (
                    c.action.expected_cost.time_s,
                    c.action.expected_cost.energy_j,
                    c.action.expected_cost.risk,
                    c.action.expected_cost.travel_m,
                    -c.action.predicted_visibility,
                    c.raw.index,
                )
            )
        else:
            feasible.sort(key=lambda c: (-c.score, c.raw.index))
        best = feasible[0]
        if calibration_check:
            # A zero uncertainty-reduction score cannot close a revision-freshness requirement. Feasibility
            # remains mandatory, but the ordinary value/gain stop is not the semantic stop for this request.
            refuse = False
        elif self.stop is not None:
            refuse = bool(self.stop(best, request))
        else:
            refuse = self.value_gate and (
                best.mission_value < best.cost or best.gain.total < self.config.min_expected_gain
            )
        if refuse:
            return self._finish(request, PlanStatus.NOT_WORTH_COST, feasible, rejected, targeted, gaps)
        return self._finish(request, PlanStatus.PLAN, feasible, rejected, targeted, gaps)

    # ------------------------------------------------------------------ internals
    def _score(
        self,
        gap: KnowledgeGap,
        raw: RawCandidate,
        vis: float,
        cost: ResourceCost | None,
        need: InformationNeed,
    ) -> ScoredCandidate:
        gain = self.eig.estimate(gap, raw, vis)
        nominal = cost or ResourceCost(time_s=0.0, energy_j=0.0, risk=1.0, travel_m=0.0)
        expected = nominal.model_copy(
            update={
                "time_s": nominal.time_s + raw.sensor.duration_s,
                "energy_j": nominal.energy_j + raw.sensor.power_w * raw.sensor.duration_s,
            }
        )
        action = ObservationAction(
            action_id=self._ids.new(),
            pose=raw.pose,
            sensor_id=raw.sensor.sensor_id,
            sensor_configuration={
                "modality": raw.sensor.modality,
                "standoff_m": raw.standoff_m,
                **raw.configuration,
            },
            target_region=gap.target_region,
            duration_s=raw.sensor.duration_s,
            expected_cost=expected,
            predicted_visibility=vis,
            expected_information_gain=gain.total,
            hypothesis_discrimination=gain.discrimination,
        )
        return ScoredCandidate(
            raw=raw,
            action=action,
            gain=gain,
            mission_value=self.eig.mission_value(gap, gain, need.priority),
            cost=self.eig.cost(expected),
            score=0.0,
        )

    def _finish(
        self,
        request: PlanningRequest,
        status: PlanStatus,
        feasible: list[ScoredCandidate],
        rejected: list[RejectedCandidate],
        targeted: tuple[Any, ...],
        gaps: Sequence[KnowledgeGap],
    ) -> PlanResult:
        need = request.need
        parents = tuple(dict.fromkeys(p for g in gaps for p in g.provenance_refs))
        provenance = ProvenanceRecord(
            record_id=self._ids.new(),
            source_type=SourceType.PLAN,
            source_ids=(need.need_id, *need.target_belief_ids),
            operation=f"mcbr.plan:{status.value}",
            module="conrad.active.planner",
            model_version=f"{self.config.model_version}+{self.name}",
            timestamp=request.now,
            parent_records=parents,
        )
        best = feasible[0] if feasible and status is PlanStatus.PLAN else None
        spread = [c.score for c in feasible[:2]]
        confidence = 0.0
        if best is not None:
            margin = spread[0] - spread[1] if len(spread) > 1 else abs(spread[0])
            confidence = float(
                min(1.0, best.action.predicted_visibility * (0.5 + margin / (abs(spread[0]) + 1e-6)))
            )
        plan = ObservationPlan(
            plan_id=self._ids.new(),
            need_id=need.need_id,
            trace_id=need.trace_id,
            status=status,
            target_beliefs=need.target_belief_ids,
            primary_action=None if best is None else best.action,
            alternatives=tuple(c.action for c in feasible[1 : 1 + self.config.n_alternatives])
            if best
            else (),
            rejected=tuple(rejected),
            expected_information_gain=0.0 if best is None else best.gain.total,
            expected_mission_gain=0.0 if best is None else best.mission_value,
            targeted_uncertainty=tuple(targeted),
            expected_cost=None if best is None else best.action.expected_cost,
            confidence=max(0.0, confidence),
            provenance=provenance.record_id,
        )
        table = [self._row(c, True, ()) for c in feasible] + [
            {
                "action_id": str(r.action.action_id),
                "feasible": False,
                "reason_codes": list(r.reason_codes),
                "modality": r.action.sensor_configuration.get("modality"),
                "position_m": list(r.action.pose.position_m),
                "visibility": r.action.predicted_visibility,
            }
            for r in rejected
        ]
        return PlanResult(plan=plan, provenance=provenance, table=table)

    @staticmethod
    def _row(c: ScoredCandidate, feasible: bool, reasons: tuple[str, ...]) -> dict[str, Any]:
        return {
            "action_id": str(c.action.action_id),
            "feasible": feasible,
            "reason_codes": list(reasons),
            "modality": c.raw.sensor.modality,
            "position_m": list(c.raw.pose.position_m),
            "visibility": c.action.predicted_visibility,
            "gain": c.gain.model_dump(),
            "mission_value": c.mission_value,
            "cost": c.cost,
            "score": c.score,
        }
