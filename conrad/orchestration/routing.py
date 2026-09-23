"""Cadenced decision cycle: EGDC decides, the routed payload goes to MCBR / navigation / BAAC / safety.

DEPLOYMENT PLANE. Proposal != permission: every routed action still passes the navigation stack's goal
checks, the safety supervisor and the CommandGateway before anything moves.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any
from uuid import UUID

from conrad.active.planner import PriorView
from conrad.active.v4 import APPROACH_REASONS
from conrad.decision.actions import REPLAN_ROUTE_BLOCKED
from conrad.decision.egdc import DecisionOutcome
from conrad.decision.router import ExecutiveDirective, RouteTarget, TransmissionRequest
from conrad.orchestration.belief_bus import BeliefBus
from conrad.orchestration.comms import ShoreLink
from conrad.orchestration.deliberation import Deliberation, view_pose, yaw_of_quat
from conrad.orchestration.executive import MissionExecutive
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.orchestration.mission_context import MissionContext
from conrad.orchestration.services import ModuleRunner, RuntimeServices
from conrad.orchestration.view_execution import (
    ABANDONED_ROUTE_BLOCKED,
    ABANDONED_SAFE_HOLD,
    ViewCommand,
)
from conrad.robotics.hardware.interface import RobotHardwareInterface
from conrad.runtime.supervisor import RuntimeState, RuntimeSupervisor
from conrad.schemas.belief import BeliefMessage, BeliefQuery, KnowledgeStatus
from conrad.schemas.decision import ActionType, InformationNeed, NavigationGoal, PlanStatus, ResourceState
from conrad.schemas.events import EventType
from conrad.schemas.robot import HealthLevel, SystemHealth
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain

MODULE = "conrad.orchestration.routing"
_UNSEEN_REVISION = object()
"""Sentinel: the condition of a revision this runtime never saw (then a newer revision is still reportable)."""


def _is_calibration_request(target: RouteTarget, payload: Any) -> bool:
    """Whether this routed payload is the explicit Model 1 calibration-freshness requirement."""

    return (
        target is RouteTarget.MCBR
        and isinstance(payload, InformationNeed)
        and bool(payload.constraints.get("calibration_check"))
    )


def _allows_calibration_deferral_refund(beliefs: list[BeliefMessage]) -> bool:
    """Whether an unflown calibration view protects a nominal INTACT decision.

    An independent confirming look is needed to authorize continued operation on an uncalibrated INTACT
    assessment.  A DEGRADED/SEVERE/FAILED assessment already warrants the conservative finding path; giving
    those confirmation views unlimited routing deferrals caused needless close-ins and safety holds.  This
    predicate reads only the target belief message, never Twin truth or a scenario identifier.
    """

    for message in beliefs:
        condition = next((claim for claim in message.state_summary if claim.name == "condition"), None)
        if (
            condition is not None
            and condition.status is KnowledgeStatus.OBSERVED
            and str(condition.value) == "INTACT"
        ):
            return True
    return False


class DecisionRouting:
    def __init__(
        self,
        s: RuntimeServices,
        ctx: MissionContext,
        cfg: MissionRuntimeConfig,
        bus: BeliefBus,
        deliberation: Deliberation,
        executive: MissionExecutive,
        shore: ShoreLink,
        supervisor: RuntimeSupervisor,
        hw: RobotHardwareInterface,
        runner: ModuleRunner,
    ) -> None:
        self.s, self.ctx, self.cfg, self.bus, self.hw, self.runner = s, ctx, cfg, bus, hw, runner
        self.d, self.x, self.shore, self.supervisor = deliberation, executive, shore, supervisor
        self.ids = s.ids.child("routing")
        self.phase = "NOT_STARTED"
        self.plan_attempts: dict[tuple[UUID, ...], int] = {}
        self.prior_views: list[PriorView] = []
        self.started_ns: int | None = None  # set by MissionRuntime.start (mission clock origin)
        self.replans: list[dict[str, Any]] = []
        # View execution bookkeeping (V4). ``empty_plans`` is reported for every arm: a plan that comes
        # back without a primary action used to be dropped in silence.
        self.empty_plans: list[dict[str, Any]] = []
        self.view_outcomes: list[dict[str, Any]] = []
        self.view_attempts: dict[tuple[UUID, ...], int] = {}
        self.deferred_replans: list[dict[str, Any]] = []
        self._plan_need: dict[str, tuple[UUID, ...]] = {}
        self._need_by_plan: dict[str, InformationNeed] = {}
        self._unavailable_information_targets: dict[UUID, str] = {}
        self._calibration_coalesced: set[tuple[str, tuple[UUID, ...]]] = set()
        self._views_seen = 0
        self._time_remaining_s: float | None = None
        self._energy_remaining_j: float | None = None
        # (belief id, revision) -> reported condition value, so a re-report needs a CHANGED condition.
        self._condition_at: dict[tuple[UUID, int], str | None] = {}
        # Gate I6 hook (conrad.orchestration.multidomain.SensingConditionsGate.check): returns a deferral reason for
        # an information request that must not go to MCBR now. None (default) = no gate.
        self.sensing_gate: Callable[[InformationNeed, DecisionOutcome, TimeStamp], str | None] | None = None

    # ------------------------------------------------------------------ helpers
    def _critical_heads(self, now: TimeStamp) -> list[BeliefMessage]:
        q = BeliefQuery(domain=Domain.TECHNICAL, entity_ids=self.ctx.critical_component_ids)
        return list(self.bus.query(q, now.time_ns).messages)

    def _modalities_used(self, now: TimeStamp) -> list[str]:
        """Modalities that already contributed DIRECT evidence to a mission-critical belief."""
        used = any(m.evidence_support for m in self._critical_heads(now))
        return [self.d.sensor.modality] if used else []

    def _pending_reports(self, now: TimeStamp) -> list[str]:
        """Critical components whose OBSERVED condition the shore does not have yet.

        A report is pending on a MEANINGFUL change, not on any new revision: every reading raises the belief
        revision, so keying off the revision alone re-queued a report for an unchanged component at every
        cycle and pushed the nominal "continue" warrant away for ever (docs/audits/I5_ACTION_MATRIX.md
        iteration 2, open item 3). The reported condition is what the shore acts on, so the change that
        matters is a change of that value."""
        out = []
        for m in self._critical_heads(now):
            if m.world_entity_id not in self.ctx.critical_component_ids:
                continue
            cond = next((c for c in m.state_summary if c.name == "condition"), None)
            if cond is None or cond.status is not KnowledgeStatus.OBSERVED:
                continue
            value = None if cond.value is None else str(cond.value)
            self._condition_at[(m.belief_id, m.revision)] = value
            known = self.shore.reported_revision(m.belief_id)
            if known is None:
                out.append(str(m.belief_id))
                continue
            delivered = self._condition_at.get((m.belief_id, known), _UNSEEN_REVISION)
            if known < m.revision and (delivered is _UNSEEN_REVISION or delivered != value):
                out.append(str(m.belief_id))
        return out

    def _health(self) -> SystemHealth:
        base = self.hw.get_health()
        return base.model_copy(update={"module_availability": dict(self.s.health.snapshot())})

    # ------------------------------------------------------------------ view execution bookkeeping
    def reconcile_views(self) -> None:
        """Read the view execution ledger's newly closed records and act on what they say.

        This is the loop the V3 protocol did not have: an accepted view that was never flown is visible
        here, with a reason, and the planner is told about it on the next cycle. With the V4 switches off
        the method only records, so the incumbent behaviour is unchanged.
        """
        ve = self.cfg.view_execution
        records = self.x.views.records
        for rec in records[self._views_seen :]:
            self.view_outcomes.append(
                {
                    "t_end_s": rec.t_end_s,
                    "plan_id": rec.plan_id,
                    "outcome": rec.outcome,
                    "closest_approach_m": (
                        rec.closest_approach_m if math.isfinite(rec.closest_approach_m) else None
                    ),
                    "reached": rec.reached,
                    "flown": rec.flown,
                }
            )
            key = self._plan_need.get(rec.plan_id or "")
            if not ve.enabled or key is None:
                continue
            if rec.outcome in APPROACH_REASONS or rec.outcome == ABANDONED_SAFE_HOLD:
                if ve.drop_abandoned_prior_view:
                    self._drop_prior_view(rec.commanded_position_m)
                if ve.refund_abandoned_attempt and self.view_attempts.get(key, 0) < ve.max_view_attempts:
                    self.plan_attempts[key] = max(0, self.plan_attempts.get(key, 1) - 1)
        self._views_seen = len(records)

    def _drop_prior_view(self, position_m: list[float]) -> None:
        """A view that was never flown delivered nothing, so it is not a prior view of the belief."""
        for i in range(len(self.prior_views) - 1, -1, -1):
            if all(
                abs(a - b) < 1e-6 for a, b in zip(self.prior_views[i].position_m, position_m, strict=True)
            ):
                self.prior_views.pop(i)
                return

    # ------------------------------------------------------------------ cycle
    def cycle(self, now: TimeStamp, goal_finished: bool) -> DecisionOutcome | None:
        active = self.x.active
        if goal_finished and active is not None:
            if active.purpose == "TRANSIT":
                self.x.transit_done = True
                self.phase = "RESOLVING"  # survey-then-resolve: lane pass done, open needs are planned now
                self.x.hold_goal(now, "HOLD")
            elif active.purpose in ("INSPECT", "REVISIT"):
                if self.x.transit_done:
                    self.x.hold_goal(now, "HOLD")
                else:
                    self.x.resume_transit(now)
        self.reconcile_views()
        state = self.x.stack.estimator.get_state()
        power = self.hw.get_power_state()
        budget = self.cfg.time_budget_s or self.ctx.spec.time_budget_s
        start = now.time_ns if self.started_ns is None else self.started_ns
        resources = ResourceState(
            timestamp=now,
            battery_fraction=None if power is None else power.remaining_fraction,
            time_remaining_s=None if budget is None else budget - (now.time_ns - start) / 1e9,
        )
        elapsed_s = (now.time_ns - start) / 1e9
        horizon = budget
        if horizon is None and self.cfg.view_execution.budget_from_mission_duration:
            horizon = self.cfg.duration_s  # no operator budget: the mission's own clock is the horizon
        self._time_remaining_s = None if horizon is None else max(0.0, horizon - elapsed_s)
        # BatteryState reports a remaining FRACTION and the energy already used, so the energy still
        # available is what has been used scaled by what is left of the pack.
        self._energy_remaining_j = (
            None
            if power is None or power.remaining_fraction >= 1.0 - 1e-9
            else power.energy_used_j * power.remaining_fraction / (1.0 - power.remaining_fraction)
        )
        link = self.shore.link_state(now.time_ns / 1e9)
        safety_ok = self.x.safety_state in ("NORMAL", "DEGRADED")
        motion = safety_ok and self.supervisor.state in (RuntimeState.RUNNING, RuntimeState.DEGRADED)
        notes: dict[str, Any] = {
            "phase": "INSPECTING" if active is not None and active.purpose == "INSPECT" else self.phase,
            "modalities_used": self._modalities_used(now),
            "pending_report_belief_ids": self._pending_reports(now),
        }
        health = self._health()
        if health.overall is HealthLevel.UNKNOWN:
            health = health.model_copy(update={"overall": HealthLevel.OK})

        def decide() -> DecisionOutcome:
            route_notes, route_msgs = self.d.route_context(self.x.planned_route(), now)
            notes.update(route_notes)
            current = self.x.active
            active_needs: tuple[InformationNeed, ...] = ()
            if current is not None and current.purpose in ("INSPECT", "REVISIT"):
                plan_id = None if current.plan_id is None else str(current.plan_id)
                need = self._need_by_plan.get(plan_id or "")
                if need is not None:
                    active_needs = (need,)
            return self.d.decide(
                now,
                state,
                resources,
                link,
                health,
                motion,
                notes,
                route_msgs,
                active_information_needs=active_needs,
                unavailable_information_targets=dict(self._unavailable_information_targets),
            )

        out: DecisionOutcome | None = self.runner.call("egdc", decide)
        if out is None:
            return None
        self.x.latest_decision_record = out.provenance.record_id
        self._route(out, now)
        return out

    def _route(self, out: DecisionOutcome, now: TimeStamp) -> None:
        routed = out.routed
        if routed is None or routed.target is None:
            return
        payload = routed.payload
        busy = self.x.active is not None and self.x.active.purpose in ("INSPECT", "REVISIT", "SAFE_HOLD")
        surveying = not self.x.transit_done
        calibration = _is_calibration_request(routed.target, payload)
        if surveying and routed.target in (RouteTarget.MCBR, RouteTarget.NAVIGATION) and not calibration:
            self.s.emit(
                EventType.ACTION_REJECTED,
                MODULE,
                out.record.trace_id,
                {
                    "decision_id": str(out.record.decision_id),
                    "route": routed.target.value,
                    "reason": "DEFERRED_UNTIL_LANE_SURVEY_COMPLETE",
                },
            )
            self.d.mark_not_executed(out.record.decision_id)  # never carried out: not an attempt
            return
        if busy and routed.target in (RouteTarget.MCBR, RouteTarget.NAVIGATION) and not calibration:
            # an inspection / revisit is already running (it is the attempt); this request is dropped
            self.d.mark_not_executed(out.record.decision_id)
            return
        if routed.target is RouteTarget.MCBR and isinstance(payload, InformationNeed) and not busy:
            self._inspect(payload, out, now)
        elif routed.target is RouteTarget.NAVIGATION and isinstance(payload, NavigationGoal) and not busy:
            goal = payload.model_copy(
                update={
                    "observation_constraints": {"primitive": "INSPECT", "standoff_m": 2.0, "dwell_s": 4.0}
                }
            )
            self.x.set_goal(goal, "REVISIT", out.provenance.record_id, now)
        elif routed.target is RouteTarget.BAAC and isinstance(payload, TransmissionRequest):
            heads = [m for b in payload.belief_ids if (m := self.bus.head(b)) is not None]
            self.shore.offer(heads, now, floor=payload.priority)
        elif routed.target is RouteTarget.OPERATOR:
            self.shore.offer(self._critical_heads(now), now, floor=0.5)
        elif routed.target is RouteTarget.SAFETY_SUPERVISOR and isinstance(payload, ExecutiveDirective):
            self.supervisor.safe_hold(f"decision {payload.action_type.value}")
        elif (
            routed.target is RouteTarget.MISSION_EXECUTIVE
            and isinstance(payload, ExecutiveDirective)
            and payload.action_type is ActionType.REPLAN
            and payload.parameters.get("reason") == REPLAN_ROUTE_BLOCKED
        ):
            inspecting = self.x.active is not None and self.x.active.purpose in ("INSPECT", "REVISIT")
            if self.cfg.view_execution.enabled and self.cfg.view_execution.protect_active_view and inspecting:
                # A route-blocked detour of a NON-transit goal is a hold, and a hold abandons the view.
                # The running inspection IS the attempt on this need, exactly as the busy guard above
                # already treats a fresh MCBR or NAVIGATION request, so the detour is deferred instead.
                # Nothing about safety changes: the local planner and the safety supervisor are untouched.
                row = {
                    "t_s": now.time_ns / 1e9,
                    "decision_id": str(out.record.decision_id),
                    "reason": "DEFERRED_WHILE_VIEW_IN_FLIGHT",
                    "purpose": self.x.active.purpose if self.x.active else None,
                }
                self.deferred_replans.append(row)
                self.s.emit(EventType.ACTION_REJECTED, MODULE, out.record.trace_id, row)
                self.d.mark_not_executed(out.record.decision_id)
                return
            self._replan(payload, out, now)

    def _replan(self, payload: ExecutiveDirective, out: DecisionOutcome, now: TimeStamp) -> None:
        """REPLAN(ROUTE_BLOCKED): the executive re-routes around the legs the blocking beliefs occupy."""
        blockers = {str(b) for b in payload.parameters.get("blocking_belief_ids", [])}
        legs, occupancy = self.d.last_route
        blocked = [leg for i, leg in enumerate(legs) if blockers & set(occupancy.get(str(i), []))]
        # only a transit goal gets a detour; any other goal is held (replan_detour), which aborts it
        transit = self.x.active is not None and self.x.active.purpose == "TRANSIT"
        accepted = self.x.replan_detour(
            blocked, now, out.provenance.record_id, preempt_reason=ABANDONED_ROUTE_BLOCKED
        )
        row = {
            "t_s": now.time_ns / 1e9,
            "decision_id": str(out.record.decision_id),
            "blocked_legs": len(blocked),
            "accepted": accepted,
            "mode": "DETOUR" if transit else "HOLD",
        }
        self.replans.append(row)
        self.s.emit(EventType.PLAN_PROPOSED, MODULE, out.record.trace_id, {"replan": "ROUTE_BLOCKED", **row})

    def _inspect(self, need: InformationNeed, out: DecisionOutcome, now: TimeStamp) -> None:
        if self.sensing_gate is not None and self.sensing_gate(need, out, now) is not None:
            self.d.mark_not_executed(out.record.decision_id)  # deferred, not carried out: not an attempt
            return
        key = tuple(sorted(need.target_belief_ids, key=str))
        calibration = bool(need.constraints.get("calibration_check"))
        active = self.x.active
        active_plan = None if active is None or active.plan_id is None else str(active.plan_id)
        if calibration and active_plan is not None and self._plan_need.get(active_plan) == key:
            token = (active_plan, key)
            if token in self._calibration_coalesced:
                self.d.mark_not_executed(out.record.decision_id)
                return
            self._calibration_coalesced.add(token)
            self.s.emit(
                EventType.PLAN_PROPOSED,
                MODULE,
                need.trace_id,
                {
                    "need_id": str(need.need_id),
                    "execution": "CALIBRATION_COALESCED_WITH_ACTIVE_VIEW",
                    "active_plan_id": active_plan,
                    "target_belief_ids": [str(b) for b in key],
                },
            )
            return
        if self.plan_attempts.get(key, 0) >= self.cfg.max_plans_per_need:
            for belief_id in key:
                self._unavailable_information_targets[belief_id] = "PLAN_ATTEMPTS_EXHAUSTED"
            self.s.emit(
                EventType.ACTION_REJECTED,
                MODULE,
                need.trace_id,
                {"need_id": str(need.need_id), "reason": "PLAN_ATTEMPTS_EXHAUSTED"},
            )
            return
        ve = self.cfg.view_execution
        if ve.enabled and self.view_attempts.get(key, 0) >= ve.max_view_attempts:
            for belief_id in key:
                self._unavailable_information_targets[belief_id] = "VIEW_ATTEMPTS_EXHAUSTED"
            self.s.emit(
                EventType.ACTION_REJECTED,
                MODULE,
                need.trace_id,
                {"need_id": str(need.need_id), "reason": "VIEW_ATTEMPTS_EXHAUSTED"},
            )
            return
        self.plan_attempts[key] = self.plan_attempts.get(key, 0) + 1
        beliefs = [m for b in need.target_belief_ids if (m := self.bus.head(b)) is not None]
        pose = self.x.stack.estimator.get_state().pose
        abandoned = self.x.views.abandoned() if ve.enabled else ()
        t_left = self._time_remaining_s if ve.enabled and ve.declare_budgets else None
        e_left = self._energy_remaining_j if ve.enabled and ve.declare_budgets else None
        adopted = self.runner.call(
            "mcbr",
            lambda: self.d.plan(need, out, beliefs, pose, now, self.prior_views, abandoned, t_left, e_left),
            need.trace_id,
        )
        if adopted is None or adopted.plan.status is not PlanStatus.PLAN:
            self._report_empty_plan(need, adopted, now)
            information_limit = min(
                self.cfg.max_plans_per_need, self.d.decision_config.max_information_attempts
            )
            if self.plan_attempts.get(key, 0) >= information_limit:
                plan_status = "MODULE_UNAVAILABLE" if adopted is None else adopted.plan.status.value
                status = f"{plan_status}:INFORMATION_ATTEMPT_BUDGET_EXHAUSTED"
                for belief_id in key:
                    self._unavailable_information_targets[belief_id] = status
            return
        plan = adopted.plan
        act = plan.primary_action
        assert act is not None
        inline = bool(act.sensor_configuration.get("calibration_inline"))
        if inline:
            for belief_id in key:
                self._unavailable_information_targets.pop(belief_id, None)
            self.view_attempts[key] = self.view_attempts.get(key, 0) + 1
            self.prior_views.append(
                PriorView(position_m=act.pose.position_m, modality=self.d.sensor.modality)
            )
            self.s.emit(
                EventType.PLAN_PROPOSED,
                MODULE,
                need.trace_id,
                {
                    "need_id": str(need.need_id),
                    "plan_id": str(plan.plan_id),
                    "action_id": str(act.action_id),
                    "execution": "CALIBRATION_INLINE_CURRENT_VIEW",
                    "predicted_visibility": act.predicted_visibility,
                },
            )
            return
        if calibration and (
            not self.x.transit_done
            or (self.x.active is not None and self.x.active.purpose in ("INSPECT", "REVISIT", "SAFE_HOLD"))
        ):
            self.d.mark_not_executed(out.record.decision_id)
            # The planner found a feasible view, but routing never launched it because the active goal is
            # deliberately non-preemptible.  Under the frozen V4 accounting rule, an unflown view is not an
            # information-acquisition attempt.  Refund only this routing deferral; empty/infeasible plans and
            # views actually handed to the executive remain charged.  The active goal continues to make
            # progress, so this cannot reproduce the old no-goal 58/60 unexecutable-request loop.
            if ve.enabled and ve.refund_abandoned_attempt and _allows_calibration_deferral_refund(beliefs):
                self.plan_attempts[key] = max(0, self.plan_attempts.get(key, 1) - 1)
            self.s.emit(
                EventType.ACTION_REJECTED,
                MODULE,
                need.trace_id,
                {
                    "need_id": str(need.need_id),
                    "plan_id": str(plan.plan_id),
                    "reason": "CALIBRATION_VIEW_REQUIRES_GOAL_PREEMPTION",
                },
            )
            return
        target = view_pose(plan, self.d.mount_yaw)
        goal = NavigationGoal(
            goal_id=self.ids.new(),
            trace_id=plan.trace_id,
            target_pose=target,
            position_tolerance_m=self.cfg.view_position_tolerance_m,
            orientation_tolerance_rad=self.cfg.view_orientation_tolerance_rad,
            observation_constraints={"primitive": "STATION_KEEP", "duration_s": self.cfg.inspection_dwell_s},
            risk_limit=0.3,
            source_plan_id=plan.plan_id,
            source_action_id=act.action_id,
        )
        view = ViewCommand(
            plan_id=plan.plan_id,
            action_id=act.action_id,
            need_id=need.need_id,
            position_m=act.pose.position_m,
            yaw_rad=yaw_of_quat(target.orientation_wxyz),
            aim_point_m=act.target_region.center_m,
            position_tolerance_m=goal.position_tolerance_m,
            orientation_tolerance_rad=goal.orientation_tolerance_rad,
            dwell_s=self.cfg.inspection_dwell_s,
            mount_yaw_rad=self.d.mount_yaw,
            predicted_visibility=act.predicted_visibility,
        )
        plan_key = str(plan.plan_id)
        self._plan_need[plan_key] = key
        if self.x.set_goal(goal, "INSPECT", adopted.adoption_record, now, plan.plan_id, view=view):
            for belief_id in key:
                self._unavailable_information_targets.pop(belief_id, None)
            self._need_by_plan[plan_key] = need
            self.view_attempts[key] = self.view_attempts.get(key, 0) + 1
            self.prior_views.append(
                PriorView(position_m=act.pose.position_m, modality=self.d.sensor.modality)
            )

    def _report_empty_plan(self, need: InformationNeed, adopted: Any, now: TimeStamp) -> None:
        """A plan without a primary action is recorded, not dropped: it is the candidate-supply signal."""
        row = empty_plan_row(need, adopted, now.time_ns / 1e9)
        self.empty_plans.append(row)
        self.s.emit(EventType.ACTION_REJECTED, MODULE, need.trace_id, {"empty_plan": True, **row})


def empty_plan_row(need: InformationNeed, adopted: Any, t_s: float) -> dict[str, Any]:
    """One record of a plan that produced no action, with the reason codes that refused its candidates.

    A plan with no primary action used to return silently, so "MCBR planned" and "MCBR produced a view"
    looked identical in the artifacts. This row is what makes the candidate-supply failure countable.
    """
    plan = None if adopted is None else adopted.plan
    reasons: dict[str, int] = {}
    for r in plan.rejected if plan is not None else ():
        for code in r.reason_codes:
            reasons[code] = reasons.get(code, 0) + 1
    table = [] if adopted is None else list(adopted.table)
    return {
        "t_s": float(t_s),
        "need_id": str(need.need_id),
        "plan_id": None if plan is None else str(plan.plan_id),
        "status": "MODULE_UNAVAILABLE" if plan is None else plan.status.value,
        "candidates": len(table),
        "feasible": sum(1 for r in table if r.get("feasible")),
        "rejection_reasons": dict(sorted(reasons.items())),
    }
