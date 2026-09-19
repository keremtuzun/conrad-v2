"""Cadenced decision cycle: EGDC decides, the routed payload goes to MCBR / navigation / BAAC / safety.

DEPLOYMENT PLANE. Proposal != permission: every routed action still passes the navigation stack's goal
checks, the safety supervisor and the CommandGateway before anything moves.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from conrad.active.planner import PriorView
from conrad.decision.actions import REPLAN_ROUTE_BLOCKED
from conrad.decision.egdc import DecisionOutcome
from conrad.decision.router import ExecutiveDirective, RouteTarget, TransmissionRequest
from conrad.orchestration.belief_bus import BeliefBus
from conrad.orchestration.comms import ShoreLink
from conrad.orchestration.deliberation import Deliberation, view_pose
from conrad.orchestration.executive import MissionExecutive
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.orchestration.mission_context import MissionContext
from conrad.orchestration.services import ModuleRunner, RuntimeServices
from conrad.robotics.hardware.interface import RobotHardwareInterface
from conrad.runtime.supervisor import RuntimeState, RuntimeSupervisor
from conrad.schemas.belief import BeliefMessage, BeliefQuery, KnowledgeStatus
from conrad.schemas.decision import ActionType, InformationNeed, NavigationGoal, PlanStatus, ResourceState
from conrad.schemas.events import EventType
from conrad.schemas.robot import HealthLevel, SystemHealth
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain

MODULE = "conrad.orchestration.routing"


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

    # ------------------------------------------------------------------ helpers
    def _critical_heads(self, now: TimeStamp) -> list[BeliefMessage]:
        q = BeliefQuery(domain=Domain.TECHNICAL, entity_ids=self.ctx.critical_component_ids)
        return list(self.bus.query(q, now.time_ns).messages)

    def _modalities_used(self, now: TimeStamp) -> list[str]:
        """Modalities that already contributed DIRECT evidence to a mission-critical belief."""
        used = any(m.evidence_support for m in self._critical_heads(now))
        return [self.d.sensor.modality] if used else []

    def _pending_reports(self, now: TimeStamp) -> list[str]:
        out = []
        for m in self._critical_heads(now):
            if m.world_entity_id not in self.ctx.critical_component_ids:
                continue
            cond = next((c for c in m.state_summary if c.name == "condition"), None)
            if cond is None or cond.status is not KnowledgeStatus.OBSERVED:
                continue
            known = self.shore.reported_revision(m.belief_id)
            if known is None or known < m.revision:
                out.append(str(m.belief_id))
        return out

    def _health(self) -> SystemHealth:
        base = self.hw.get_health()
        return base.model_copy(update={"module_availability": dict(self.s.health.snapshot())})

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
        state = self.x.stack.estimator.get_state()
        power = self.hw.get_power_state()
        budget = self.cfg.time_budget_s or self.ctx.spec.time_budget_s
        start = now.time_ns if self.started_ns is None else self.started_ns
        resources = ResourceState(
            timestamp=now,
            battery_fraction=None if power is None else power.remaining_fraction,
            time_remaining_s=None if budget is None else budget - (now.time_ns - start) / 1e9,
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
            return self.d.decide(now, state, resources, link, health, motion, notes, route_msgs)

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
        if surveying and routed.target in (RouteTarget.MCBR, RouteTarget.NAVIGATION):
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
            self._replan(payload, out, now)

    def _replan(self, payload: ExecutiveDirective, out: DecisionOutcome, now: TimeStamp) -> None:
        """REPLAN(ROUTE_BLOCKED): the executive re-routes around the legs the blocking beliefs occupy."""
        blockers = {str(b) for b in payload.parameters.get("blocking_belief_ids", [])}
        legs, occupancy = self.d.last_route
        blocked = [leg for i, leg in enumerate(legs) if blockers & set(occupancy.get(str(i), []))]
        accepted = self.x.replan_detour(blocked, now, out.provenance.record_id)
        row = {
            "t_s": now.time_ns / 1e9,
            "decision_id": str(out.record.decision_id),
            "blocked_legs": len(blocked),
            "accepted": accepted,
        }
        self.replans.append(row)
        self.s.emit(EventType.PLAN_PROPOSED, MODULE, out.record.trace_id, {"replan": "ROUTE_BLOCKED", **row})

    def _inspect(self, need: InformationNeed, out: DecisionOutcome, now: TimeStamp) -> None:
        key = tuple(sorted(need.target_belief_ids, key=str))
        if self.plan_attempts.get(key, 0) >= self.cfg.max_plans_per_need:
            self.s.emit(
                EventType.ACTION_REJECTED,
                MODULE,
                need.trace_id,
                {"need_id": str(need.need_id), "reason": "PLAN_ATTEMPTS_EXHAUSTED"},
            )
            return
        self.plan_attempts[key] = self.plan_attempts.get(key, 0) + 1
        beliefs = [m for b in need.target_belief_ids if (m := self.bus.head(b)) is not None]
        pose = self.x.stack.estimator.get_state().pose
        adopted = self.runner.call(
            "mcbr", lambda: self.d.plan(need, out, beliefs, pose, now, self.prior_views), need.trace_id
        )
        if adopted is None or adopted.plan.status is not PlanStatus.PLAN:
            return
        plan = adopted.plan
        assert plan.primary_action is not None
        goal = NavigationGoal(
            goal_id=self.ids.new(),
            trace_id=plan.trace_id,
            target_pose=view_pose(plan, self.d.mount_yaw),
            position_tolerance_m=0.35,
            orientation_tolerance_rad=0.3,
            observation_constraints={"primitive": "STATION_KEEP", "duration_s": self.cfg.inspection_dwell_s},
            risk_limit=0.3,
            source_plan_id=plan.plan_id,
            source_action_id=plan.primary_action.action_id,
        )
        if self.x.set_goal(goal, "INSPECT", adopted.adoption_record, now, plan.plan_id):
            self.prior_views.append(
                PriorView(position_m=plan.primary_action.pose.position_m, modality=self.d.sensor.modality)
            )
