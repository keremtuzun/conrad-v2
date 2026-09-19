"""Mission executive: navigation goals, the control tick and gateway submission. DEPLOYMENT PLANE.

NavigationStack produces an AllocatedCommand every tick; it reaches ``RobotHardwareInterface.send`` only
through the CommandGateway and only when the safety supervisor authorized it. Each submitted command gets a
COMMAND provenance record whose parent is the plan that motivated its goal.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np

from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.orchestration.mission_context import MissionContext
from conrad.orchestration.services import RuntimeServices
from conrad.robotics.navigation import GoalStatus, NavigationStack
from conrad.runtime.command_gateway import CommandGateway
from conrad.schemas.decision import NavigationGoal
from conrad.schemas.events import EventType, Severity
from conrad.schemas.frames import WORLD, Pose, SpatialSupport
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp

MODULE = "conrad.orchestration.executive"


@dataclass
class ActiveGoal:
    goal: NavigationGoal
    purpose: str  # TRANSIT | INSPECT | HOLD | REVISIT
    parent_record: UUID
    started_ns: int
    plan_id: UUID | None = None


@dataclass
class ExecutiveStats:
    steps: int = 0
    refused_by_supervisor: int = 0
    accepted: int = 0
    rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    safety_events: list[dict[str, Any]] = field(default_factory=list)
    goals: list[dict[str, Any]] = field(default_factory=list)


class MissionExecutive:
    def __init__(
        self,
        s: RuntimeServices,
        ctx: MissionContext,
        cfg: MissionRuntimeConfig,
        stack: NavigationStack,
        gateway: CommandGateway,
    ) -> None:
        self.s, self.ctx, self.cfg, self.stack, self.gateway = s, ctx, cfg, stack, gateway
        self.ids = s.ids.child("executive")
        self.active: ActiveGoal | None = None
        self.stats = ExecutiveStats()
        self.safety_state = "NORMAL"
        self._seen_reasons: set[str] = set()
        self.transit_done = False
        self.command_rows: list[dict[str, Any]] = []
        self.latest_decision_record: UUID | None = None  # parent of goals not motivated by a specific plan

    # ------------------------------------------------------------------ goals
    def _plan_record(
        self, goal: NavigationGoal, operation: str, parents: tuple[UUID, ...], now: TimeStamp
    ) -> UUID:
        rec = ProvenanceRecord(
            record_id=self.ids.new(),
            source_type=SourceType.PLAN,
            source_ids=(goal.goal_id,),
            operation=operation,
            module=MODULE,
            model_version="mission-executive-0.1",
            timestamp=now,
            parent_records=parents,
            subject_id=goal.goal_id,
        )
        self.s.repo.put_provenance(self.s.run_id, [rec])
        return rec.record_id

    def set_goal(
        self,
        goal: NavigationGoal,
        purpose: str,
        parent: UUID | None,
        now: TimeStamp,
        plan_id: UUID | None = None,
    ) -> bool:
        parent = parent or self.latest_decision_record
        record = self._plan_record(goal, f"goal:{purpose}", () if parent is None else (parent,), now)
        traj = self.stack.set_goal(goal)
        if traj is None:
            reason = self.stack.records[-1].payload if self.stack.records else {}
            self.s.emit(
                EventType.ACTION_REJECTED,
                MODULE,
                goal.trace_id,
                {"goal_id": str(goal.goal_id), "purpose": purpose, "reason": reason},
                severity=Severity.WARNING,
            )
            self.stats.goals.append(
                {
                    "goal_id": str(goal.goal_id),
                    "purpose": purpose,
                    "accepted": False,
                    "t_s": now.time_ns / 1e9,
                    "reason": reason,
                }
            )
            return False
        self.active = ActiveGoal(goal, purpose, record, now.time_ns, plan_id)
        self.s.emit(
            EventType.GOAL_ACCEPTED,
            MODULE,
            goal.trace_id,
            {
                "goal_id": str(goal.goal_id),
                "purpose": purpose,
                "plan_id": None if plan_id is None else str(plan_id),
                "provenance_id": str(record),
            },
        )
        self.s.emit(
            EventType.TRAJECTORY_PLANNED,
            MODULE,
            goal.trace_id,
            {
                "goal_id": str(goal.goal_id),
                "trajectory_id": str(traj.trajectory_id),
                "points": len(traj.points),
                "duration_s": traj.points[-1].t_s,
                "waypoints": [
                    list(p.pose.position_m) for p in traj.points[:: max(1, len(traj.points) // 12)]
                ],
            },
        )
        self.stats.goals.append(
            {
                "goal_id": str(goal.goal_id),
                "purpose": purpose,
                "accepted": True,
                "t_s": now.time_ns / 1e9,
                "target": None if goal.target_pose is None else list(goal.target_pose.position_m),
            }
        )
        return True

    def transit_goal(self, now: TimeStamp, from_index: int = 1) -> bool:
        lane = self.ctx.transit_lane
        via = [list(p) for p in lane[from_index:-1]]
        goal = NavigationGoal(
            goal_id=self.ids.new(),
            trace_id=self.ids.new(),
            target_pose=Pose(frame_id=WORLD, position_m=lane[-1]),
            position_tolerance_m=0.5,
            orientation_tolerance_rad=0.5,
            observation_constraints={"primitive": "GO_TO", "via": via},
            risk_limit=0.3,
        )
        return self.set_goal(goal, "TRANSIT", None, now)

    def resume_transit(self, now: TimeStamp) -> bool:
        p = np.asarray(self.stack.estimator.get_state().pose.position_m)
        lane = np.asarray(self.ctx.transit_lane)
        d = lane[-1] - lane[0]
        progress = (p - lane[0]) @ d / float(d @ d)
        ahead = [i for i in range(1, len(lane)) if (lane[i] - lane[0]) @ d / float(d @ d) > progress + 0.02]
        return self.transit_goal(now, ahead[0] if ahead else len(lane) - 1)

    def hold_goal(self, now: TimeStamp, purpose: str = "HOLD") -> bool:
        pose = self.stack.estimator.get_state().pose
        goal = NavigationGoal(
            goal_id=self.ids.new(),
            trace_id=self.ids.new(),
            target_pose=Pose(
                frame_id=WORLD, position_m=pose.position_m, orientation_wxyz=pose.orientation_wxyz
            ),
            position_tolerance_m=0.5,
            orientation_tolerance_rad=0.5,
            observation_constraints={"primitive": "STATION_KEEP", "duration_s": 3600.0},
            risk_limit=0.3,
        )
        return self.set_goal(goal, purpose, None, now)

    # ------------------------------------------------------------------ planned route (Model 1 context)
    def planned_path(self) -> list[np.ndarray]:
        """Remaining path of the active motion goal's navigation trajectory, starting at the estimated pose.

        Deployment plane only: the trajectory is the navigation stack's own plan and the start point is the
        EKF estimate. Station keeping (HOLD / SAFE_HOLD) has no route.
        """
        traj = self.stack.trajectory
        if self.active is None or self.active.purpose in ("HOLD", "SAFE_HOLD") or traj is None:
            return []
        p = np.asarray(self.stack.estimator.get_state().pose.position_m, dtype=np.float64)
        pts = np.asarray([tp.pose.position_m for tp in traj.points], dtype=np.float64)
        if len(pts) < 2:
            return []
        seg = pts[1:] - pts[:-1]
        rel = p - pts[:-1]
        t = np.clip(
            np.einsum("ij,ij->i", rel, seg) / np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-12), 0, 1
        )
        k = int(np.argmin(np.linalg.norm(pts[:-1] + t[:, None] * seg - p, axis=1)))
        return [p, *list(pts[k + 1 :])]

    def planned_route(self) -> list[SpatialSupport]:
        """Corridor legs (axis-aligned WORLD boxes) along the next ``lookahead_m`` of the planned path."""
        rc = self.cfg.route
        path = self.planned_path()
        if not rc.enabled or len(path) < 2:
            return []
        dense = [path[0]]
        for a, b in itertools.pairwise(path):
            n = max(1, int(np.ceil(float(np.linalg.norm(b - a)) / (rc.leg_length_m / 4))))
            dense += [a + (b - a) * (i / n) for i in range(1, n + 1)]
        arr = np.asarray(dense)
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(arr, axis=0), axis=1))])
        keep = dist <= rc.lookahead_m
        arr, dist = arr[keep], dist[keep]
        legs: list[SpatialSupport] = []
        start = rc.skip_near_m
        while start < float(dist[-1]) - 1e-6:
            sel = arr[(dist >= start) & (dist <= start + rc.leg_length_m)]
            if len(sel):
                lo, hi = sel.min(axis=0) - rc.corridor_half_m, sel.max(axis=0) + rc.corridor_half_m
                legs.append(
                    SpatialSupport(
                        frame_id=WORLD,
                        center_m=tuple(float(v) for v in (lo + hi) / 2),
                        half_extent_m=tuple(float(v) for v in (hi - lo) / 2),
                    )
                )
            start += rc.leg_length_m
        return legs

    def replan_detour(self, blocked: list[SpatialSupport], now: TimeStamp, parent: UUID | None) -> bool:
        """REPLAN(ROUTE_BLOCKED): re-route the active transit over the blocked legs (climb, pass, descend).

        Only the remaining lane via-points are changed; the navigation stack plans the new trajectory and the
        safety supervisor / gateway still authorize every command. Non-transit goals are held instead.
        """
        if not blocked:
            return False
        if self.active is None or self.active.purpose != "TRANSIT":
            return self.hold_goal(now, "HOLD")
        path = self.planned_path()
        climb = self.cfg.route.detour_climb_m
        los = np.asarray([np.asarray(b.center_m) - np.asarray(b.half_extent_m) for b in blocked])
        his = np.asarray([np.asarray(b.center_m) + np.asarray(b.half_extent_m) for b in blocked])
        lo, hi = los.min(axis=0), his.max(axis=0)
        lane_end = np.asarray(self.ctx.transit_lane[-1], dtype=np.float64)
        d = lane_end - path[0]
        d_xy = d[:2] / max(float(np.linalg.norm(d[:2])), 1e-9)
        corners = np.asarray([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        proj = (corners - path[0][:2]) @ d_xy
        top = float(hi[2]) + climb
        before = path[0][:2] + d_xy * max(float(proj.min()) - 1.0, 0.3)
        after = path[0][:2] + d_xy * (float(proj.max()) + 1.0)
        past = float((np.append(after, top) - path[0]) @ d)
        via = [[float(before[0]), float(before[1]), top], [float(after[0]), float(after[1]), top]]
        via += [list(p) for p in self.ctx.transit_lane[1:-1] if float((np.asarray(p) - path[0]) @ d) > past]
        goal = NavigationGoal(
            goal_id=self.ids.new(),
            trace_id=self.ids.new(),
            target_pose=Pose(frame_id=WORLD, position_m=self.ctx.transit_lane[-1]),
            position_tolerance_m=0.5,
            orientation_tolerance_rad=0.5,
            observation_constraints={"primitive": "GO_TO", "via": via, "replan": "ROUTE_BLOCKED"},
            risk_limit=0.3,
        )
        return self.set_goal(goal, "TRANSIT", parent, now)

    def goal_finished(self, now_ns: int) -> bool:
        if self.active is None:
            return True
        if self.stack.status in (GoalStatus.COMPLETE, GoalStatus.REJECTED):
            return True
        if self.stack.status is GoalStatus.ARRIVED and self.active.purpose == "TRANSIT":
            return True
        return (
            self.active.purpose == "INSPECT"
            and (now_ns - self.active.started_ns) / 1e9 > self.cfg.inspection_timeout_s
        )

    # ------------------------------------------------------------------ control tick
    def control_tick(self) -> None:
        res = self.stack.step()
        self.stats.steps += 1
        state = res.assessment.state.value
        now_ns = self.s.clock_ns()
        new_reasons = set(res.assessment.reason_codes) - self._seen_reasons
        self._seen_reasons |= new_reasons
        if state != self.safety_state or new_reasons:
            event = {
                "t_s": now_ns / 1e9,
                "from": self.safety_state,
                "to": state,
                "reasons": list(res.assessment.reason_codes),
            }
            self.stats.safety_events.append(event)
            self.s.emit(
                EventType.STATE_CHANGED,
                "conrad.robotics.safety",
                self.s.run_id,
                event,
                severity=Severity.WARNING if state != "NORMAL" else Severity.INFO,
            )
            self.safety_state = state
        cmd = res.command
        goal_id = None if self.active is None else str(self.active.goal.goal_id)
        self.s.emit(
            EventType.WRENCH_REQUESTED,
            "conrad.robotics.navigation",
            cmd.trace_id,
            {
                "wrench_id": str(cmd.source_wrench_id),
                "goal_id": goal_id,
                "authorized": res.decision.authorized,
            },
        )
        if not res.decision.authorized:
            self.stats.refused_by_supervisor += 1
            return
        parent = self.active.parent_record if self.active is not None else None
        rec = ProvenanceRecord(
            record_id=self.ids.new(),
            source_type=SourceType.COMMAND,
            source_ids=(cmd.command_id, cmd.source_wrench_id),
            operation="navigation.allocate",
            module=cmd.producer,
            model_version="navigation-stack",
            timestamp=TimeStamp(time_ns=cmd.issued_time_ns, clock_domain=cmd.clock_domain),
            parent_records=() if parent is None else (parent,),
            subject_id=cmd.command_id,
        )
        self.s.provenance_buffer.append(rec)
        ack = self.gateway.submit(cmd)
        if ack.accepted:
            self.stats.accepted += 1
        else:
            self.stats.rejected += 1
            for r in ack.reason_codes:
                self.stats.rejection_reasons[r] = self.stats.rejection_reasons.get(r, 0) + 1
        self.command_rows.append(
            {
                "command_id": cmd.command_id,
                "trace_id": cmd.trace_id,
                "issued_ns": cmd.issued_time_ns,
                "accepted": ack.accepted,
                "reasons": list(ack.reason_codes),
                "payload": json.dumps(
                    {
                        "provenance_record_id": str(rec.record_id),
                        "goal_id": goal_id,
                        "thruster_commands": cmd.thruster_commands,
                    },
                    sort_keys=True,
                ),
            }
        )

    def flush(self) -> None:
        self.s.flush_provenance()
        for row in self.command_rows:
            self.s.repo.put_command(
                self.s.run_id,
                row["command_id"],
                row["trace_id"],
                row["issued_ns"],
                row["accepted"],
                row["reasons"],
                row["payload"],
            )
        self.command_rows = []
