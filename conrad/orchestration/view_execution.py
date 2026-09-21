"""View execution ledger: did the mission actually fly the view MCBR asked for? DEPLOYMENT PLANE.

MCBR outputs an ``ObservationPlan``; navigation moves the robot. Between the two there is a gap that the
V3 diagnostics measured and nothing recorded: about 45 % of accepted inspection goals never came within
0.5 m of the commanded pose, and nothing told the planner so. This module closes the record, not the loop:
every accepted view goal opens a ``ViewExecutionRecord``, every control tick updates the closest approach
and the aim error, and the goal is closed with exactly one reason code.

It holds no truth. Positions come from the runtime's own EKF estimate, the commanded pose comes from the
plan, and the tolerances come from the ``NavigationGoal`` the executive accepted.

``AbandonedView`` is the belief-side summary a V4 planner may read back through
``PlanningRequest.abandoned_views``; it carries no truth either.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import UUID

import numpy as np

from conrad.active.gap import AbandonedView

# ---------------------------------------------------------------------------------------- reason codes
FLOWN = "FLOWN"
"""Reached the commanded pose inside the declared tolerance and held it for the declared dwell."""
ARRIVED_DWELL_SHORT = "ARRIVED_DWELL_SHORT"
"""Reached the commanded pose inside tolerance, but the goal ended before the dwell was complete."""
ABANDONED_ROUTE_BLOCKED = "ABANDONED_ROUTE_BLOCKED"
"""A REPLAN(ROUTE_BLOCKED) decision pre-empted the view before it was flown."""
ABANDONED_SAFE_HOLD = "ABANDONED_SAFE_HOLD"
ABANDONED_PREEMPTED = "ABANDONED_PREEMPTED"
"""Some other goal replaced the view goal while it was still executing."""
ABANDONED_TIMEOUT = "ABANDONED_TIMEOUT"
"""``inspection_timeout_s`` elapsed with the vehicle still in transit to the view."""
ABANDONED_TRAJECTORY_LOST = "ABANDONED_TRAJECTORY_LOST"
"""A later goal was rejected by the navigation stack, which cleared the objective of the running view."""
ABANDONED_NAV_REJECTED = "ABANDONED_NAV_REJECTED"
"""The navigation stack refused the view goal itself (no record is opened; counted at goal time)."""
ABANDONED_MISSION_END = "ABANDONED_MISSION_END"
"""The mission clock ran out with the view still executing."""
COMPLETED_OFF_POSE = "COMPLETED_OFF_POSE"
"""Navigation declared the goal COMPLETE while the vehicle was outside the declared pose tolerance."""

ABANDON_REASONS = (
    ABANDONED_ROUTE_BLOCKED,
    ABANDONED_SAFE_HOLD,
    ABANDONED_PREEMPTED,
    ABANDONED_TIMEOUT,
    ABANDONED_TRAJECTORY_LOST,
    ABANDONED_NAV_REJECTED,
    ABANDONED_MISSION_END,
    COMPLETED_OFF_POSE,
    ARRIVED_DWELL_SHORT,
)
ALL_REASONS = (FLOWN, *ABANDON_REASONS)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class ViewCommand:
    """What the executive was asked to fly, in the deployment plane only."""

    plan_id: UUID | None
    action_id: UUID | None
    need_id: UUID | None
    position_m: tuple[float, float, float]
    yaw_rad: float
    aim_point_m: tuple[float, float, float]
    position_tolerance_m: float
    orientation_tolerance_rad: float
    dwell_s: float
    mount_yaw_rad: float = 0.0
    predicted_visibility: float | None = None


@dataclass
class ViewExecutionRecord:
    plan_id: str | None
    action_id: str | None
    goal_id: str
    purpose: str
    t_start_s: float
    commanded_position_m: list[float]
    commanded_yaw_rad: float
    aim_point_m: list[float]
    position_tolerance_m: float
    orientation_tolerance_rad: float
    dwell_s: float
    start_distance_m: float
    predicted_visibility: float | None = None
    t_end_s: float | None = None
    outcome: str | None = None
    closest_approach_m: float = float("inf")
    t_closest_s: float | None = None
    realised_position_m: list[float] | None = None
    """Estimated position at the point of closest approach to the commanded pose."""
    yaw_error_rad: float | None = None
    """|realised vehicle yaw - commanded vehicle yaw| at the closest approach."""
    view_direction_error_rad: float | None = None
    """Angle at the aim point between the commanded and the realised view directions."""
    boresight_error_rad: float | None = None
    """Angle between the realised sensor boresight and the direction to the aim point."""
    t_within_tolerance_s: float | None = None
    time_within_tolerance_s: float = 0.0
    """Seconds spent inside BOTH the declared pose tolerance and the declared aim tolerance."""
    t_within_position_tolerance_s: float | None = None
    time_within_position_tolerance_s: float = 0.0
    min_boresight_error_in_position_rad: float | None = None
    """Best aim achieved while inside the pose tolerance (None = never inside it)."""
    ticks: int = 0
    final_status: str | None = None

    @property
    def reached(self) -> bool:
        """Inside the declared POSE tolerance at some point (aim not required)."""
        return self.t_within_position_tolerance_s is not None

    @property
    def aimed(self) -> bool:
        """Inside the declared pose AND aim tolerance at some point."""
        return self.t_within_tolerance_s is not None

    @property
    def flown(self) -> bool:
        return self.outcome == FLOWN

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["reached_tolerance"] = self.reached
        if out["closest_approach_m"] == float("inf"):
            out["closest_approach_m"] = None
        return out


@dataclass
class ViewLedger:
    """One open record at a time; closed records are kept in order."""

    records: list[ViewExecutionRecord] = field(default_factory=list)
    open_record: ViewExecutionRecord | None = None
    _command: ViewCommand | None = None

    # ------------------------------------------------------------------ lifecycle
    def open(self, command: ViewCommand, goal_id: UUID, purpose: str, now_ns: int, position: Any) -> None:
        p = np.asarray(position, dtype=np.float64)
        self._command = command
        self.open_record = ViewExecutionRecord(
            plan_id=None if command.plan_id is None else str(command.plan_id),
            action_id=None if command.action_id is None else str(command.action_id),
            goal_id=str(goal_id),
            purpose=purpose,
            t_start_s=now_ns / 1e9,
            commanded_position_m=[float(x) for x in command.position_m],
            commanded_yaw_rad=float(command.yaw_rad),
            aim_point_m=[float(x) for x in command.aim_point_m],
            position_tolerance_m=float(command.position_tolerance_m),
            orientation_tolerance_rad=float(command.orientation_tolerance_rad),
            dwell_s=float(command.dwell_s),
            start_distance_m=float(np.linalg.norm(p - np.asarray(command.position_m, dtype=np.float64))),
            predicted_visibility=command.predicted_visibility,
        )

    def update(self, now_ns: int, position: Any, yaw_rad: float, dt_s: float) -> None:
        rec, cmd = self.open_record, self._command
        if rec is None or cmd is None:
            return
        rec.ticks += 1
        p = np.asarray(position, dtype=np.float64)
        target = np.asarray(cmd.position_m, dtype=np.float64)
        d = float(np.linalg.norm(p - target))
        boresight = _boresight_error(cmd.aim_point_m, p, yaw_rad + cmd.mount_yaw_rad)
        if d < rec.closest_approach_m:
            rec.closest_approach_m = d
            rec.t_closest_s = now_ns / 1e9
            rec.realised_position_m = [float(x) for x in p]
            rec.yaw_error_rad = abs(wrap_angle(float(yaw_rad) - float(cmd.yaw_rad)))
            rec.view_direction_error_rad = _direction_error(cmd.aim_point_m, cmd.position_m, p)
            rec.boresight_error_rad = boresight
        if d <= rec.position_tolerance_m:
            if rec.t_within_position_tolerance_s is None:
                rec.t_within_position_tolerance_s = now_ns / 1e9
            rec.time_within_position_tolerance_s += max(dt_s, 0.0)
            if rec.min_boresight_error_in_position_rad is None:
                rec.min_boresight_error_in_position_rad = boresight
            else:
                rec.min_boresight_error_in_position_rad = min(
                    rec.min_boresight_error_in_position_rad, boresight
                )
            if boresight <= rec.orientation_tolerance_rad:
                if rec.t_within_tolerance_s is None:
                    rec.t_within_tolerance_s = now_ns / 1e9
                rec.time_within_tolerance_s += max(dt_s, 0.0)

    def close(self, now_ns: int, outcome: str, final_status: str | None = None) -> ViewExecutionRecord | None:
        rec = self.open_record
        if rec is None:
            return None
        rec.t_end_s = now_ns / 1e9
        rec.outcome = outcome
        rec.final_status = final_status
        self.records.append(rec)
        self.open_record, self._command = None, None
        return rec

    # ------------------------------------------------------------------ views
    def abandoned(self) -> tuple[AbandonedView, ...]:
        out = []
        for r in self.records:
            if r.outcome in ABANDON_REASONS and r.outcome is not None:
                out.append(
                    AbandonedView(
                        position_m=(
                            r.commanded_position_m[0],
                            r.commanded_position_m[1],
                            r.commanded_position_m[2],
                        ),
                        aim_point_m=(r.aim_point_m[0], r.aim_point_m[1], r.aim_point_m[2]),
                        reason=r.outcome,
                        closest_approach_m=(
                            r.closest_approach_m if math.isfinite(r.closest_approach_m) else float("nan")
                        ),
                        plan_id=UUID(r.plan_id) if r.plan_id else None,
                    )
                )
        return tuple(out)

    def to_json(self) -> list[dict[str, Any]]:
        rows = [r.to_json() for r in self.records]
        if self.open_record is not None:
            rows.append(self.open_record.to_json())
        return rows


def summarize(records: list[ViewExecutionRecord]) -> dict[str, Any]:
    """Deployment-side execution summary of one mission. Reported whatever it says."""
    n = len(records)
    by_reason: dict[str, int] = {}
    for r in records:
        key = r.outcome or "OPEN"
        by_reason[key] = by_reason.get(key, 0) + 1
    reached = [r for r in records if r.reached]
    aimed = [r for r in records if r.aimed]
    finite = [r.closest_approach_m for r in records if math.isfinite(r.closest_approach_m)]
    aims = [r.view_direction_error_rad for r in records if r.view_direction_error_rad is not None]
    bores = [r.boresight_error_rad for r in records if r.boresight_error_rad is not None]
    flown = sum(1 for r in records if r.flown)
    return {
        "accepted_views": n,
        "flown": flown,
        "reached_tolerance": len(reached),
        "reached_pose_and_aim_tolerance": len(aimed),
        "fraction_flown": (flown / n) if n else None,
        "fraction_reached_tolerance": (len(reached) / n) if n else None,
        "fraction_reached_pose_and_aim": (len(aimed) / n) if n else None,
        "outcomes": dict(sorted(by_reason.items())),
        "mean_closest_approach_m": float(np.mean(finite)) if finite else None,
        "median_closest_approach_m": float(np.median(finite)) if finite else None,
        "mean_view_direction_error_rad": float(np.mean(aims)) if aims else None,
        "mean_boresight_error_rad": float(np.mean(bores)) if bores else None,
        "mean_time_within_tolerance_s": float(np.mean([r.time_within_tolerance_s for r in records]))
        if n
        else None,
    }


def _direction_error(aim: tuple[float, float, float], commanded: Any, realised: Any) -> float:
    a = np.asarray(aim, dtype=np.float64)
    u = np.asarray(commanded, dtype=np.float64) - a
    v = np.asarray(realised, dtype=np.float64) - a
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return float(math.acos(max(-1.0, min(1.0, float(u @ v) / (nu * nv)))))


def _boresight_error(aim: tuple[float, float, float], realised: Any, sensor_yaw_rad: float) -> float:
    a = np.asarray(aim, dtype=np.float64)
    p = np.asarray(realised, dtype=np.float64)
    d = a - p
    if float(np.linalg.norm(d[:2])) < 1e-9:
        return 0.0
    return abs(wrap_angle(float(math.atan2(d[1], d[0])) - float(sensor_yaw_rad)))


__all__ = [
    "ABANDONED_MISSION_END",
    "ABANDONED_NAV_REJECTED",
    "ABANDONED_PREEMPTED",
    "ABANDONED_ROUTE_BLOCKED",
    "ABANDONED_SAFE_HOLD",
    "ABANDONED_TIMEOUT",
    "ABANDONED_TRAJECTORY_LOST",
    "ABANDON_REASONS",
    "ALL_REASONS",
    "ARRIVED_DWELL_SHORT",
    "COMPLETED_OFF_POSE",
    "FLOWN",
    "AbandonedView",
    "ViewCommand",
    "ViewExecutionRecord",
    "ViewLedger",
    "summarize",
    "wrap_angle",
]
