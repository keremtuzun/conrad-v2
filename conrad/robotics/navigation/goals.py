"""NAV-GOAL-01 Goal Manager + NAV-INSPECT-01 primitives: NavigationGoal -> executable objective.

Primitive selection and parameters travel in ``NavigationGoal.observation_constraints`` (domain-local
vocabulary; the frozen schema has no dedicated field):

* ``{"primitive": "GO_TO"}`` (default when ``target_pose`` is given): waypoint(s); optional
  ``"via": [[x,y,z], ...]`` WORLD intermediate waypoints.
* ``{"primitive": "STATION_KEEP", "duration_s": T}``: hold ``target_pose``.
* ``{"primitive": "PIPELINE_FOLLOW", "polyline": [[x,y,z], ...], "standoff_m": d,
  "standoff_direction": [0,0,1]}``: follow the polyline offset by ``d`` along the direction.
* ``{"primitive": "INSPECT", "standoff_m": d, "incidence_range_rad": [lo, hi],
  "surface_normal": [nx,ny,nz]}`` with ``target_region``: face the region centre from a viewpoint at
  ``d`` whose view ray makes an incidence angle inside the range (forward-looking, level sensor).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from conrad.robotics.estimation.rotations import quat_from_yaw, yaw_of
from conrad.robotics.trajectory.generator import YawMode
from conrad.schemas.decision import NavigationGoal
from conrad.schemas.frames import WORLD, FrameError


class ObjectiveKind(str, Enum):
    GO_TO = "GO_TO"
    STATION_KEEP = "STATION_KEEP"
    PIPELINE_FOLLOW = "PIPELINE_FOLLOW"
    INSPECT = "INSPECT"


class GoalRejectedError(ValueError):
    """The goal cannot be turned into an executable objective; ``reason_code`` says why."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code


@dataclass(frozen=True)
class NavigationObjective:
    kind: ObjectiveKind
    goal: NavigationGoal
    waypoints: np.ndarray  # WORLD, excluding the start pose
    yaw_mode: YawMode
    final_orientation_wxyz: np.ndarray | None = None
    look_at: np.ndarray | None = None
    hold_duration_s: float = 0.0
    route_is_prescribed: bool = False  # True: follow waypoints literally (no global replanning)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def final_position(self) -> np.ndarray:
        return np.asarray(self.waypoints[-1])


def _vec3(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise GoalRejectedError("MALFORMED_CONSTRAINT", f"{name} must be three finite numbers")
    return arr


def incidence_angle(viewpoint: np.ndarray, target: np.ndarray, normal: np.ndarray) -> float:
    """Angle between the reversed view ray and the surface normal (0 = head-on)."""
    ray = target - viewpoint
    n = normal / np.linalg.norm(normal)
    c = float(-ray @ n) / max(float(np.linalg.norm(ray)), 1e-9)
    return math.acos(max(-1.0, min(1.0, c)))


class GoalManager:
    name = "NAV-GOAL-01"

    def to_objective(self, goal: NavigationGoal, current_position: np.ndarray) -> NavigationObjective:
        oc = dict(goal.observation_constraints)
        kind_name = str(oc.get("primitive", "INSPECT" if goal.target_pose is None else "GO_TO"))
        try:
            kind = ObjectiveKind(kind_name)
        except ValueError as exc:
            raise GoalRejectedError("UNSUPPORTED_PRIMITIVE", kind_name) from exc
        for frame in (goal.target_pose, goal.target_region):
            if frame is not None and frame.frame_id != WORLD:
                raise FrameError(f"navigation goal frame {frame.frame_id!r} is not WORLD; transform it first")
        if kind is ObjectiveKind.INSPECT:
            return self._inspect(goal, oc, np.asarray(current_position, dtype=np.float64))
        if kind is ObjectiveKind.PIPELINE_FOLLOW:
            return self._pipeline(goal, oc)
        if goal.target_pose is None:
            raise GoalRejectedError("MISSING_TARGET_POSE", f"{kind.value} requires target_pose")
        target = np.asarray(goal.target_pose.position_m, dtype=np.float64)
        q = np.asarray(goal.target_pose.orientation_wxyz, dtype=np.float64)
        if kind is ObjectiveKind.STATION_KEEP:
            duration = float(oc.get("duration_s", 0.0))
            return NavigationObjective(
                kind, goal, target[None, :], YawMode.FIXED, q, hold_duration_s=duration
            )
        via = [_vec3(v, "via") for v in oc.get("via", [])]
        return NavigationObjective(kind, goal, np.stack([*via, target]), YawMode.FACE_TRAVEL, q)

    def _pipeline(self, goal: NavigationGoal, oc: dict[str, Any]) -> NavigationObjective:
        poly = np.asarray(oc.get("polyline", []), dtype=np.float64)
        if poly.ndim != 2 or poly.shape[0] < 2 or poly.shape[1] != 3:
            raise GoalRejectedError("MALFORMED_CONSTRAINT", "PIPELINE_FOLLOW needs a polyline of >= 2 points")
        standoff = float(oc.get("standoff_m", 0.0))
        direction = _vec3(oc.get("standoff_direction", [0.0, 0.0, 1.0]), "standoff_direction")
        direction = direction / np.linalg.norm(direction)
        path = poly + standoff * direction[None, :]
        return NavigationObjective(
            ObjectiveKind.PIPELINE_FOLLOW,
            goal,
            path,
            YawMode.FACE_TRAVEL,
            route_is_prescribed=True,
            params={"polyline": poly, "standoff_m": standoff, "standoff_direction": direction},
        )

    def _inspect(self, goal: NavigationGoal, oc: dict[str, Any], current: np.ndarray) -> NavigationObjective:
        if goal.target_region is None:
            raise GoalRejectedError("MISSING_TARGET_REGION", "INSPECT requires target_region")
        centre = np.asarray(goal.target_region.center_m, dtype=np.float64)
        if "standoff_m" not in oc:
            raise GoalRejectedError("MALFORMED_CONSTRAINT", "INSPECT requires standoff_m")
        standoff = float(oc["standoff_m"])
        lo, hi = (float(x) for x in oc.get("incidence_range_rad", (0.0, math.pi / 2)))
        normal = oc.get("surface_normal")
        if normal is None:  # no surface model: view from the side the robot is on
            away = current - centre
            away[2] = 0.0
            n = away if np.linalg.norm(away) > 1e-6 else np.array([1.0, 0.0, 0.0])
        else:
            n = _vec3(normal, "surface_normal")
        n = n / np.linalg.norm(n)
        if abs(n[2]) > 0.95:
            raise GoalRejectedError(
                "VIEWPOINT_INFEASIBLE", "vertical surface normal needs a tilting sensor; mount is level"
            )
        nh = np.array([n[0], n[1], 0.0]) / math.hypot(n[0], n[1])
        base = math.atan2(nh[1], nh[0])
        rel = current - centre
        side = math.atan2(rel[1], rel[0])
        # keep the robot's current bearing if its incidence is admissible, otherwise clamp into range
        offset = (side - base + math.pi) % (2 * math.pi) - math.pi
        magnitude = min(max(abs(offset), lo), hi)
        bearing = base + math.copysign(magnitude, offset if offset != 0 else 1.0)
        viewpoint = centre + standoff * np.array([math.cos(bearing), math.sin(bearing), 0.0])
        viewpoint[2] = centre[2]
        face = quat_from_yaw(math.atan2(centre[1] - viewpoint[1], centre[0] - viewpoint[0]))
        return NavigationObjective(
            ObjectiveKind.INSPECT,
            goal,
            viewpoint[None, :],
            YawMode.LOOK_AT,
            final_orientation_wxyz=face,
            look_at=centre,
            hold_duration_s=float(oc.get("dwell_s", 0.0)),
            params={
                "standoff_m": standoff,
                "incidence_range_rad": (lo, hi),
                "surface_normal": n,
                "planned_incidence_rad": incidence_angle(viewpoint, centre, n),
                "final_yaw": yaw_of(face),
            },
        )
