"""Dynamically feasible trajectories: arc-length speed profile with speed/accel/corner limits (NAV-TRAJ-01)."""

from __future__ import annotations

import math
from enum import Enum
from uuid import UUID

import numpy as np
from pydantic import Field

from conrad.robotics.control.pid import ControlReference
from conrad.robotics.estimation.rotations import quat_from_yaw, quat_normalize, wrap_pi, yaw_of
from conrad.schemas.base import ConradModel
from conrad.schemas.decision import Trajectory, TrajectoryPoint
from conrad.schemas.frames import WORLD, Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import RobotConfig

PLANNER_NAME = "NAV-TRAJ-01-trapezoidal"


class YawMode(str, Enum):
    FACE_TRAVEL = "FACE_TRAVEL"
    FIXED = "FIXED"
    LOOK_AT = "LOOK_AT"


class TrajectoryConfig(ConradModel):
    cruise_speed_fraction: float = Field(default=0.5, gt=0, le=1, description="of safety.max_speed_mps")
    accel_limit_mps2: float | None = Field(default=None, gt=0)
    accel_authority_fraction: float = Field(default=0.1, gt=0, le=1)
    arc_step_m: float = Field(default=0.05, gt=0)
    sample_dt_s: float = Field(default=0.2, gt=0)
    corner_floor_fraction: float = Field(default=0.15, gt=0, le=1)
    max_yaw_rate_rps: float = Field(default=0.5, gt=0)


def derive_accel_limit(robot_config: RobotConfig, fraction: float) -> float:
    """Conservative linear acceleration bound from thruster geometry and total (rigid + added) mass."""
    mass = float(robot_config.mass_kg.require("mass_kg"))
    added = np.asarray(robot_config.added_mass_diag.require("added_mass_diag"), dtype=np.float64)[:3]
    authority = np.zeros(3)
    for t in robot_config.thrusters:
        d = np.asarray(t.direction_body.require("direction_body"), dtype=np.float64)
        d = d / np.linalg.norm(d)
        lim = min(
            float(t.max_forward_thrust_n.require("max_forward_thrust_n")),
            float(t.max_reverse_thrust_n.require("max_reverse_thrust_n")),
        )
        authority += np.abs(d) * lim
    if not np.all(authority > 0):
        raise ValueError("vehicle lacks translational authority on some axis; supply accel_limit_mps2")
    return float(fraction * np.min(authority / (mass + added)))


class TrajectoryGenerator:
    def __init__(
        self, robot_config: RobotConfig, ids: IdFactory, config: TrajectoryConfig | None = None
    ) -> None:
        self.config = c = config or TrajectoryConfig()
        self._ids = ids
        self.max_speed = float(robot_config.safety.max_speed_mps.require("safety.max_speed_mps"))
        self.accel_limit = c.accel_limit_mps2 or derive_accel_limit(robot_config, c.accel_authority_fraction)

    def hold(
        self, position: np.ndarray, orientation_wxyz: np.ndarray, goal_id: UUID, trace_id: UUID
    ) -> Trajectory:
        return Trajectory(
            trajectory_id=self._ids.new(),
            goal_id=goal_id,
            trace_id=trace_id,
            frame_id=WORLD,
            points=(TrajectoryPoint(t_s=0.0, pose=_pose(position, orientation_wxyz)),),
            max_speed_mps=self.max_speed,
            planner=PLANNER_NAME + ":hold",
        )

    def generate(
        self,
        waypoints: np.ndarray,
        goal_id: UUID,
        trace_id: UUID,
        start_yaw: float,
        yaw_mode: YawMode = YawMode.FACE_TRAVEL,
        final_orientation_wxyz: np.ndarray | None = None,
        look_at: np.ndarray | None = None,
        speed_limit_mps: float | None = None,
    ) -> Trajectory:
        c = self.config
        pts = _dedupe(np.asarray(waypoints, dtype=np.float64))
        v_max = min(self.max_speed * c.cruise_speed_fraction, speed_limit_mps or math.inf, self.max_speed)
        if len(pts) < 2:
            q = final_orientation_wxyz if final_orientation_wxyz is not None else quat_from_yaw(start_yaw)
            return self.hold(pts[0], q, goal_id, trace_id)
        path, vertex_limit = _resample(pts, c.arc_step_m, v_max, c.corner_floor_fraction)
        ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
        v = vertex_limit.copy()
        v[0] = v[-1] = 0.0
        for i in range(1, len(v)):
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * self.accel_limit * ds[i - 1]))
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * self.accel_limit * ds[i]))
        t = np.concatenate([[0.0], np.cumsum(2 * ds / np.maximum(v[:-1] + v[1:], 1e-6))])
        ts = np.append(np.arange(0.0, t[-1], c.sample_dt_s), t[-1])
        pos = np.stack([np.interp(ts, t, path[:, k]) for k in range(3)], axis=1)
        speed = np.interp(ts, t, v)
        tangent = np.gradient(path, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        direction = np.stack([np.interp(ts, t, tangent[:, k]) for k in range(3)], axis=1)
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        vel = direction * speed[:, None]
        yaw_final = None if final_orientation_wxyz is None else yaw_of(final_orientation_wxyz)
        yaws = self._yaw_profile(ts, pos, vel, start_yaw, yaw_mode, yaw_final, look_at)
        points = []
        for i in range(len(ts)):
            points.append(
                TrajectoryPoint(
                    t_s=float(ts[i]),
                    pose=_pose(pos[i], quat_from_yaw(float(yaws[i]))),
                    linear_velocity_mps=(float(vel[i, 0]), float(vel[i, 1]), float(vel[i, 2])),
                )
            )
        return Trajectory(
            trajectory_id=self._ids.new(),
            goal_id=goal_id,
            trace_id=trace_id,
            frame_id=WORLD,
            points=tuple(points),
            max_speed_mps=v_max,
            planner=PLANNER_NAME,
        )

    def _yaw_profile(
        self,
        ts: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
        start_yaw: float,
        mode: YawMode,
        yaw_final: float | None,
        look_at: np.ndarray | None,
    ) -> np.ndarray:
        desired = np.full(len(ts), start_yaw)
        if mode is YawMode.FACE_TRAVEL:
            last = start_yaw
            for i in range(len(ts)):
                if math.hypot(vel[i, 0], vel[i, 1]) > 0.02:
                    last = math.atan2(vel[i, 1], vel[i, 0])
                desired[i] = last
        elif mode is YawMode.LOOK_AT and look_at is not None:
            d = np.asarray(look_at)[None, :2] - pos[:, :2]
            desired = np.arctan2(d[:, 1], d[:, 0])
        elif mode is YawMode.FIXED and yaw_final is not None:
            desired[:] = yaw_final
        out = np.empty(len(ts))
        current = start_yaw
        for i in range(len(ts)):
            step = self.config.max_yaw_rate_rps * (ts[i] - ts[i - 1]) if i else 0.0
            current = current + float(np.clip(wrap_pi(desired[i] - current), -step, step))
            out[i] = current
        return out


def _pose(position: np.ndarray, q: np.ndarray) -> Pose:
    return Pose(
        frame_id=WORLD,
        position_m=(float(position[0]), float(position[1]), float(position[2])),
        orientation_wxyz=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
    )


def _dedupe(pts: np.ndarray) -> np.ndarray:
    keep = [0] + [i for i in range(1, len(pts)) if np.linalg.norm(pts[i] - pts[i - 1]) > 1e-6]
    return pts[keep]


def _resample(pts: np.ndarray, step: float, v_max: float, floor: float) -> tuple[np.ndarray, np.ndarray]:
    out, limit = [pts[0]], [v_max]
    for i in range(1, len(pts)):
        seg = pts[i] - pts[i - 1]
        n = max(1, math.ceil(float(np.linalg.norm(seg)) / step))
        for k in range(1, n + 1):
            out.append(pts[i - 1] + seg * (k / n))
            limit.append(v_max)
        if i < len(pts) - 1:
            a, b = seg / np.linalg.norm(seg), pts[i + 1] - pts[i]
            cos_half = math.sqrt(max(0.0, (1.0 + float(a @ (b / np.linalg.norm(b)))) / 2.0))
            limit[-1] = v_max * max(floor, cos_half**4)
    return np.array(out), np.array(limit)


class TrajectorySampler:
    """Time lookup on a Trajectory -> controller reference (nlerp attitude, clamped at the end)."""

    def __init__(self, trajectory: Trajectory) -> None:
        self.trajectory = trajectory
        self._t = np.array([p.t_s for p in trajectory.points])
        self._p = np.array([p.pose.position_m for p in trajectory.points])
        self._q = np.array([p.pose.orientation_wxyz for p in trajectory.points])
        self._v = np.array([p.linear_velocity_mps for p in trajectory.points])

    @property
    def duration_s(self) -> float:
        return float(self._t[-1])

    def sample(self, t_s: float) -> ControlReference:
        t = float(np.clip(t_s, 0.0, self._t[-1]))
        i = int(np.clip(np.searchsorted(self._t, t, side="right") - 1, 0, len(self._t) - 1))
        j = min(i + 1, len(self._t) - 1)
        span = self._t[j] - self._t[i]
        a = 0.0 if span <= 0 else (t - self._t[i]) / span
        qi, qj = self._q[i], self._q[j]
        if float(qi @ qj) < 0:
            qj = -qj
        done = t_s >= self._t[-1]
        return ControlReference(
            position_world_m=(1 - a) * self._p[i] + a * self._p[j],
            orientation_wxyz=quat_normalize((1 - a) * qi + a * qj),
            velocity_world_mps=np.zeros(3) if done else (1 - a) * self._v[i] + a * self._v[j],
            speed_limit_mps=self.trajectory.max_speed_mps,
        )
