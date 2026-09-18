"""CTRL-B1 cascaded PID baseline: position -> velocity -> force, attitude PD -> torque (body frame)."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from pydantic import Field

from conrad.robotics.estimation.rotations import quat_error_body, quat_to_rot
from conrad.schemas.base import ConradModel
from conrad.schemas.frames import ROBOT
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import RobotConfig, RobotState, WrenchCommand
from conrad.schemas.timebase import TimeStamp

Gain3 = tuple[float, float, float]


class ControlConfig(ConradModel):
    """Tuned on the SYNTHETIC_ONLY sim_reference vehicle; physical gains need re-identification."""

    kp_position: Gain3 = (0.9, 0.9, 1.0)
    kp_velocity: Gain3 = (60.0, 80.0, 90.0)
    ki_velocity: Gain3 = (25.0, 30.0, 35.0)
    kd_velocity: Gain3 = (0.0, 0.0, 0.0)
    kp_attitude: Gain3 = (4.0, 4.0, 4.0)
    kd_attitude: Gain3 = (1.5, 1.5, 1.5)
    integral_limit_n: Gain3 = (25.0, 25.0, 25.0)
    force_limit_n: Gain3 = (80.0, 80.0, 100.0)
    torque_limit_nm: Gain3 = (8.0, 8.0, 12.0)
    drag_feedforward: bool = True
    buoyancy_feedforward: bool = True
    water_density_kgm3: float = Field(
        default=1025.0, gt=0, description="environment assumption (SYNTHETIC_ONLY)"
    )
    gravity_mps2: float = Field(default=9.80665, gt=0)


@dataclass(frozen=True)
class ControlReference:
    position_world_m: np.ndarray
    orientation_wxyz: np.ndarray
    velocity_world_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    speed_limit_mps: float | None = None


class CascadedPidController:
    name = "CTRL-B1-cascaded-pid"

    def __init__(
        self, robot_config: RobotConfig, ids: IdFactory, config: ControlConfig | None = None
    ) -> None:
        self.config = config or ControlConfig()
        self._ids = ids
        c = self.config
        self._max_speed = float(robot_config.safety.max_speed_mps.require("safety.max_speed_mps"))
        mass = float(robot_config.mass_kg.require("mass_kg"))
        volume = float(robot_config.displaced_volume_m3.require("displaced_volume_m3"))
        self._net_buoyancy_n = (c.water_density_kgm3 * volume - mass) * c.gravity_mps2
        self._d_lin = np.asarray(robot_config.linear_drag.require("linear_drag"), dtype=np.float64)[:3]
        self._d_quad = np.asarray(robot_config.quadratic_drag.require("quadratic_drag"), dtype=np.float64)[:3]
        self._integral = np.zeros(3)
        self._prev_err_v = np.zeros(3)
        self.last_velocity_setpoint_body = np.zeros(3)

    def reset(self) -> None:
        self._integral[:] = 0.0
        self._prev_err_v[:] = 0.0

    def compute(
        self, state: RobotState, reference: ControlReference, dt: float, trace_id: UUID, timestamp: TimeStamp
    ) -> WrenchCommand:
        c = self.config
        p = np.asarray(state.pose.position_m)
        q = np.asarray(state.pose.orientation_wxyz)
        rot = quat_to_rot(q)
        v_body = (
            np.zeros(3)
            if state.linear_velocity_body_mps is None
            else np.asarray(state.linear_velocity_body_mps)
        )
        w_body = (
            np.zeros(3)
            if state.angular_velocity_body_rps is None
            else np.asarray(state.angular_velocity_body_rps)
        )

        # outer loop: position error (world) -> velocity setpoint, limited by the safety envelope
        v_set_world = reference.velocity_world_mps + rot @ (
            np.asarray(c.kp_position) * (rot.T @ (reference.position_world_m - p))
        )
        limit = (
            self._max_speed
            if reference.speed_limit_mps is None
            else min(self._max_speed, reference.speed_limit_mps)
        )
        speed = float(np.linalg.norm(v_set_world))
        if speed > limit:
            v_set_world = v_set_world * (limit / speed)
        v_set = rot.T @ v_set_world
        self.last_velocity_setpoint_body = v_set

        # inner loop: body velocity PID with conditional-integration anti-windup
        err = v_set - v_body
        deriv = (err - self._prev_err_v) / dt if dt > 0 else np.zeros(3)
        self._prev_err_v = err
        feedforward = np.zeros(3)
        if c.drag_feedforward:
            feedforward += (self._d_lin + self._d_quad * np.abs(v_set)) * v_set
        if c.buoyancy_feedforward:
            feedforward += rot.T @ np.array([0.0, 0.0, -self._net_buoyancy_n])
        unsat = (
            np.asarray(c.kp_velocity) * err
            + np.asarray(c.ki_velocity) * self._integral
            + np.asarray(c.kd_velocity) * deriv
            + feedforward
        )
        f_lim = np.asarray(c.force_limit_n)
        force = np.clip(unsat, -f_lim, f_lim)
        winding = (unsat != force) & (np.sign(err) == np.sign(unsat))
        self._integral = np.where(winding, self._integral, self._integral + err * dt)
        i_lim = np.asarray(c.integral_limit_n) / np.maximum(np.asarray(c.ki_velocity), 1e-9)
        self._integral = np.clip(self._integral, -i_lim, i_lim)

        # attitude PD on the quaternion error (rotation vector in the body frame)
        e_rot = quat_error_body(reference.orientation_wxyz, q)
        t_lim = np.asarray(c.torque_limit_nm)
        torque = np.clip(
            np.asarray(c.kp_attitude) * e_rot - np.asarray(c.kd_attitude) * w_body, -t_lim, t_lim
        )

        return WrenchCommand(
            command_id=self._ids.new(),
            trace_id=trace_id,
            timestamp=timestamp,
            frame_id=ROBOT,
            force_n=(float(force[0]), float(force[1]), float(force[2])),
            torque_nm=(float(torque[0]), float(torque[1]), float(torque[2])),
        )
