"""Conventional 15-state error-state EKF baseline (p, v, attitude, accel bias, gyro bias).

IMU propagation; updates: pressure depth, AHRS orientation (when the IMU supplies one), WORLD
position fixes (USBL/DVL-like, visual, map constraint) and a weak body-velocity prior from the
static thrust/drag model. Frame: SIMULATION_DEFAULT (+Z up, depth = -z). Sensor lever arms ignored.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from conrad.robotics.estimation.interface import POSITION_FIX_KIND, MapConstraint, StateEstimator
from conrad.robotics.estimation.rotations import (
    quat_conj,
    quat_exp,
    quat_log,
    quat_mul,
    quat_normalize,
    quat_to_rot,
    skew,
)
from conrad.schemas.base import ConradModel
from conrad.schemas.frames import WORLD, Pose
from conrad.schemas.observation import Observation
from conrad.schemas.robot import DepthSample, HealthLevel, ImuSample, RobotConfig, RobotState, ThrusterState
from conrad.schemas.timebase import TimeStamp

LOCALIZATION_LOST = "LOCALIZATION_LOST"


class EkfConfig(ConradModel):
    gravity_mps2: float = Field(default=9.80665, gt=0)
    accel_noise: float = Field(
        default=0.03, gt=0, description="m/s^2/sqrt(Hz), inflated for unmodelled effects"
    )
    gyro_noise: float = Field(default=0.005, gt=0)
    accel_bias_walk: float = Field(default=1e-3, ge=0)
    gyro_bias_walk: float = Field(default=1e-4, ge=0)
    orientation_sigma_rad: float = Field(default=0.02, gt=0)
    velocity_prior_sigma_mps: tuple[float, float, float] = (0.25, 0.25, 0.4)
    use_velocity_prior: bool = True
    velocity_prior_correlation_s: float = Field(default=2.0, gt=0)
    fix_gate_nis: float = Field(default=16.27, gt=0, description="chi-square 3 dof, p=0.001")
    fix_rejections_for_lost: int = Field(default=5, ge=1)
    degraded_sigma_fraction: float = Field(default=0.5, gt=0, le=1)
    initial_position_sigma_m: float = Field(default=0.1, gt=0)
    initial_velocity_sigma_mps: float = Field(default=0.05, gt=0)
    initial_attitude_sigma_rad: float = Field(default=0.05, gt=0)
    stale_imu_noise_scale: float = Field(default=25.0, ge=1)


class EkfStateEstimator(StateEstimator):
    name = "EST-B0-ekf15"

    def __init__(
        self,
        robot_config: RobotConfig,
        initial_pose: Pose,
        initial_stamp: TimeStamp,
        config: EkfConfig | None = None,
    ) -> None:
        if initial_pose.frame_id != WORLD:
            raise ValueError("EKF initial pose must be expressed in WORLD")
        self.config = c = config or EkfConfig()
        self._max_sigma = float(robot_config.safety.max_pose_sigma_m.require("safety.max_pose_sigma_m"))
        self._p = np.asarray(initial_pose.position_m, dtype=np.float64)
        self._v = np.zeros(3)
        self._q = np.asarray(initial_pose.orientation_wxyz, dtype=np.float64)
        self._ba = np.zeros(3)
        self._bg = np.zeros(3)
        self._w = np.zeros(3)
        diag = [c.initial_position_sigma_m] * 3 + [c.initial_velocity_sigma_mps] * 3
        diag += [c.initial_attitude_sigma_rad] * 3 + [0.05] * 3 + [0.005] * 3
        self._P = np.diag(np.square(diag))
        self._stamp = initial_stamp
        self._last_imu_ns = -1
        self._fix_rejections = 0
        self.last_nis: float | None = None
        # static thrust/drag model (weak velocity prior)
        self._ids = tuple(t.thruster_id for t in robot_config.thrusters)
        dirs = [
            np.asarray(t.direction_body.require("direction_body"), dtype=np.float64)
            for t in robot_config.thrusters
        ]
        self._dirs = np.stack([d / np.linalg.norm(d) for d in dirs], axis=1) if dirs else np.zeros((3, 0))
        mass = float(robot_config.mass_kg.require("mass_kg"))
        self._m_tot = mass + np.asarray(robot_config.added_mass_diag.require("added_mass_diag"))[:3]
        self._d_lin = np.asarray(robot_config.linear_drag.require("linear_drag"))[:3]
        self._d_quad = np.asarray(robot_config.quadratic_drag.require("quadratic_drag"))[:3]
        self._v_model = np.zeros(3)

    # -- propagation ------------------------------------------------------------------------------
    def predict(self, imu: ImuSample, control: tuple[ThrusterState, ...] | None, dt: float) -> RobotState:
        if dt <= 0:
            raise ValueError("dt must be positive (physical elapsed seconds)")
        c = self.config
        fresh = imu.timestamp.time_ns > self._last_imu_ns
        self._last_imu_ns = max(self._last_imu_ns, imu.timestamp.time_ns)
        rot = quat_to_rot(self._q)
        g = np.array([0.0, 0.0, -c.gravity_mps2])
        if fresh:
            a_body = np.asarray(imu.linear_acceleration_mps2) - self._ba
            self._w = np.asarray(imu.angular_velocity_rps) - self._bg
            a_world = rot @ a_body + g
        else:  # no new IMU sample: hold velocity, inflate process noise
            a_body = rot.T @ (-g)
            a_world = np.zeros(3)
        self._p = self._p + self._v * dt + 0.5 * a_world * dt * dt
        self._v = self._v + a_world * dt
        self._q = quat_normalize(quat_mul(self._q, quat_exp(self._w * dt)))

        f = np.zeros((15, 15))
        f[0:3, 3:6] = np.eye(3)
        f[3:6, 6:9] = -rot @ skew(a_body)
        f[3:6, 9:12] = -rot
        f[6:9, 6:9] = -skew(self._w)
        f[6:9, 12:15] = -np.eye(3)
        phi = np.eye(15) + f * dt
        scale = 1.0 if fresh else c.stale_imu_noise_scale
        qd = np.zeros(15)
        qd[3:6] = (c.accel_noise * scale) ** 2 * dt
        qd[6:9] = (c.gyro_noise * scale) ** 2 * dt
        qd[9:12] = c.accel_bias_walk**2 * dt
        qd[12:15] = c.gyro_bias_walk**2 * dt
        self._P = phi @ self._P @ phi.T + np.diag(qd)
        if fresh:  # never advance the stamp without a measurement: staleness must stay visible
            self._stamp = imu.timestamp
        if fresh and imu.orientation_wxyz is not None:
            self._update_orientation(np.asarray(imu.orientation_wxyz, dtype=np.float64))
        if control is not None and c.use_velocity_prior:
            self._velocity_prior(control, dt)
        return self.get_state()

    def _velocity_prior(self, control: tuple[ThrusterState, ...], dt: float) -> None:
        thrust = {s.thruster_id: s.estimated_thrust_n for s in control}
        f_body = self._dirs @ np.array([thrust.get(tid, 0.0) for tid in self._ids])
        vm = self._v_model
        drag = (self._d_lin + self._d_quad * np.abs(vm)) * vm
        self._v_model = vm + dt * (f_body - drag) / self._m_tot
        rot = quat_to_rot(self._q)
        v_body = rot.T @ self._v
        h = np.zeros((3, 15))
        h[:, 3:6] = rot.T
        h[:, 6:9] = skew(v_body)
        # the model error (e.g. unknown current) is time-correlated: de-weight per-tick updates
        inflate = max(1.0, self.config.velocity_prior_correlation_s / max(dt, 1e-6))
        r = np.diag(np.square(self.config.velocity_prior_sigma_mps)) * inflate
        self._correct(self._v_model - v_body, h, r)

    # -- measurement updates -------------------------------------------------------------------------
    def _correct(self, residual: np.ndarray, h: np.ndarray, r: np.ndarray) -> float:
        s = h @ self._P @ h.T + r
        k = self._P @ h.T @ np.linalg.inv(s)
        dx = k @ residual
        self._p = self._p + dx[0:3]
        self._v = self._v + dx[3:6]
        self._q = quat_normalize(quat_mul(self._q, quat_exp(dx[6:9])))
        self._ba = self._ba + dx[9:12]
        self._bg = self._bg + dx[12:15]
        ikh = np.eye(15) - k @ h
        self._P = ikh @ self._P @ ikh.T + k @ r @ k.T
        return float(residual @ np.linalg.solve(s, residual))

    def _update_orientation(self, q_meas: np.ndarray) -> None:
        residual = quat_log(quat_mul(quat_conj(self._q), quat_normalize(q_meas)))
        h = np.zeros((3, 15))
        h[:, 6:9] = np.eye(3)
        self._correct(residual, h, np.eye(3) * self.config.orientation_sigma_rad**2)

    def update_depth(self, depth: DepthSample) -> None:
        if depth.health is HealthLevel.FAULT:
            return
        sigma = 0.05 if depth.health is HealthLevel.OK else 0.2
        h = np.zeros((1, 15))
        h[0, 2] = 1.0
        self._correct(np.array([-depth.depth_m - self._p[2]]), h, np.array([[sigma**2]]))

    def update_position_fix(self, position_world: np.ndarray, sigma_m: float) -> bool:
        """Gated WORLD position fix. Returns False (and counts an inconsistency) when rejected."""
        h = np.zeros((3, 15))
        h[:, 0:3] = np.eye(3)
        r = np.eye(3) * max(sigma_m, 1e-3) ** 2
        residual = np.asarray(position_world, dtype=np.float64) - self._p
        s = h @ self._P @ h.T + r
        nis = float(residual @ np.linalg.solve(s, residual))
        self.last_nis = nis
        if nis > self.config.fix_gate_nis:
            self._fix_rejections += 1
            return False
        self._fix_rejections = 0
        self._correct(residual, h, r)
        return True

    def _update_from_observation(self, obs: Observation) -> bool:
        ctx = obs.sensor_context
        if ctx.get("kind") != POSITION_FIX_KIND or obs.inline_values is None or len(obs.inline_values) != 3:
            return False
        if ctx.get("frame_id", WORLD) != WORLD or "sigma_m" not in ctx:
            return False
        return self.update_position_fix(np.asarray(obs.inline_values), float(ctx["sigma_m"]))

    def update_visual(self, visual: Observation) -> bool:
        return self._update_from_observation(visual)

    def update_sonar(self, sonar: Observation) -> bool:
        return self._update_from_observation(sonar)

    def update_map_constraint(self, constraint: MapConstraint) -> bool:
        if constraint.frame_id != WORLD:
            return False
        return self.update_position_fix(constraint.position_world_m, constraint.sigma_m)

    # -- output ------------------------------------------------------------------------------------------
    @property
    def position_sigma_m(self) -> float:
        return float(np.sqrt(max(np.trace(self._P[0:3, 0:3]), 0.0) / 3.0))

    @property
    def localization_lost(self) -> bool:
        return (
            self.position_sigma_m > self._max_sigma
            or self._fix_rejections >= self.config.fix_rejections_for_lost
            or not bool(np.all(np.isfinite(self._P)))
        )

    def health(self) -> HealthLevel:
        if self.localization_lost:
            return HealthLevel.FAULT
        if (
            self.position_sigma_m > self.config.degraded_sigma_fraction * self._max_sigma
            or self._fix_rejections > 0
        ):
            return HealthLevel.DEGRADED
        return HealthLevel.OK

    def health_reasons(self) -> tuple[str, ...]:
        return (LOCALIZATION_LOST,) if self.localization_lost else ()

    def get_state(self) -> RobotState:
        idx = [0, 1, 2, 6, 7, 8]
        cov = self._P[np.ix_(idx, idx)]
        v_body = quat_to_rot(self._q).T @ self._v
        q = self._q if self._q[0] >= 0 else -self._q
        return RobotState(
            timestamp=self._stamp,
            pose=Pose(
                frame_id=WORLD,
                position_m=(float(self._p[0]), float(self._p[1]), float(self._p[2])),
                orientation_wxyz=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
                covariance_6x6=tuple(float(x) for x in cov.reshape(-1)),
            ),
            linear_velocity_body_mps=(float(v_body[0]), float(v_body[1]), float(v_body[2])),
            angular_velocity_body_rps=(float(self._w[0]), float(self._w[1]), float(self._w[2])),
            estimator_health=self.health(),
            estimator_name=self.name,
        )
