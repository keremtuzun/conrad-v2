"""18-state error-state EKF (p, v, attitude, accel bias, gyro bias, prior mismatch) with consistency monitoring.

IMU propagation; updates: pressure depth, AHRS orientation (when the IMU supplies one), WORLD position
fixes (USBL/DVL-like, visual, map constraint) and a body-velocity prior from the static thrust/drag model.
Frame: SIMULATION_DEFAULT (+Z up, depth = -z). Sensor lever arms ignored.

Why the prior-mismatch state exists (NAV-007 finding): the thrust/drag model predicts the WATER-relative
velocity. Any water current or drag-model error is a slowly varying (time-correlated) velocity offset.
Treating it as white noise lets repeated prior updates pull the velocity estimate onto a biased value
while the covariance shrinks, so the position covariance grows like sqrt(t) although the true error
grows linearly in t. Here the offset is an explicit random-walk WORLD velocity state ``c``
(prior residual = v_model - R^T (v - c)). It is observable while fixes arrive; when they stop its
estimate is held (no mean reversion toward zero current) and its variance grows with the configured
walk density, so the position uncertainty grows like t^1.5 instead of sqrt(t).

Adaptive terms (the estimator is never told a noise level; it only sees its own data):

* IMU white-noise estimate from first differences of consecutive fresh IMU samples (EWMA); its variance is
  added to the configured accel/gyro process noise.
* AHRS orientation noise by covariance matching of the orientation innovations.
* Normalized-innovation-squared (NIS) monitoring of position fixes and depth: chi-square gating plus an
  EWMA of NIS/dof. When that ratio exceeds the configured limit the estimator reports DEGRADED with
  ``ESTIMATOR_INCONSISTENT`` and scales its model-mismatch process noise by the observed ratio
  (bounded), i.e. covariance matching of the process noise.

Every numeric default below is a SYNTHETIC_ONLY configured value (simulation reference), not a
measurement of a physical vehicle or sea state.
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
FIX_REJECTED = "POSITION_FIX_REJECTED"
DEPTH_REJECTED = "DEPTH_REJECTED"
ESTIMATOR_INCONSISTENT = "ESTIMATOR_INCONSISTENT"
POSE_SIGMA_ELEVATED = "POSE_SIGMA_ELEVATED"

N_STATE = 18
_SYN = "SYNTHETIC_ONLY configured default"


class EkfConfig(ConradModel):
    gravity_mps2: float = Field(default=9.80665, gt=0)
    # -- process noise ------------------------------------------------------------------------------
    accel_noise: float = Field(
        default=0.03, gt=0, description=f"m/s^2/sqrt(Hz) incl. unmodelled effects; {_SYN}"
    )
    gyro_noise: float = Field(default=0.005, gt=0, description=f"rad/s/sqrt(Hz); {_SYN}")
    accel_bias_walk: float = Field(default=1e-3, ge=0, description=f"m/s^3/sqrt(Hz); {_SYN}")
    gyro_bias_walk: float = Field(default=1e-4, ge=0, description=f"rad/s^2/sqrt(Hz); {_SYN}")
    orientation_sigma_rad: float = Field(default=0.02, gt=0, description=f"AHRS prior 1-sigma; {_SYN}")
    # -- thrust/drag velocity prior and its mismatch state -----------------------------------------------
    velocity_prior_sigma_mps: tuple[float, float, float] = Field(
        default=(0.25, 0.25, 0.4),
        description=f"white part of the prior error (unmodelled turning/coupling; correlated part is c); {_SYN}",
    )
    use_velocity_prior: bool = True
    velocity_prior_correlation_s: float = Field(default=2.0, gt=0, description=_SYN)
    model_mismatch_state: bool = Field(
        default=True, description="estimate the time-correlated prior offset c (False = legacy EST-B0)"
    )
    mismatch_initial_sigma_mps: tuple[float, float, float] = Field(
        default=(0.25, 0.25, 0.05),
        description=f"initial 1-sigma of c (water current + drag-model error), WORLD; {_SYN}",
    )
    mismatch_walk_mps_per_sqrt_s: tuple[float, float, float] = Field(
        default=(0.02, 0.02, 0.005),
        description=f"random-walk density of c (how fast the offset may change unobserved); {_SYN}",
    )
    # -- adaptive noise --------------------------------------------------------------------------------------
    imu_noise_adaptation: bool = True
    imu_noise_window_s: float = Field(default=5.0, gt=0, description=f"EWMA time constant; {_SYN}")
    orientation_noise_adaptation: bool = True
    # -- consistency monitoring ---------------------------------------------------------------------------------
    fix_gate_nis: float = Field(default=16.27, gt=0, description="chi-square 3 dof, p=0.001")
    depth_gate_nis: float | None = Field(default=10.83, gt=0, description="chi-square 1 dof, p=0.001")
    consistency_monitoring: bool = True
    nis_window_fixes: float = Field(default=10.0, ge=1, description=f"EWMA length in fixes; {_SYN}")
    nis_ratio_limit: float = Field(
        default=2.0,
        gt=1,
        description="EWMA(NIS)/dof above which the filter is inconsistent (~chi2(30,0.999)/30); configured",
    )
    max_process_noise_scale: float = Field(default=25.0, ge=1, description=f"cap on adaptation; {_SYN}")
    fix_rejections_for_lost: int = Field(default=5, ge=1)
    degraded_sigma_fraction: float = Field(default=0.5, gt=0, le=1)
    # -- initial covariance ----------------------------------------------------------------------------------------
    initial_position_sigma_m: float = Field(default=0.1, gt=0)
    initial_velocity_sigma_mps: float = Field(default=0.05, gt=0)
    initial_attitude_sigma_rad: float = Field(default=0.05, gt=0)
    stale_imu_noise_scale: float = Field(default=25.0, ge=1)

    @classmethod
    def legacy_baseline(cls) -> EkfConfig:
        """The pre-fix EST-B0 behaviour (white-noise prior, no adaptation, no consistency monitor)."""
        return cls(
            model_mismatch_state=False,
            imu_noise_adaptation=False,
            orientation_noise_adaptation=False,
            consistency_monitoring=False,
            depth_gate_nis=None,
        )


class EkfStateEstimator(StateEstimator):
    name = "EST-B1-ekf18"

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
        if not c.model_mismatch_state:
            self.name = "EST-B0-ekf15"
        self._max_sigma = float(robot_config.safety.max_pose_sigma_m.require("safety.max_pose_sigma_m"))
        self._p = np.asarray(initial_pose.position_m, dtype=np.float64)
        self._v = np.zeros(3)
        self._q = np.asarray(initial_pose.orientation_wxyz, dtype=np.float64)
        self._ba = np.zeros(3)
        self._bg = np.zeros(3)
        self._c = np.zeros(3)
        self._w = np.zeros(3)
        diag = [c.initial_position_sigma_m] * 3 + [c.initial_velocity_sigma_mps] * 3
        diag += [c.initial_attitude_sigma_rad] * 3 + [0.05] * 3 + [0.005] * 3
        diag += list(c.mismatch_initial_sigma_mps) if c.model_mismatch_state else [0.0] * 3
        self._P = np.diag(np.square(diag))
        self._stamp = initial_stamp
        self._last_imu_ns = -1
        self._fix_rejections = 0
        self._depth_rejections = 0
        self.last_nis: float | None = None
        # consistency / adaptation state (all inferred from received data)
        self.nis_ratio_ewma: float | None = None
        self.process_noise_scale = 1.0
        self._prev_imu: np.ndarray | None = None
        self.imu_accel_var_est = 0.0
        self.imu_gyro_var_est = 0.0
        self.orientation_var_est: float | None = None
        # static thrust/drag model (velocity prior)
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
    def _observe_imu_noise(self, imu: ImuSample, dt: float) -> None:
        """EWMA of squared first differences of fresh IMU samples: var(x_k - x_{k-1}) = 2 sigma^2 + dynamics.

        True dynamics inflate the estimate, which only makes the added process noise conservative.
        """
        x = np.concatenate([imu.linear_acceleration_mps2, imu.angular_velocity_rps])
        if self._prev_imu is not None:
            d = x - self._prev_imu
            beta = min(1.0, dt / self.config.imu_noise_window_s)
            self.imu_accel_var_est += beta * (float(d[:3] @ d[:3]) / 6.0 - self.imu_accel_var_est)
            self.imu_gyro_var_est += beta * (float(d[3:] @ d[3:]) / 6.0 - self.imu_gyro_var_est)
        self._prev_imu = x

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
            if c.imu_noise_adaptation:
                self._observe_imu_noise(imu, dt)
        else:  # no new IMU sample: hold velocity, inflate process noise
            a_body = rot.T @ (-g)
            a_world = np.zeros(3)
        self._p = self._p + self._v * dt + 0.5 * a_world * dt * dt
        self._v = self._v + a_world * dt
        self._q = quat_normalize(quat_mul(self._q, quat_exp(self._w * dt)))

        f = np.zeros((N_STATE, N_STATE))
        f[0:3, 3:6] = np.eye(3)
        f[3:6, 6:9] = -rot @ skew(a_body)
        f[3:6, 9:12] = -rot
        f[6:9, 6:9] = -skew(self._w)
        f[6:9, 12:15] = -np.eye(3)
        phi = np.eye(N_STATE) + f * dt
        scale = 1.0 if fresh else c.stale_imu_noise_scale
        qd = np.zeros(N_STATE)
        qd[3:6] = (c.accel_noise * scale) ** 2 * dt + self.imu_accel_var_est * dt * dt
        qd[6:9] = (c.gyro_noise * scale) ** 2 * dt + self.imu_gyro_var_est * dt * dt
        qd[9:12] = c.accel_bias_walk**2 * dt
        qd[12:15] = c.gyro_bias_walk**2 * dt
        if c.model_mismatch_state:
            walk = np.square(np.asarray(c.mismatch_walk_mps_per_sqrt_s))
            qd[15:18] = walk * self.process_noise_scale * dt
        self._P = phi @ self._P @ phi.T + np.diag(qd)
        if fresh:  # never advance the stamp without a measurement: staleness must stay visible
            self._stamp = imu.timestamp
        if fresh and imu.orientation_wxyz is not None:
            self._update_orientation(np.asarray(imu.orientation_wxyz, dtype=np.float64), dt)
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
        v_rel_body = rot.T @ (self._v - self._c)
        h = np.zeros((3, N_STATE))
        h[:, 3:6] = rot.T
        h[:, 6:9] = skew(v_rel_body)
        if self.config.model_mismatch_state:
            h[:, 15:18] = -rot.T
        # the white part of the model error is still band-limited: de-weight per-tick updates
        inflate = max(1.0, self.config.velocity_prior_correlation_s / max(dt, 1e-6))
        r = np.diag(np.square(self.config.velocity_prior_sigma_mps)) * inflate
        self._correct(self._v_model - v_rel_body, h, r)

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
        self._c = self._c + dx[15:18]
        ikh = np.eye(N_STATE) - k @ h
        self._P = ikh @ self._P @ ikh.T + k @ r @ k.T
        return float(residual @ np.linalg.solve(s, residual))

    def _update_orientation(self, q_meas: np.ndarray, dt: float) -> None:
        c = self.config
        residual = quat_log(quat_mul(quat_conj(self._q), quat_normalize(q_meas)))
        r_var = c.orientation_sigma_rad**2
        if c.orientation_noise_adaptation:
            # covariance matching: E[nu nu^T] = H P H^T + R  ->  R_hat = mean(nu^2) - mean(diag(P_att))
            sample = float(residual @ residual) / 3.0 - float(np.trace(self._P[6:9, 6:9])) / 3.0
            beta = min(1.0, dt / c.imu_noise_window_s)
            prev = self.orientation_var_est if self.orientation_var_est is not None else r_var
            self.orientation_var_est = prev + beta * (sample - prev)
            r_var = max(r_var, self.orientation_var_est)
        h = np.zeros((3, N_STATE))
        h[:, 6:9] = np.eye(3)
        self._correct(residual, h, np.eye(3) * r_var)

    def _record_nis(self, nis: float, dof: int) -> None:
        c = self.config
        if not c.consistency_monitoring:
            return
        ratio = min(nis / dof, c.fix_gate_nis)  # bounded so one outlier cannot dominate the window
        beta = 1.0 / c.nis_window_fixes
        self.nis_ratio_ewma = (
            ratio
            if self.nis_ratio_ewma is None
            else self.nis_ratio_ewma + beta * (ratio - self.nis_ratio_ewma)
        )
        # covariance matching of the (unobservable-while-blind) mismatch process noise
        self.process_noise_scale = float(np.clip(self.nis_ratio_ewma, 1.0, c.max_process_noise_scale))

    def update_depth(self, depth: DepthSample) -> None:
        if depth.health is HealthLevel.FAULT:
            return
        sigma = 0.05 if depth.health is HealthLevel.OK else 0.2
        h = np.zeros((1, N_STATE))
        h[0, 2] = 1.0
        residual = np.array([-depth.depth_m - self._p[2]])
        r = np.array([[sigma**2]])
        gate = self.config.depth_gate_nis
        if gate is not None:
            nis = float(residual[0] ** 2 / (self._P[2, 2] + r[0, 0]))
            if nis > gate:
                self._depth_rejections += 1
                return
        self._depth_rejections = 0
        self._correct(residual, h, r)

    def update_position_fix(self, position_world: np.ndarray, sigma_m: float) -> bool:
        """Gated WORLD position fix. Returns False (and counts an inconsistency) when rejected."""
        h = np.zeros((3, N_STATE))
        h[:, 0:3] = np.eye(3)
        r = np.eye(3) * max(sigma_m, 1e-3) ** 2
        residual = np.asarray(position_world, dtype=np.float64) - self._p
        s = h @ self._P @ h.T + r
        nis = float(residual @ np.linalg.solve(s, residual))
        self.last_nis = nis
        self._record_nis(nis, 3)
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
        """RMS per-axis position 1-sigma (same convention as ``Pose.position_sigma_m``)."""
        return float(np.sqrt(max(np.trace(self._P[0:3, 0:3]), 0.0) / 3.0))

    @property
    def position_covariance(self) -> np.ndarray:
        return np.array(self._P[0:3, 0:3], copy=True)

    @property
    def inconsistent(self) -> bool:
        return self.nis_ratio_ewma is not None and self.nis_ratio_ewma > self.config.nis_ratio_limit

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
        return HealthLevel.DEGRADED if self.health_reasons() else HealthLevel.OK

    def health_reasons(self) -> tuple[str, ...]:
        if self.localization_lost:
            return (LOCALIZATION_LOST,)
        reasons: list[str] = []
        if self.position_sigma_m > self.config.degraded_sigma_fraction * self._max_sigma:
            reasons.append(POSE_SIGMA_ELEVATED)
        if self._fix_rejections > 0:
            reasons.append(FIX_REJECTED)
        if self._depth_rejections > 0:
            reasons.append(DEPTH_REJECTED)
        if self.inconsistent:
            reasons.append(ESTIMATOR_INCONSISTENT)
        return tuple(reasons)

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
