"""TRUTH-side 6-DOF underwater rigid-body kernel (L1 approximate physics).

M nu_dot + C(nu) nu + D(nu_r) nu_r + g(eta) = tau + tau_env, diagonal added mass, fixed-step RK4.
The true state is reachable only through :class:`conrad.sim.kernel.truth.TruthAccess`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from conrad.robotics.estimation.rotations import quat_mul, quat_normalize, quat_to_rot
from conrad.sim.kernel.params import SimKernelConfig, SimValidityLevel, VehicleParams
from conrad.sim.kernel.thrusters import ThrusterBank

CurrentField = Callable[[np.ndarray, float], np.ndarray]
"""``v_c(position_world[3], t_s) -> world velocity[3]`` (supplied by the environment twin)."""
SignedDistance = Callable[[np.ndarray], np.ndarray]
"""``sdf(points[N,3]) -> [N]`` signed distance to the nearest obstacle surface, metres."""


@dataclass
class _State:
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body linear velocity
    w: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body angular velocity


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _no_current(_p: np.ndarray, _t: float) -> np.ndarray:
    return np.zeros(3)


class SimKernel:
    validity_level = SimValidityLevel.L1_APPROXIMATE_PHYSICS

    def __init__(
        self,
        params: VehicleParams,
        config: SimKernelConfig,
        rng: np.random.Generator,
        current_field: CurrentField | None = None,
        sdf: SignedDistance | None = None,
    ) -> None:
        self.params = params
        self.config = config
        self._rng = rng
        self._current = current_field or _no_current
        self._sdf = sdf
        self._s = _State()
        self.t_s = 0.0
        self.step_count = 0
        self.thrusters = ThrusterBank(params.thrusters, config.thrust_noise_fraction)
        self._B = params.allocation_matrix
        self._m_lin = params.mass + params.added_mass[:3]
        self._i_tot = params.inertia + params.added_mass[3:]
        self._gust = np.zeros(3)
        self._gust_until_s = -1.0
        self._accel_body = np.zeros(3)
        self.energy_used_j = 0.0
        self.power_w = 0.0
        self.battery_capacity_scale = 1.0
        self.collision_count = 0
        self._in_collision = False
        self.min_clearance_m = float("inf")

    # -- setup -----------------------------------------------------------------------------
    def reset(self, position: np.ndarray, orientation_wxyz: np.ndarray | None = None) -> None:
        self._s = _State(p=np.asarray(position, dtype=np.float64).copy())
        if orientation_wxyz is not None:
            self._s.q = quat_normalize(np.asarray(orientation_wxyz, dtype=np.float64))

    def set_gust(self, velocity_world: np.ndarray, until_s: float) -> None:
        self._gust = np.asarray(velocity_world, dtype=np.float64)
        self._gust_until_s = until_s

    def current_at(self, p: np.ndarray, t_s: float) -> np.ndarray:
        v = np.asarray(self._current(p, t_s), dtype=np.float64)
        return v + self._gust if t_s <= self._gust_until_s else v

    @property
    def vehicle_radius_m(self) -> float:
        return 0.5 * float(np.max(self.params.dims))

    # -- physics ---------------------------------------------------------------------------
    def _restoring(self, rot: np.ndarray) -> np.ndarray:
        g, pr = self.config.gravity_mps2, self.params
        weight = rot.T @ np.array([0.0, 0.0, -pr.mass * g])
        buoy = rot.T @ np.array([0.0, 0.0, self.config.water_density_kgm3 * g * pr.volume])
        return np.concatenate([weight + buoy, _cross(pr.r_g, weight) + _cross(pr.r_b, buoy)])

    def _derivative(self, s: _State, tau: np.ndarray, t_s: float) -> tuple[np.ndarray, ...]:
        pr = self.params
        rot = quat_to_rot(s.q)
        v_r = s.v - rot.T @ self.current_at(s.p, t_s)
        nu_r = np.concatenate([v_r, s.w])
        drag = -(pr.d_lin + pr.d_quad * np.abs(nu_r)) * nu_r
        wrench = tau + drag + self._restoring(rot)
        a_lin = pr.added_mass[:3] * v_r
        force = wrench[:3] - pr.mass * _cross(s.w, s.v) - _cross(s.w, a_lin)
        torque = wrench[3:] - _cross(s.w, self._i_tot * s.w) - _cross(v_r, a_lin)
        v_dot = force / self._m_lin
        w_dot = torque / self._i_tot
        q_dot = 0.5 * quat_mul(s.q, np.array([0.0, *s.w]))
        return rot @ s.v, q_dot, v_dot, w_dot

    @staticmethod
    def _advance(s: _State, d: tuple[np.ndarray, ...], h: float) -> _State:
        return _State(p=s.p + h * d[0], q=s.q + h * d[1], v=s.v + h * d[2], w=s.w + h * d[3])

    def step(self) -> None:
        """One fixed physics step (``config.physics_dt_s``)."""
        dt = self.config.physics_dt_s
        thrust = self.thrusters.step(self.t_s, dt, self._rng)
        tau = self._B @ thrust
        s = self._s
        k1 = self._derivative(s, tau, self.t_s)
        k2 = self._derivative(self._advance(s, k1, dt / 2), tau, self.t_s + dt / 2)
        k3 = self._derivative(self._advance(s, k2, dt / 2), tau, self.t_s + dt / 2)
        k4 = self._derivative(self._advance(s, k3, dt), tau, self.t_s + dt)
        mean = tuple((a + 2 * b + 2 * c + d) / 6 for a, b, c, d in zip(k1, k2, k3, k4, strict=True))
        new = self._advance(s, mean, dt)
        new.q = quat_normalize(new.q)
        self._accel_body = mean[2] + _cross(s.w, s.v)
        self._s = new
        self.step_count += 1
        self.t_s = self.step_count * dt
        self._power(thrust, dt)
        self._collide()

    def _power(self, thrust: np.ndarray, dt: float) -> None:
        cfg = self.config
        p_thr = cfg.thruster_power_w_per_n15 * float(np.sum(np.abs(thrust) ** 1.5))
        self.power_w = p_thr + cfg.hotel_power_w
        self.energy_used_j += self.power_w * dt

    def _collide(self) -> None:
        if self._sdf is None:
            return
        s = self._s
        clearance = float(self._sdf(s.p[None, :])[0]) - self.vehicle_radius_m
        self.min_clearance_m = min(self.min_clearance_m, clearance)
        if clearance >= 0:
            if clearance > 0.02:  # hysteresis: a sustained contact is one collision event
                self._in_collision = False
            return
        if not self._in_collision:
            self.collision_count += 1
            self._in_collision = True
        eps = 1e-3
        grad = np.array(
            [
                float(self._sdf((s.p + eps * e)[None, :])[0] - self._sdf((s.p - eps * e)[None, :])[0])
                for e in np.eye(3)
            ]
        )
        n = float(np.linalg.norm(grad))
        if n < 1e-9:
            return
        normal = grad / n
        rot = quat_to_rot(s.q)
        v_world = rot @ s.v
        vn = float(v_world @ normal)
        if vn < 0:
            v_world = v_world - (1.0 + self.config.collision_restitution) * vn * normal
            s.v = rot.T @ v_world
        s.p = s.p - clearance * normal

    # -- read-only views used by SimRobotHardware / TruthAccess -----------------------------
    def _snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = self._s
        return s.p.copy(), s.q.copy(), s.v.copy(), s.w.copy(), self._accel_body.copy()

    @property
    def battery_remaining_fraction(self) -> float:
        cap = self.params.battery_capacity_j * self.battery_capacity_scale
        return float(min(1.0, max(0.0, 1.0 - self.energy_used_j / cap))) if cap > 0 else 0.0
