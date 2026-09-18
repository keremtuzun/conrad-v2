"""Forward models shared by identification, validation and the synthetic log generator.

Thruster (same form as conrad.sim.kernel and the Unity Propulsion module):
    T_c(u) = 0 for |u| < d;  k_f u^2 for u > 0;  -k_r u^2 for u < 0     (T = k u|u| with deadzone)
    tau dT/dt + T = T_c(u(t - L))                                        (first-order lag + latency)
Single axis (surge / sway / heave / yaw), still water unless a current is supplied:
    (M_rb + M_a) dv/dt = F - D1 v_r - D2 |v_r| v_r + b,   v_r = v - v_c
Static trim (Z-up body frame, pitch about +Y), applied moment M holding pitch theta:
    M = B (z_BG sin(theta) + x_BG cos(theta)),   r_BG = r_B - r_G

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import numpy as np

G0 = 9.80665  # standard gravity (exact by definition)


def thrust_command(u: np.ndarray, k_fwd: float, k_rev: float, deadzone: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    out = np.where(u >= 0.0, k_fwd * u * u, -k_rev * u * u)
    return np.where(np.abs(u) < deadzone, 0.0, out)


def thruster_response(
    t: np.ndarray, u: np.ndarray, k_fwd: float, k_rev: float, deadzone: float, tau: float, latency: float
) -> np.ndarray:
    """Exact response to a piecewise-constant (zero-order-hold) command via step superposition.

    Continuous in ``tau`` and ``latency`` so least squares can move them smoothly.
    """
    t = np.asarray(t, dtype=np.float64)
    tc = thrust_command(u, k_fwd, k_rev, deadzone)
    change = np.flatnonzero(np.diff(tc)) + 1
    steps = [(t[0], tc[0])] + [(t[i], tc[i] - tc[i - 1]) for i in change]
    out = np.zeros_like(t)
    tau = max(tau, 1e-6)
    for s, delta in steps:
        dt = t - s - latency
        out += np.where(dt > 0.0, delta * (1.0 - np.exp(-np.clip(dt, 0.0, None) / tau)), 0.0)
        if s == t[0]:
            # initial level is assumed to be at steady state before the log started
            out += np.where(dt <= 0.0, delta, 0.0)
    return out


def simulate_axis(
    t: np.ndarray,
    force: np.ndarray,
    inertia_total: float,
    d1: float,
    d2: float,
    bias: float = 0.0,
    v0: float = 0.0,
    current: float = 0.0,
    substeps: int = 4,
) -> np.ndarray:
    """RK4 with zero-order-hold force between samples; returns velocity at every sample time."""
    t = np.asarray(t, dtype=np.float64)
    f = np.asarray(force, dtype=np.float64)
    v = np.empty_like(t)
    v[0] = v0

    def acc(vel: float, fk: float) -> float:
        vr = vel - current
        return (fk - d1 * vr - d2 * abs(vr) * vr + bias) / inertia_total

    for k in range(t.size - 1):
        h = (t[k + 1] - t[k]) / substeps
        x = v[k]
        for _ in range(substeps):
            a1 = acc(x, f[k])
            a2 = acc(x + 0.5 * h * a1, f[k])
            a3 = acc(x + 0.5 * h * a2, f[k])
            a4 = acc(x + h * a3, f[k])
            x += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0
        v[k + 1] = x
    return v


def trim_moment(theta: np.ndarray, buoyancy_n: float, z_bg: float, x_bg: float) -> np.ndarray:
    th = np.asarray(theta, dtype=np.float64)
    return buoyancy_n * (z_bg * np.sin(th) + x_bg * np.cos(th))
