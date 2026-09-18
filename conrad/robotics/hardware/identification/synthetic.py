"""SYNTHETIC identification logs generated from KNOWN parameters, to validate the tooling itself.

Every segment is marked ``synthetic=True`` and ``source="SYNTHETIC_ONLY:..."``; reports built from them
are SYNTHETIC_TOOLING_CHECK and can never become IDENTIFIED RobotConfig values.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import numpy as np

from conrad.robotics.hardware.identification.dataset import ExperimentKind, LogSegment
from conrad.robotics.hardware.identification.models import G0, simulate_axis, thruster_response, trim_moment

SOURCE = "SYNTHETIC_ONLY:conrad.robotics.hardware.identification.synthetic"


def static_buoyancy_log(
    tid: str, rng: np.random.Generator, mass_kg: float, volume_m3: float, rho: float, n: int, noise_n: float
) -> LogSegment:
    t = np.arange(n) * 0.1
    w = (mass_kg - rho * volume_m3) * G0 + rng.normal(0.0, noise_n, n)
    return LogSegment(
        tid,
        ExperimentKind.STATIC_TRIM,
        t,
        {"net_weight_n": w},
        {"net_weight_n": "N"},
        synthetic=True,
        source=SOURCE,
        metadata={"water_density_kgm3": rho},
    )


def tilt_log(
    tid: str,
    rng: np.random.Generator,
    buoyancy_n: float,
    z_bg: float,
    x_bg: float,
    moments_nm: np.ndarray,
    angle_noise_rad: float,
) -> LogSegment:
    th_grid = np.linspace(-0.6, 0.6, 2001)
    m_grid = trim_moment(th_grid, buoyancy_n, z_bg, x_bg)  # monotonic for z_bg > |x_bg| in this range
    theta = np.interp(moments_nm, m_grid, th_grid) + rng.normal(0.0, angle_noise_rad, moments_nm.size)
    t = np.arange(moments_nm.size) * 1.0
    return LogSegment(
        tid,
        ExperimentKind.STATIC_TRIM,
        t,
        {"applied_pitch_moment_nm": moments_nm, "pitch_rad": theta},
        {"applied_pitch_moment_nm": "N*m", "pitch_rad": "rad"},
        synthetic=True,
        source=SOURCE,
    )


def thruster_step_log(
    tid: str,
    rng: np.random.Generator,
    levels: tuple[float, ...],
    k_fwd: float,
    k_rev: float,
    deadzone: float,
    tau: float,
    latency: float,
    dwell_s: float = 1.0,
    dt: float = 0.005,
    noise_n: float = 0.05,
) -> LogSegment:
    """Alternating zero / level plateaus so every step starts from rest and has a noise floor."""
    plateau = round(dwell_s / dt)
    u = np.concatenate(
        [np.concatenate([np.zeros(plateau), np.full(plateau, lv)]) for lv in levels] + [np.zeros(plateau)]
    )
    t = np.arange(u.size) * dt
    thrust = thruster_response(t, u, k_fwd, k_rev, deadzone, tau, latency) + rng.normal(0.0, noise_n, u.size)
    return LogSegment(
        tid,
        ExperimentKind.THRUSTER_STEP,
        t,
        {"command": u, "thrust_n": thrust},
        {"command": "unitless", "thrust_n": "N"},
        synthetic=True,
        source=SOURCE,
    )


def axis_log(
    tid: str,
    rng: np.random.Generator,
    kind: ExperimentKind,
    rigid_inertia: float,
    added: float,
    d1: float,
    d2: float,
    force_profile: np.ndarray,
    dt: float,
    bias: float = 0.0,
    noise: float = 0.002,
    force_units: str = "N",
    velocity_units: str = "m/s",
) -> LogSegment:
    t = np.arange(force_profile.size) * dt
    v = simulate_axis(t, force_profile, rigid_inertia + added, d1, d2, bias) + rng.normal(0.0, noise, t.size)
    return LogSegment(
        tid,
        kind,
        t,
        {"force": force_profile, "velocity": v},
        {"force": force_units, "velocity": velocity_units},
        synthetic=True,
        source=SOURCE,
    )


def station_keeping_log(
    tid: str, rng: np.random.Generator, d1: float, d2: float, current_mps: float, n: int, noise_n: float
) -> LogSegment:
    f = d1 * current_mps + d2 * abs(current_mps) * current_mps + rng.normal(0.0, noise_n, n)
    return LogSegment(
        tid,
        ExperimentKind.STATION_KEEPING,
        np.arange(n) * 0.1,
        {"hold_force_n": f},
        {"hold_force_n": "N"},
        synthetic=True,
        source=SOURCE,
    )


def pulse_profile(amplitudes: tuple[float, ...], on_s: float, off_s: float, dt: float) -> np.ndarray:
    """Force plateaus separated by coast phases: excites both damping terms and the added inertia."""
    parts = [np.concatenate([np.full(round(on_s / dt), a), np.zeros(round(off_s / dt))]) for a in amplitudes]
    return np.concatenate(parts)
