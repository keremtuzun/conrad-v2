"""Targeted identification experiments A..F (ch22 "Identification experiment sequence").

Each function fits ONLY the identification split it is handed and returns a :class:`FitResult` with
Jacobian-based uncertainty. Required physical inputs that are not fitted (mass, water density,
buoyancy) are explicit arguments or log metadata; there are no silent defaults.

Signal names (SI):
    A static:  net_weight_n [N] (submerged scale, +down)  |  applied_pitch_moment_nm [N*m], pitch_rad [rad]
    B thruster: command [unitless], thrust_n [N]
    C/D/E axis: force [N | N*m], velocity [m/s | rad/s]
    F station keeping: hold_force_n [N] along one axis

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from conrad.robotics.hardware.identification.dataset import ExperimentKind, LogSegment
from conrad.robotics.hardware.identification.fitting import FitResult, IdentificationError, fit
from conrad.robotics.hardware.identification.models import G0, simulate_axis, thruster_response, trim_moment

_AXIS_UNITS = {
    ExperimentKind.SURGE_ACCELERATION: ("N", "m/s", "kg", "N*s/m", "N*s^2/m^2"),
    ExperimentKind.SWAY: ("N", "m/s", "kg", "N*s/m", "N*s^2/m^2"),
    ExperimentKind.HEAVE: ("N", "m/s", "kg", "N*s/m", "N*s^2/m^2"),
    ExperimentKind.YAW_ROTATION: ("N*m", "rad/s", "kg*m^2", "N*m*s/rad", "N*m*s^2/rad^2"),
}


def _require(segments: Sequence[LogSegment], kind: ExperimentKind) -> None:
    if not segments:
        raise IdentificationError(f"no {kind.value} segments supplied")
    wrong = [s.trajectory_id for s in segments if s.kind is not kind]
    if wrong:
        raise IdentificationError(f"segments {wrong} are not {kind.value}")


def _meta(seg: LogSegment, key: str) -> float:
    if key not in seg.metadata:
        raise IdentificationError(
            f"{seg.trajectory_id}: metadata {key!r} is required (no default is assumed)"
        )
    return float(seg.metadata[key])


def identify_displaced_volume(segments: Sequence[LogSegment], mass_kg: float) -> FitResult:
    """A: V = (m g - W_net) / (rho g) from submerged scale readings."""
    _require(segments, ExperimentKind.STATIC_TRIM)
    rho = {_meta(s, "water_density_kgm3") for s in segments}
    if len(rho) != 1:
        raise IdentificationError("segments were recorded at different water densities; fit them separately")
    rho_w = rho.pop()
    w = np.concatenate([s.signal("net_weight_n", "N") for s in segments])
    return fit(
        lambda x: (mass_kg * G0 - rho_w * G0 * x[0]) - w,
        [mass_kg / rho_w],
        ["displaced_volume_m3"],
        ["m^3"],
        lower=[0.0],
    )


def identify_trim(segments: Sequence[LogSegment], buoyancy_n: float) -> FitResult:
    """A: CoB-CoG offset (z_BG, x_BG) from tilt tests with known applied pitch moments."""
    _require(segments, ExperimentKind.STATIC_TRIM)
    th = np.concatenate([s.signal("pitch_rad", "rad") for s in segments])
    m = np.concatenate([s.signal("applied_pitch_moment_nm", "N*m") for s in segments])
    return fit(
        lambda x: trim_moment(th, buoyancy_n, x[0], x[1]) - m, [0.01, 0.0], ["z_bg_m", "x_bg_m"], ["m", "m"]
    )


class DeadzoneBracket:
    def __init__(self, lower: float, upper: float) -> None:
        self.lower, self.upper = lower, upper

    @property
    def value(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def sigma(self) -> float:
        return (self.upper - self.lower) / (2.0 * np.sqrt(3.0))  # uniform over the bracket


def _runs(u: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges of constant command."""
    edges = np.flatnonzero(np.diff(u)) + 1
    bounds = [0, *edges.tolist(), u.size]
    return list(pairwise(bounds))


def bracket_deadzone(segments: Sequence[LogSegment], noise_sigmas: float = 5.0) -> DeadzoneBracket:
    """Largest |u| with no steady response and smallest |u| with one (a hard deadzone is not LS-smooth).

    Per constant-command run only the last quarter is used (the lag has settled). The noise floor comes
    from differenced samples of zero-command tails, so a slow decay does not inflate it; a level
    counts as active when its tail mean exceeds ``noise_sigmas`` standard errors.
    """
    silent: list[float] = []
    active: list[float] = []
    for s in segments:
        u, thrust = s.signal("command", "unitless"), s.signal("thrust_n", "N")
        runs = [(a, b, float(u[a])) for a, b in _runs(u)]
        diffs = [np.diff(thrust[b - (b - a) // 4 : b]) for a, b, lv in runs if lv == 0.0 and b - a >= 8]
        if not diffs:
            raise IdentificationError(
                f"{s.trajectory_id}: needs zero-command plateaus to measure the noise floor"
            )
        noise = float(np.std(np.concatenate(diffs))) / np.sqrt(2.0)
        for a, b, lv in runs:
            if lv == 0.0 or b - a < 8:
                continue
            tail = thrust[b - (b - a) // 4 : b]
            floor = noise_sigmas * max(noise, 1e-9) / np.sqrt(tail.size)  # standard error of the tail mean
            (active if abs(float(np.mean(tail))) > floor else silent).append(abs(lv))
    if not active:
        raise IdentificationError("no step produced thrust; deadzone cannot be bracketed")
    lo = max([x for x in silent if x < min(active)], default=0.0)
    return DeadzoneBracket(lo, min(active))


def identify_thruster(segments: Sequence[LogSegment]) -> tuple[FitResult, DeadzoneBracket]:
    """B: k_fwd, k_rev, tau, latency by least squares; deadzone by bracketing."""
    _require(segments, ExperimentKind.THRUSTER_STEP)
    dz = bracket_deadzone(segments)
    has_rev = any(np.any(s.signal("command", "unitless") < -dz.upper) for s in segments)
    has_fwd = any(np.any(s.signal("command", "unitless") > dz.upper) for s in segments)
    if not (has_fwd and has_rev):
        raise IdentificationError(
            "forward AND reverse steps are required (asymmetry is measured, not assumed)"
        )

    def residual(x: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                thruster_response(s.time_s, s.signals["command"], x[0], x[1], dz.value, x[2], x[3])
                - s.signals["thrust_n"]
                for s in segments
            ]
        )

    peak = max(float(np.max(np.abs(s.signals["thrust_n"]))) for s in segments)
    result = fit(
        residual,
        [peak, peak, 0.1, 0.01],
        ["k_fwd", "k_rev", "time_constant_s", "latency_s"],
        ["N", "N", "s", "s"],
        lower=[0.0, 0.0, 1e-4, 0.0],
        upper=[np.inf, np.inf, 5.0, 1.0],
    )
    return result, dz


def identify_axis(
    segments: Sequence[LogSegment], kind: ExperimentKind, rigid_inertia: float, *, fit_bias: bool = False
) -> FitResult:
    """C/D/E: added inertia, linear and quadratic damping (+ constant bias, e.g. net buoyancy in heave)."""
    _require(segments, kind)
    if kind not in _AXIS_UNITS:
        raise IdentificationError(f"{kind.value} is not a single-axis experiment")
    fu, vu, iu, d1u, d2u = _AXIS_UNITS[kind]
    data = [(s.time_s, s.signal("force", fu), s.signal("velocity", vu)) for s in segments]

    def residual(x: np.ndarray) -> np.ndarray:
        bias = x[3] if fit_bias else 0.0
        return np.concatenate(
            [
                simulate_axis(t, f, rigid_inertia + x[0], x[1], x[2], bias, v0=float(v[0])) - v
                for t, f, v in data
            ]
        )

    names = ["added_inertia", "linear_damping", "quadratic_damping"] + (["bias"] if fit_bias else [])
    units = [iu, d1u, d2u] + ([fu] if fit_bias else [])
    x0 = [0.3 * rigid_inertia, 1.0, 1.0] + ([0.0] if fit_bias else [])
    lower = [0.0, 0.0, 0.0] + ([-np.inf] if fit_bias else [])
    return fit(residual, x0, names, units, lower=lower)


def identify_station_keeping_current(segments: Sequence[LogSegment], d1: float, d2: float) -> FitResult:
    """F: steady hold force balances drag at the water velocity: F = D1 v_c + D2 |v_c| v_c."""
    _require(segments, ExperimentKind.STATION_KEEPING)
    f = np.concatenate([s.signal("hold_force_n", "N") for s in segments])
    return fit(lambda x: d1 * x[0] + d2 * abs(x[0]) * x[0] - f, [0.1], ["current_mps"], ["m/s"])
