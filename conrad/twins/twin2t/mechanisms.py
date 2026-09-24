"""Individual degradation mechanism dynamics (ch10 Corrosion engine, Fatigue/crack engine).

These are published engineering model FORMS with scenario-sampled parameters; they are baseline
engineering models, not inventions and not calibrated physics (ch10: "V1 should not pretend to contain
universal corrosion physics").

Corrosion families (d = wall loss, tau = effective exposure time in years):
  * POWER_LAW:  d(tau) = A * tau**n            (ISO 9224 / Melchers 2003 long-term form)
  * BILINEAR:   d(tau) = r0*tau                 for tau <= t_s
                d(tau) = r0*t_s + rs*(tau-t_s)  for tau >  t_s   (Melchers 2003 bilinear approximation)
  Environmental modifier: q10**((T-10)/10) * (DO/7 mg/L) for oxygen-diffusion control, burial factor,
  cathodic protection residual factor; exposed fraction = coating breakdown (DNV-RP-B401 f_c = a + b*t).

Fatigue: Paris & Erdogan (1963) da/dN = C*(dK)**m with dK = Y*dsigma*sqrt(pi*a) [MPa*sqrt(m)], zero growth
below the threshold dK_th, integrated in closed form over N = f*dt cycles.

Stochastic residual epsilon ~ N(0, sigma_mechanism*sqrt(dt_yr)); the increment is truncated at zero so
material loss never reverses without an intervention (monotonicity invariant).

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from conrad.twins.twin2t.state import ComponentParameters, DegradationState, Environment

SECONDS_PER_YEAR = 365.25 * 86400.0
AREA_GROWTH_TIMESCALE_YR = 1.0
"""ENGINEERING_ESTIMATE: e-folding time for corroded area to reach the exposed fraction."""


class CorrosionModel(str, Enum):
    POWER_LAW = "POWER_LAW"
    BILINEAR = "BILINEAR"


def corrosion_curve(model: CorrosionModel, tau_yr: float, p: ComponentParameters) -> float:
    tau = max(tau_yr, 0.0)
    if model is CorrosionModel.POWER_LAW:
        return p.value("corrosion_A") * tau ** p.value("corrosion_n")
    t_s = p.value("corrosion_t_transition_yr")
    r0, rs = p.value("corrosion_r0"), p.value("corrosion_rs")
    return r0 * tau if tau <= t_s else r0 * t_s + rs * (tau - t_s)


def environment_modifier(env: Environment, p: ComponentParameters, conditioned: bool = True) -> float:
    """Multiplicative rate modifier. With ``conditioned=False`` (ablation) it is 1."""
    if not conditioned:
        return 1.0
    temp = p.value("temperature_q10") ** ((env.temperature_c - 10.0) / 10.0)
    oxygen = max(env.dissolved_oxygen_mg_l, 0.0) / 7.0
    mod = temp * oxygen
    if env.buried:
        mod *= p.value("burial_factor")
    if env.cp_active:
        mod *= p.value("cp_residual_factor")
    return float(mod)


def coating_breakdown(age_yr: float, p: ComponentParameters) -> float:
    """DNV-RP-B401 style mean coating breakdown factor, clipped to [0, 1]."""
    return float(min(1.0, max(0.0, p.value("coating_a") + p.value("coating_b_per_yr") * max(age_yr, 0.0))))


@dataclass(frozen=True)
class CorrosionResult:
    state: DegradationState
    rate_m_per_yr: float


def corrosion_step(
    state: DegradationState,
    p: ComponentParameters,
    env: Environment,
    exposed_fraction: float,
    dt_s: float,
    rng: np.random.Generator,
    model: CorrosionModel,
    *,
    material_factor: float,
    stochastic: bool,
    environment_conditioned: bool,
) -> CorrosionResult:
    """d_{t+dt} = d_t + [F(tau+dt) - F(tau)] * modifier * material + eps (truncated at 0).

    ``exposed_fraction`` is the fraction of the surface without effective coating; depth is the local
    (maximum) wall loss on exposed metal, area fraction relaxes monotonically toward the exposed fraction.
    """
    if not state.validity.get("corrosion_depth_m", False) or dt_s <= 0:
        return CorrosionResult(state, 0.0)
    dt_yr = dt_s / SECONDS_PER_YEAR
    exposed = min(1.0, max(0.0, exposed_fraction))
    mod = environment_modifier(env, p, environment_conditioned) * material_factor
    tau0 = state.corrosion_exposure_yr
    tau1 = tau0 + dt_yr * (1.0 if exposed > 0 else 0.0)
    delta = (corrosion_curve(model, tau1, p) - corrosion_curve(model, tau0, p)) * mod
    if exposed <= 0:
        delta = 0.0
    if stochastic and exposed > 0:
        delta += float(rng.normal(0.0, p.value("sigma_corrosion") * math.sqrt(dt_yr))) * mod
    delta = max(0.0, delta)
    depth = min(p.wall_thickness_m, state.corrosion_depth_m + delta)
    area = state.corrosion_area_fraction
    if exposed > area:
        area = area + (exposed - area) * (1.0 - math.exp(-dt_yr / AREA_GROWTH_TIMESCALE_YR))
    new = state.with_(
        corrosion_depth_m=depth, corrosion_area_fraction=min(1.0, area), corrosion_exposure_yr=tau1
    )
    return CorrosionResult(new, delta / dt_yr if dt_yr > 0 else 0.0)


def coating_step(state: DegradationState, p: ComponentParameters, dt_s: float) -> DegradationState:
    """Coating ages; breakdown follows the DNV linear form but never decreases without renewal."""
    if not state.validity.get("coating_breakdown_fraction", False) or dt_s <= 0:
        return state
    age = state.coating_age_yr + dt_s / SECONDS_PER_YEAR
    frac = max(state.coating_breakdown_fraction, coating_breakdown(age, p))
    return state.with_(coating_age_yr=age, coating_breakdown_fraction=frac)


def stress_intensity_range(a_m: float, stress_range_pa: float, y: float) -> float:
    """dK = Y * dsigma * sqrt(pi * a) in MPa*sqrt(m)."""
    return y * (stress_range_pa / 1.0e6) * math.sqrt(math.pi * max(a_m, 0.0))


@dataclass(frozen=True)
class FatigueResult:
    state: DegradationState
    delta_k: float
    threshold: float
    grew: bool


def paris_integrate(a0: float, cycles: float, c: float, m: float, y: float, dsigma_mpa: float) -> float:
    """Closed-form integral of da/dN = C (Y dsigma sqrt(pi a))**m over ``cycles`` (m != 2)."""
    if a0 <= 0 or cycles <= 0:
        return a0
    k = c * (y * dsigma_mpa * math.sqrt(math.pi)) ** m
    e = 1.0 - m / 2.0
    if abs(e) < 1e-9:
        return a0 * math.exp(k * cycles)
    rhs = a0**e + e * k * cycles
    if rhs <= 0:  # unstable growth within this tick: grows without bound
        return math.inf
    return float(rhs ** (1.0 / e))


def fatigue_step(
    state: DegradationState,
    p: ComponentParameters,
    stress_range_pa: float,
    cycles_per_s: float,
    dt_s: float,
    rng: np.random.Generator,
    *,
    stochastic: bool,
    max_crack_length_m: float,
    effective_wall_m: float | None = None,
) -> FatigueResult:
    """Paris-law growth with threshold. No growth when dK < dK_th or when there is no crack."""
    thr = p.value("delta_k_threshold")
    if not state.validity.get("crack_length_m", False):
        return FatigueResult(state, 0.0, thr, False)
    a0 = state.crack_length_m
    y = p.value("geometry_factor_Y")
    dk = stress_intensity_range(a0, stress_range_pa, y)
    if a0 <= 0 or dk < thr or dt_s <= 0:
        return FatigueResult(state, dk, thr, False)
    cycles = cycles_per_s * dt_s
    a1 = paris_integrate(a0, cycles, p.value("paris_C"), p.value("paris_m"), y, stress_range_pa / 1e6)
    if stochastic:
        a1 += float(rng.normal(0.0, p.value("sigma_fatigue") * math.sqrt(dt_s / SECONDS_PER_YEAR)))
    a1 = max(a0, min(a1, max_crack_length_m))
    wall = p.wall_thickness_m if effective_wall_m is None else effective_wall_m
    if wall <= 0:
        raise ValueError("effective wall thickness must be positive")
    depth = min(wall, max(state.crack_depth_m, a1 / p.value("crack_aspect_ratio")))
    return FatigueResult(state.with_(crack_length_m=a1, crack_depth_m=depth), dk, thr, a1 > a0)
