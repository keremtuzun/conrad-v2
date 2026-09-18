"""Logged scenario prior for hidden Twin2T parameters (ch33: "sampled from a logged scenario prior").

Every number here is a PRIOR, never a measurement. ``source`` is a ``SourceKind``-style label restricted
to LITERATURE_PRIOR / ENGINEERING_ESTIMATE. Ranges are deliberately wide; the exact calibrated ranges
remain OPEN (ch11 "Still unresolved: exact parameter ranges").

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

LITERATURE_PRIOR = "LITERATURE_PRIOR"
ENGINEERING_ESTIMATE = "ENGINEERING_ESTIMATE"
SCENARIO_SUPPLIED = "SCENARIO_SUPPLIED"
ALLOWED_PRIOR_SOURCES = frozenset({LITERATURE_PRIOR, ENGINEERING_ESTIMATE})


@dataclass(frozen=True)
class PriorSpec:
    name: str
    units: str
    low: float
    high: float
    dist: str  # "uniform" | "loguniform"
    source: str
    citation: str

    def __post_init__(self) -> None:
        if self.source not in ALLOWED_PRIOR_SOURCES:
            raise ValueError(f"prior {self.name}: source must be one of {sorted(ALLOWED_PRIOR_SOURCES)}")
        if not self.citation:
            raise ValueError(f"prior {self.name}: citation string required")
        if self.dist not in ("uniform", "loguniform"):
            raise ValueError(f"prior {self.name}: unknown dist {self.dist!r}")
        if not self.low <= self.high or (self.dist == "loguniform" and self.low <= 0):
            raise ValueError(f"prior {self.name}: invalid range")

    def sample(self, rng: np.random.Generator) -> float:
        if self.dist == "loguniform":
            return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class SampledParameter:
    """One logged hidden parameter. ``source`` is never 'MEASURED'."""

    name: str
    value: float
    units: str
    source: str
    citation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "units": self.units,
            "source": self.source,
            "citation": self.citation,
        }


_MELCHERS = (
    "model form: Melchers (2003) 'Modeling of marine immersion corrosion for mild and low-alloy steels', "
    "Corrosion 59(4); power-law form d=A*t^n as in ISO 9224. Range is an order-of-magnitude estimate."
)
_DNV = (
    "model form: DNV-RP-B401 linear coating breakdown factor f_c = a + b*t; range brackets the "
    "tabulated coating-category constants."
)
_PARIS = (
    "model form: Paris & Erdogan (1963) da/dN = C*(dK)^m; steel constants of the order given in BS 7910 "
    "(units m/cycle, MPa*sqrt(m)). Range is an order-of-magnitude estimate."
)
_EST = "engineering estimate for synthetic scenario generation; not calibrated against data"


def _p(name: str, units: str, low: float, high: float, dist: str, source: str, cit: str) -> PriorSpec:
    return PriorSpec(name, units, low, high, dist, source, cit)


DEFAULT_PRIORS: dict[str, PriorSpec] = {
    p.name: p
    for p in (
        _p("wall_thickness_m", "m", 0.008, 0.030, "uniform", ENGINEERING_ESTIMATE, _EST),
        _p("initial_corrosion_depth_m", "m", 0.0, 1.0e-3, "uniform", ENGINEERING_ESTIMATE, _EST),
        _p("initial_crack_length_m", "m", 0.0, 3.0e-3, "uniform", ENGINEERING_ESTIMATE, _EST),
        _p("load_cycles_per_s", "1/s", 0.01, 0.2, "loguniform", ENGINEERING_ESTIMATE, _EST),
        _p("stress_range_pa", "Pa", 20.0e6, 80.0e6, "uniform", ENGINEERING_ESTIMATE, _EST),
        _p("corrosion_A", "m/yr^n", 5.0e-5, 2.0e-4, "loguniform", ENGINEERING_ESTIMATE, _MELCHERS),
        _p("corrosion_n", "1", 0.4, 0.8, "uniform", ENGINEERING_ESTIMATE, _MELCHERS),
        _p("corrosion_r0", "m/yr", 5.0e-5, 2.0e-4, "loguniform", ENGINEERING_ESTIMATE, _MELCHERS),
        _p("corrosion_rs", "m/yr", 2.0e-5, 1.0e-4, "loguniform", ENGINEERING_ESTIMATE, _MELCHERS),
        _p("corrosion_t_transition_yr", "yr", 1.0, 4.0, "uniform", ENGINEERING_ESTIMATE, _MELCHERS),
        _p("temperature_q10", "1", 1.3, 2.0, "uniform", ENGINEERING_ESTIMATE, _MELCHERS),
        _p("coating_a", "1", 0.02, 0.10, "uniform", LITERATURE_PRIOR, _DNV),
        _p("coating_b_per_yr", "1/yr", 0.008, 0.05, "uniform", LITERATURE_PRIOR, _DNV),
        _p("cp_residual_factor", "1", 0.02, 0.10, "uniform", ENGINEERING_ESTIMATE, _EST),
        _p("burial_factor", "1", 0.3, 0.7, "uniform", ENGINEERING_ESTIMATE, _EST),
        _p(
            "paris_C", "m/cycle/(MPa*sqrt(m))^m", 1.0e-12, 3.0e-11, "loguniform", ENGINEERING_ESTIMATE, _PARIS
        ),
        _p("paris_m", "1", 2.8, 3.2, "uniform", LITERATURE_PRIOR, _PARIS),
        _p("delta_k_threshold", "MPa*sqrt(m)", 2.0, 5.0, "uniform", ENGINEERING_ESTIMATE, _PARIS),
        _p(
            "geometry_factor_Y",
            "1",
            1.0,
            1.2,
            "uniform",
            LITERATURE_PRIOR,
            "edge/surface crack geometry factor ~1.12, Tada, Paris & Irwin, 'The Stress Analysis of "
            "Cracks Handbook'",
        ),
        _p("crack_aspect_ratio", "1", 2.0, 6.0, "uniform", ENGINEERING_ESTIMATE, _EST),
        _p("sigma_corrosion", "m/sqrt(yr)", 1.0e-5, 5.0e-5, "loguniform", ENGINEERING_ESTIMATE, _EST),
        _p("sigma_fatigue", "m/sqrt(yr)", 1.0e-5, 1.0e-4, "loguniform", ENGINEERING_ESTIMATE, _EST),
    )
}

HIDDEN_PARAMETER_NAMES: tuple[str, ...] = tuple(
    n
    for n in DEFAULT_PRIORS
    if n
    not in (
        "wall_thickness_m",
        "initial_corrosion_depth_m",
        "initial_crack_length_m",
        "load_cycles_per_s",
        "stress_range_pa",
    )
)


def merged_priors(overrides: dict[str, dict[str, Any]] | None) -> dict[str, PriorSpec]:
    """Apply config overrides ({name: {low, high, ...}}) on top of the defaults."""
    out = dict(DEFAULT_PRIORS)
    for name, fields in (overrides or {}).items():
        if name not in out:
            raise KeyError(f"unknown prior {name!r}")
        base = out[name]
        out[name] = PriorSpec(
            name=name,
            units=str(fields.get("units", base.units)),
            low=float(fields.get("low", base.low)),
            high=float(fields.get("high", base.high)),
            dist=str(fields.get("dist", base.dist)),
            source=str(fields.get("source", base.source)),
            citation=str(fields.get("citation", base.citation)),
        )
    return out


def sample_parameter(spec: PriorSpec, rng: np.random.Generator) -> SampledParameter:
    return SampledParameter(spec.name, spec.sample(rng), spec.units, spec.source, spec.citation)


def supplied_parameter(name: str, value: float, units: str) -> SampledParameter:
    return SampledParameter(
        name, float(value), units, SCENARIO_SUPPLIED, "value given by the shared scenario"
    )
