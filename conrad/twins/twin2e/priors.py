"""Logged priors for every Twin2E quantity a scenario leaves unspecified. TRUTH PLANE.

No value here is a measurement. Each sampled value is written to ``ecological_state["prior_log"]``
with its label (ENGINEERING_ESTIMATE / SYNTHETIC_ONLY / DERIVED) and the basis for the range.
Ranges are broad engineering ranges, NOT calibrated against any dataset (calibration is OPEN).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.world import Scenario, WorldEntity

ENGINEERING_ESTIMATE = "ENGINEERING_ESTIMATE"
LITERATURE_PRIOR = "LITERATURE_PRIOR"
SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
DERIVED = "DERIVED"

UNIT_ALIASES: dict[str, set[str]] = {
    "degC": {"degC", "C", "celsius"},
    "NTU": {"NTU"},
    "m s-1": {"m s-1", "m/s"},
    "W m-2": {"W m-2", "W/m^2", "W/m2"},
}

KIND_BIOFOULING = "BIOFOULING_ON_STRUCTURE"
KIND_BENTHIC = "BENTHIC_PATCH"
KIND_MOBILE = "MOBILE_GROUP"
KIND_HABITAT = "HABITAT_REGION"
ENTITY_KINDS = (KIND_BIOFOULING, KIND_BENTHIC, KIND_MOBILE, KIND_HABITAT)

_KIND_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (KIND_HABITAT, ("habitat", "region")),
    (KIND_MOBILE, ("fish", "school", "mobile")),
    (KIND_BENTHIC, ("coral", "benthic", "seagrass", "kelp", "reef", "sponge")),
    (KIND_BIOFOULING, ("pipeline", "pipe", "riser", "jacket", "structure", "pile", "cable", "anode")),
)


class Twin2EScenarioError(ValueError):
    """The scenario asks Twin2E to attach state it cannot justify."""


@dataclass(frozen=True)
class Prior:
    sampler: Callable[[np.random.Generator], Any]
    units: str
    label: str
    basis: str


def _u(lo: float, hi: float) -> Callable[[np.random.Generator], float]:
    return lambda r: float(r.uniform(lo, hi))


def _logu(lo: float, hi: float) -> Callable[[np.random.Generator], float]:
    return lambda r: float(math.exp(r.uniform(math.log(lo), math.log(hi))))


def _beta(a: float, b: float) -> Callable[[np.random.Generator], float]:
    return lambda r: float(r.beta(a, b))


def _current(r: np.random.Generator) -> list[float]:
    speed = float(r.uniform(0.02, 0.4))
    theta = float(r.uniform(0.0, 2.0 * math.pi))
    return [speed * math.cos(theta), speed * math.sin(theta), 0.0]


E = ENGINEERING_ESTIMATE
ENVIRONMENT_PRIORS: dict[str, Prior] = {
    "temperature": Prior(_u(8.0, 26.0), "degC", E, "broad shelf-sea surface range"),
    "turbidity": Prior(
        lambda r: float(2.0 * math.exp(0.6 * r.standard_normal())), "NTU", E, "clear-to-moderate coastal"
    ),
    "current": Prior(_current, "m s-1", E, "tidal/residual speed 0.02-0.4, random heading, w=0"),
    "light": Prior(_u(200.0, 900.0), "W m-2", E, "daytime broadband surface irradiance"),
}

PARAMETER_PRIORS: dict[str, Prior] = {
    "temperature_lapse_c_per_m": Prior(_u(0.0, 0.15), "degC m-1", E, "warmer surface; T=Ts+lapse*z"),
    "kd_water_per_m": Prior(_u(0.04, 0.15), "m-1", E, "clear-water diffuse attenuation"),
    "k_turbidity_per_m_per_ntu": Prior(_u(0.02, 0.08), "m-1 NTU-1", E, "particle attenuation slope"),
    "diffusivity_h_m2_s": Prior(_logu(0.1, 5.0), "m2 s-1", E, "horizontal eddy diffusivity"),
    "diffusivity_v_m2_s": Prior(_logu(1e-5, 1e-3), "m2 s-1", E, "vertical eddy diffusivity"),
    "resuspension_critical_speed_m_s": Prior(_u(0.15, 0.35), "m s-1", E, "bed erosion threshold"),
    "resuspension_rate": Prior(_logu(1e-4, 1e-3), "NTU s-1 (m s-1)-2", E, "excess-speed^2 law"),
    "settling_time_s": Prior(_logu(3600.0, 86400.0), "s", E, "relaxation to background turbidity"),
    "temperature_relaxation_time_s": Prior(_logu(86400.0, 864000.0), "s", E, "relaxation to background T"),
    "current_fluctuation_sigma_m_s": Prior(_u(0.0, 0.05), "m s-1", E, "OU fluctuation amplitude"),
    "current_fluctuation_time_s": Prior(_u(1800.0, 7200.0), "s", E, "OU correlation time"),
    "filtration_m3_s_per_m2": Prior(_logu(1e-6, 1e-5), "m3 s-1 m-2", E, "suspension-feeder clearance"),
    "start_time_of_day_h": Prior(_u(0.0, 24.0), "h", SYNTHETIC_ONLY, "diel phase at t=0"),
    "surface_z_m": Prior(lambda r: 0.0, "m", SYNTHETIC_ONLY, "flat sea surface at z=0 (not sampled)"),
}

_COMMON_ENTITY: dict[str, Prior] = {
    "initial_condition": Prior(_beta(8.0, 2.0), "1", E, "mostly healthy at t=0"),
    "radius_m": Prior(_u(0.5, 3.0), "m", SYNTHETIC_ONLY, "ecological footprint radius"),
    "independent_mortality_per_day": Prior(
        lambda r: 0.0, "d-1", E, "non-environmental cause; 0 unless confounded"
    ),
}
_SESSILE: dict[str, Prior] = {
    "growth_rate_per_day": Prior(_u(0.01, 0.08), "d-1", E, "logistic cover growth"),
    "thermal_optimum_c": Prior(_u(12.0, 24.0), "degC", E, "Gaussian thermal response centre"),
    "thermal_tolerance_c": Prior(_u(2.0, 6.0), "degC", E, "thermal response width"),
    "light_half_saturation_w_m2": Prior(_u(20.0, 150.0), "W m-2", E, "Monod light response"),
    "current_half_saturation_m_s": Prior(_u(0.05, 0.15), "m s-1", E, "larval/food supply"),
    "scour_speed_m_s": Prior(_u(0.6, 1.2), "m s-1", E, "hydrodynamic scour"),
    "settlement_rate_per_day": Prior(_logu(1e-4, 1e-2), "d-1", E, "recruitment onto free space"),
    "base_mortality_per_day": Prior(_u(0.001, 0.01), "d-1", E, "background loss"),
    "stress_threshold_c": Prior(_u(1.0, 3.0), "degC", E, "excess over optimum+tolerance -> stress"),
    "recovery_rate_per_day": Prior(_u(0.02, 0.1), "d-1", E, "condition recovery when unstressed"),
}
ENTITY_PRIORS: dict[str, dict[str, Prior]] = {
    KIND_BIOFOULING: {
        **_COMMON_ENTITY,
        **_SESSILE,
        "initial_cover": Prior(_beta(2.0, 5.0), "1", E, "partial fouling"),
        "species_group": Prior(lambda r: "mixed_fouling_community", "-", SYNTHETIC_ONLY, "label"),
    },
    KIND_BENTHIC: {
        **_COMMON_ENTITY,
        **_SESSILE,
        "initial_cover": Prior(_beta(2.0, 3.0), "1", E, "benthic cover"),
        "species_group": Prior(lambda r: "benthic_cover", "-", SYNTHETIC_ONLY, "label"),
    },
    KIND_MOBILE: {
        **_COMMON_ENTITY,
        "initial_presence_probability": Prior(_u(0.2, 0.8), "1", E, "presence at t=0"),
        "arrival_rate_per_day": Prior(_u(1.0, 4.0), "d-1", E, "Markov arrival"),
        "departure_rate_per_day": Prior(_u(1.0, 4.0), "d-1", E, "Markov departure"),
        "turbidity_avoidance_ntu": Prior(_u(5.0, 20.0), "NTU", E, "departure boost above this"),
        "species_group": Prior(lambda r: "fish_aggregation", "-", SYNTHETIC_ONLY, "label"),
    },
    KIND_HABITAT: {
        "species_group": Prior(lambda r: "habitat_aggregate", "-", SYNTHETIC_ONLY, "label"),
        "radius_m": Prior(_u(20.0, 60.0), "m", SYNTHETIC_ONLY, "habitat footprint radius"),
    },
}


def canonical_units(units: str, expected: str) -> None:
    if units not in UNIT_ALIASES.get(expected, {expected}):
        raise Twin2EScenarioError(f"units {units!r} not accepted where {expected!r} is required")


def env_setting(environment: dict[str, Any], key: str) -> Any:
    """Read ``environment[key]`` = {"value": ..., "units": ...}; bare numbers are refused."""
    entry = environment.get(key)
    if not isinstance(entry, dict) or "value" not in entry or "units" not in entry:
        raise Twin2EScenarioError(f"environment[{key!r}] must be a dict with 'value' and 'units'")
    canonical_units(str(entry["units"]), ENVIRONMENT_PRIORS[key].units)
    return entry["value"]


def infer_entity_kind(entity: WorldEntity) -> str:
    t = entity.entity_type.lower()
    for kind, words in _KIND_KEYWORDS:
        if any(w in t for w in words):
            return kind
    raise Twin2EScenarioError(
        f"cannot infer ecological entity_kind for entity_type {entity.entity_type!r}; set it explicitly"
    )


def _log(log: list[dict[str, Any]], path: str, value: Any, prior: Prior) -> None:
    log.append(
        {"path": path, "value": value, "units": prior.units, "label": prior.label, "basis": prior.basis}
    )


def populate_ecological_state(
    scenario: Scenario,
    rng: np.random.Generator,
    position_sampler: Callable[[np.random.Generator], list[float]] | None = None,
) -> Scenario:
    """Fill unspecified Twin2E keys from logged priors. Refuses state for non-ecological entities."""
    known = {e.id: e for e in scenario.world_entities}
    eco = dict(scenario.ecological_state)
    entries: dict[str, Any] = {str(k): dict(v) for k, v in eco.get("entities", {}).items()}
    for key in entries:
        ent = known.get(UUID(key))
        if ent is None:
            raise Twin2EScenarioError(f"ecological state for unknown world entity {key}")
        if not ent.domain_ownership.ecological:
            raise Twin2EScenarioError(f"entity {key} is not ecologically owned; refusing to attach state")
    log: list[dict[str, Any]] = list(eco.get("prior_log", []))
    env = {k: (dict(v) if isinstance(v, dict) else v) for k, v in scenario.environment.items()}
    for key, prior in ENVIRONMENT_PRIORS.items():
        if key in env:
            env_setting(env, key)
            continue
        value = prior.sampler(rng)
        env[key] = {"value": value, "units": prior.units, "source": prior.label}
        _log(log, f"environment.{key}", value, prior)
    params = dict(eco.get("parameters", {}))
    for key, prior in PARAMETER_PRIORS.items():
        if key not in params:
            params[key] = prior.sampler(rng)
            _log(log, f"ecological_state.parameters.{key}", params[key], prior)
    spatial = scenario.spatial_state.get("entities", {})
    for ent in sorted(
        (e for e in scenario.world_entities if e.domain_ownership.ecological), key=lambda e: str(e.id)
    ):
        sid = str(ent.id)
        entry = entries.setdefault(sid, {})
        if "entity_kind" not in entry:
            entry["entity_kind"] = infer_entity_kind(ent)
            log.append(
                {
                    "path": f"entities.{sid}.entity_kind",
                    "value": entry["entity_kind"],
                    "units": "-",
                    "label": DERIVED,
                    "basis": f"entity_type={ent.entity_type}",
                }
            )
        kind = entry["entity_kind"]
        if kind not in ENTITY_KINDS:
            raise Twin2EScenarioError(f"unknown entity_kind {kind!r} for {sid}")
        for key, prior in ENTITY_PRIORS[kind].items():
            if key not in entry:
                entry[key] = prior.sampler(rng)
                _log(log, f"entities.{sid}.{key}", entry[key], prior)
        if "position_m" not in entry:
            pos = spatial.get(sid, {}).get("position_m") or ent.metadata.get("position_m")
            if pos is not None:
                entry["position_m"] = [float(v) for v in pos]
                entry["position_source"] = "TWIN2S_SHARED_CONTEXT"
            elif position_sampler is not None and kind != KIND_HABITAT:
                entry["position_m"] = position_sampler(rng)
                entry["position_source"] = SYNTHETIC_ONLY
                log.append(
                    {
                        "path": f"entities.{sid}.position_m",
                        "value": entry["position_m"],
                        "units": "m",
                        "label": SYNTHETIC_ONLY,
                        "basis": "no Twin2S geometry supplied",
                    }
                )
    eco["entities"] = entries
    eco["parameters"] = params
    eco["prior_log"] = log
    data = scenario.model_dump(mode="python")
    data["environment"] = env
    data["ecological_state"] = eco
    return Scenario.model_validate(data)
