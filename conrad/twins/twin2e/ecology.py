"""Z: ecological entity state and rate-limited stochastic state rules (ch12-13, ch33). TRUTH PLANE.

Sessile entities (biofouling on structures, benthic patches) carry cover and condition in [0, 1];
mobile groups carry a presence flag; habitat regions aggregate their members UPWARD only (a
habitat value is never copied down onto an individual). Biofouling here is a surface-cover
process only: it has no corrosion term and never writes technical (Twin2T) state.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np

from conrad.twins.twin2e.config import EcologyConfig, MeifeSwitches
from conrad.twins.twin2e.priors import KIND_BENTHIC, KIND_BIOFOULING, KIND_HABITAT, KIND_MOBILE

SESSILE = (KIND_BIOFOULING, KIND_BENTHIC)


@dataclass
class EntityState:
    entity_id: UUID
    kind: str
    parent_id: UUID | None
    position_m: np.ndarray
    radius_m: float
    params: dict[str, Any]
    cover: float = 0.0
    condition: float = 1.0
    presence: bool = False
    thermal_stress: float = 0.0
    frozen_env: dict[str, float] = field(default_factory=dict)
    history: dict[str, float] = field(default_factory=dict)

    @property
    def covered_area_m2(self) -> float:
        return self.cover * math.pi * self.radius_m**2

    def as_truth(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "entity_kind": self.kind,
            "species_group": self.params.get("species_group"),
            "position_m": [float(v) for v in self.position_m],
            "radius_m": self.radius_m,
        }
        if self.kind == KIND_MOBILE:
            out["presence"] = self.presence
        else:
            out.update(cover=self.cover, condition=self.condition)
        if self.kind in SESSILE:
            out["thermal_stress_c_days"] = self.thermal_stress
        return out


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def env_factors(p: dict[str, Any], env: dict[str, float]) -> tuple[float, float, float]:
    ft = math.exp(-0.5 * ((env["temperature"] - p["thermal_optimum_c"]) / p["thermal_tolerance_c"]) ** 2)
    fl = env["light"] / (env["light"] + p["light_half_saturation_w_m2"])
    u = env["current_speed"]
    fu = u / (u + p["current_half_saturation_m_s"]) * math.exp(-((u / p["scour_speed_m_s"]) ** 2))
    return ft, fl, fu


class EcologyEngine:
    def __init__(self, cfg: EcologyConfig, switches: MeifeSwitches) -> None:
        self.cfg = cfg
        self.sw = switches

    def step(
        self,
        entities: dict[UUID, EntityState],
        env: dict[UUID, dict[str, float]],
        dt_s: float,
        rngs: dict[UUID, np.random.Generator],
    ) -> None:
        days = dt_s * self.cfg.time_scale / 86400.0
        for eid in sorted(entities, key=str):
            e = entities[eid]
            rng = rngs[eid]
            if e.kind == KIND_HABITAT or not self.sw.entity_dynamics:
                continue
            if self.sw.random_ecology:
                self._random(e, days, rng)
            elif e.kind in SESSILE:
                self._sessile(e, env[eid] if self.sw.field_to_entity else e.frozen_env, days, rng)
            elif e.kind == KIND_MOBILE:
                self._mobile(e, env[eid] if self.sw.field_to_entity else e.frozen_env, days, rng)
        self.aggregate(entities)

    def _limit(self, delta: float, rate_per_day: float, days: float) -> float:
        cap = rate_per_day * days
        return max(-cap, min(cap, delta))

    def _sessile(self, e: EntityState, env: dict[str, float], days: float, rng: np.random.Generator) -> None:
        p = e.params
        ft, fl, fu = env_factors(p, env)
        excess = env["temperature"] - (
            p["thermal_optimum_c"] + p["thermal_tolerance_c"] + p["stress_threshold_c"]
        )
        if excess > 0:
            e.thermal_stress += excess * days
            d_cond = -0.1 * excess * e.condition * days
        elif self.sw.recovery:
            e.thermal_stress *= math.exp(-p["recovery_rate_per_day"] * days)
            d_cond = p["recovery_rate_per_day"] * (1.0 - e.condition) * days
        else:
            d_cond = 0.0
        e.condition = _clip01(e.condition + self._limit(d_cond, self.cfg.max_condition_rate_per_day, days))
        c = e.cover
        growth = p["growth_rate_per_day"] * ft * fl * fu * c * (1.0 - c) * e.condition
        settle = p["settlement_rate_per_day"] * fu * (1.0 - c)
        mort = (
            p["base_mortality_per_day"] + p["independent_mortality_per_day"] + 0.05 * (1.0 - e.condition)
        ) * c
        delta = (growth + settle - mort) * days
        if self.sw.stochasticity:
            sigma = (
                self.cfg.growth_noise_sigma_per_sqrt_day * math.sqrt(days) * math.sqrt(c * (1.0 - c) + 1e-4)
            )
            delta += sigma * float(rng.standard_normal())
        e.cover = _clip01(c + self._limit(delta, self.cfg.max_cover_rate_per_day, days))

    def _mobile(self, e: EntityState, env: dict[str, float], days: float, rng: np.random.Generator) -> None:
        p = e.params
        cap = self.cfg.max_transition_rate_per_day
        if e.presence:
            boost = 1.0 + 2.0 * max(0.0, env["turbidity"] / p["turbidity_avoidance_ntu"] - 1.0)
            rate = min(cap, p["departure_rate_per_day"] * boost)
        else:
            rate = min(cap, p["arrival_rate_per_day"])
        prob = 1.0 - math.exp(-rate * days)
        u = float(rng.uniform()) if self.sw.stochasticity else 1.0
        if u < prob:
            e.presence = not e.presence

    def _random(self, e: EntityState, days: float, rng: np.random.Generator) -> None:
        """Independent-noise ecology baseline: no environment, no mechanism."""
        s = self.cfg.random_ecology_sigma_per_sqrt_day * math.sqrt(days)
        e.cover = _clip01(e.cover + s * float(rng.standard_normal()))
        e.condition = _clip01(e.condition + s * float(rng.standard_normal()))
        if e.kind == KIND_MOBILE and rng.uniform() < 1.0 - math.exp(-days):
            e.presence = not e.presence

    @staticmethod
    def aggregate(entities: dict[UUID, EntityState]) -> None:
        """Habitat = area-weighted mean of its sessile members (upward only)."""
        for h in entities.values():
            if h.kind != KIND_HABITAT:
                continue
            kids = [k for k in entities.values() if k.parent_id == h.entity_id and k.kind in SESSILE]
            area = sum(math.pi * k.radius_m**2 for k in kids)
            if area <= 0:
                h.cover, h.condition = 0.0, 1.0
                continue
            h.cover = _clip01(sum(k.cover * math.pi * k.radius_m**2 for k in kids) / area)
            h.condition = _clip01(sum(k.condition * math.pi * k.radius_m**2 for k in kids) / area)
