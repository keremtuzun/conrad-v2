"""Analytic CEFD coupling (ch12 CEFD, EXPERIMENTAL_CANDIDATE): only justified cross-type effects.

field -> entity
  * observability: the turbidity belief sets the survey measurement variance (beam attenuation);
  * context: the temperature belief gives a thermal-stress LIKELIHOOD (INFERRED, never damage) and,
    through the coupling gate, inflates cover process noise. It never moves the cover mean:
    environmental stress is not ecological change.
entity -> field
  * suspension-feeder filtration: sessile cover beliefs act as a small first-order turbidity sink.

The gate g = gate_prior * local field coverage: an unobserved field cannot drive an entity.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from conrad.domains.ecological.config import CouplingConfig, Model2EConfig
from conrad.domains.ecological.entity_belief import EntityBelief
from conrad.domains.ecological.field_belief import FieldBeliefGrid

DAMAGE_CLAIM = "ecological_damage"
STRESS_CLAIM = "thermal_stress_likelihood"


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def visibility(turbidity_ntu: float, range_m: float, cfg: CouplingConfig) -> float:
    c = cfg.beam_attenuation_clear_per_m + cfg.beam_attenuation_per_m_per_ntu * max(0.0, turbidity_ntu)
    return max(cfg.min_visibility, math.exp(-c * max(0.0, range_m)))


@dataclass(frozen=True)
class SurveyNoise:
    variance: float
    turbidity_used: float | None
    visibility: float | None


class AnalyticCEFD:
    def __init__(self, config: Model2EConfig) -> None:
        self.cfg = config
        self.cc = config.coupling
        self.sw = config.switches

    @property
    def field_to_entity(self) -> bool:
        return self.sw.field_to_entity and self.sw.fields and self.sw.entities

    @property
    def entity_to_field(self) -> bool:
        return self.sw.entity_to_field and self.sw.fields and self.sw.entities

    def gate(self, fields: FieldBeliefGrid, name: str, p: np.ndarray) -> float:
        fb = fields.fields.get(name)
        if fb is None or fb.n_obs == 0:
            return 0.0
        _, v = fields.sample(name, p)
        local = v - float(fb.level_var[0])
        return self.cc.gate_prior * float(np.clip(1.0 - local / fb.local_var, 0.0, 1.0))

    # ------------------------------------------------------------------ field -> entity
    def survey_noise(
        self, fields: FieldBeliefGrid | None, b: EntityBelief, range_m: float | None
    ) -> SurveyNoise:
        ec = self.cfg.entity
        if not self.field_to_entity or fields is None or "turbidity" not in fields.fields:
            return SurveyNoise(ec.uncoupled_meas_sd**2, None, None)
        fb = fields.fields["turbidity"]
        r = 3.0 if range_m is None else range_m
        if fb.n_obs == 0:
            # turbidity unknown: fall back to the turbidity-blind noise, never to clear water
            return SurveyNoise(ec.uncoupled_meas_sd**2, None, None)
        m, v = fields.sample("turbidity", b.position_m)
        c1 = self.cc.beam_attenuation_per_m_per_ntu
        vis = visibility(m, r, self.cc)
        # E[1/vis^2] under N(m, v) for the turbidity term (lognormal moment), capped by min_visibility
        spread = math.exp(2.0 * (c1 * r) ** 2 * v)
        var = ec.cover_meas_sd**2 / vis**2 * min(spread, 1.0 / self.cc.min_visibility**2)
        return SurveyNoise(min(var, 0.25), m, vis)

    def stress(self, fields: FieldBeliefGrid, b: EntityBelief) -> tuple[float, float] | None:
        """(stress likelihood, gate) or None when temperature is unobserved or coupling is off."""
        if not self.field_to_entity or not b.sessile or "temperature" not in fields.fields:
            return None
        if fields.fields["temperature"].n_obs == 0:
            return None
        m, v = fields.sample("temperature", b.position_m)
        thr = self.threshold(b)
        p = normal_sf((thr - m) / math.sqrt(max(v, 1e-12)))
        return p, self.gate(fields, "temperature", b.position_m)

    def threshold(self, b: EntityBelief) -> float:
        if b.asset is not None and b.asset.stress_threshold_c is not None:
            return b.asset.stress_threshold_c
        return self.cc.stress_threshold_c

    def process_inflation(self, b: EntityBelief, gate: float) -> float:
        if b.stress_p is None:
            return 1.0
        return 1.0 + self.cc.stress_process_inflation * gate * b.stress_p

    # ------------------------------------------------------------------ entity -> field
    def filtration(
        self, entities: list[EntityBelief], fields: FieldBeliefGrid
    ) -> tuple[list[np.ndarray], list[float]]:
        if not self.entity_to_field or "turbidity" not in fields.fields:
            return [], []
        pos, rates = [], []
        for b in entities:
            if not b.sessile or b.n_hits == 0 or b.asset is None:
                continue
            area = math.pi * b.asset.radius_m**2
            rate = self.cc.filtration_m3_s_per_m2 * area * b.cover_mean / self.cc.filtration_mixing_volume_m3
            pos.append(b.position_m)
            rates.append(rate)
        return pos, rates
