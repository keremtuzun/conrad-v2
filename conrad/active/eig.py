"""Analytical expected-information-gain estimator, one term per uncertainty cause (ch16, ch18).

    U_O  new viewpoint / coverage      gain = U_O * visibility * (1 - redundancy with ANY earlier view)
    U_A  measurement noise             gain = U_A * visibility * sensor quality * range quality
                                              * (1 - redundancy with earlier views of the SAME modality)
    U_C  contradiction                 gain = U_C * visibility * D(P(O|H_i,a), P(O|H_j,a)), with a bonus
                                              for a modality that is independent of the earlier evidence
    U_E  model ignorance / OOD         gain = U_E * visibility * epistemic_resolvability, and only for a
                                              modality that has not been tried (more of the same is 0)

The estimator predicts from BELIEF only. It is an estimate; experiments must score it against actual
hidden-state error reduction, never against its own uncertainty numbers (ch18 caveat).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math

from conrad.active.candidates import RawCandidate
from conrad.active.config import MCBRConfig
from conrad.active.gap import KnowledgeGap
from conrad.schemas.base import ConradModel
from conrad.schemas.decision import ResourceCost, UncertaintyType


class GainBreakdown(ConradModel):
    aleatoric: float = 0.0
    epistemic: float = 0.0
    contradiction: float = 0.0
    observational: float = 0.0
    discrimination: float = 0.0
    redundancy: float = 0.0

    @property
    def total(self) -> float:
        return self.aleatoric + self.epistemic + self.contradiction + self.observational

    def by_channel(self, channel: UncertaintyType) -> float:
        return float(getattr(self, channel.value.lower()))


class InformationGainEstimator:
    def __init__(self, config: MCBRConfig) -> None:
        self.config = config

    def redundancy(self, gap: KnowledgeGap, candidate: RawCandidate, same_modality_only: bool) -> float:
        best = 0.0
        p = candidate.pose.position_m
        for view in gap.prior_views:
            if same_modality_only and view.modality != candidate.sensor.modality:
                continue
            d2 = sum((p[i] - view.position_m[i]) ** 2 for i in range(3))
            best = max(best, math.exp(-d2 / (2 * self.config.redundancy_sigma_m**2)))
        return best

    def estimate(self, gap: KnowledgeGap, candidate: RawCandidate, visibility: float) -> GainBreakdown:
        sensor = candidate.sensor
        u = gap.uncertainty
        span = max(sensor.max_range_m - sensor.min_range_m, 1e-9)
        range_quality = 1.0 - 0.5 * min(1.0, max(0.0, (candidate.standoff_m - sensor.min_range_m) / span))
        used = {v.modality for v in gap.prior_views}
        new_modality = sensor.modality not in used
        red_any = self.redundancy(gap, candidate, same_modality_only=False)
        red_same = self.redundancy(gap, candidate, same_modality_only=True)
        separability = sensor.separability(gap.hypotheses)
        if new_modality and used:
            separability = min(1.0, separability * (1.0 + self.config.independent_modality_bonus))
        return GainBreakdown(
            observational=u.observational * visibility * (1.0 - red_any),
            aleatoric=u.aleatoric
            * visibility
            * sensor.measurement_quality
            * range_quality
            * (1.0 - red_same),
            contradiction=u.contradiction * visibility * separability,
            epistemic=u.epistemic
            * visibility
            * self.config.epistemic_resolvability
            * (1.0 if (new_modality or not used) else 0.0),
            discrimination=visibility * separability,
            redundancy=red_same,
        )

    def mission_value(self, gap: KnowledgeGap, gain: GainBreakdown, priority: float) -> float:
        """MIG = priority x sum_c w_c gain_c, where the channels the need targets have weight 1."""
        value = 0.0
        for channel in UncertaintyType:
            weight = 1.0 if channel in gap.targeted else self.config.off_target_weight
            value += weight * gain.by_channel(channel)
        return priority * value

    def cost(self, cost: ResourceCost) -> float:
        w = self.config.cost
        return (
            w.time_per_s * cost.time_s
            + w.energy_per_j * cost.energy_j
            + w.risk * cost.risk
            + w.travel_per_m * cost.travel_m
        )
