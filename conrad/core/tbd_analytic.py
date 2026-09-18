"""Analytic temporal belief dynamics over PHYSICAL elapsed seconds (ch7).

Uncertainty growth is driven by the declared dynamics of each property (process noise), not by a
universal ``u0 + k*dt`` decay: a static property (zero process noise) keeps its certainty.

implementation_status: EXPERIMENTAL_CANDIDATE (baseline family: classical state-space, H-CORE-04)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

from conrad.core.config import TbdConfig
from conrad.core.state import AnalyticBeliefState, PropertyEstimate
from conrad.schemas.belief import KnowledgeStatus


@dataclass(frozen=True)
class PropertyDynamics:
    """Declared by the domain child. ``process_noise_per_s`` is a variance rate in units^2 / s."""

    rate_per_s: float = 0.0
    process_noise_per_s: float = 0.0


@dataclass(frozen=True)
class AnalyticPrediction:
    state: AnalyticBeliefState
    delta_t_s: float
    information_retained: float


class AnalyticTBD:
    def __init__(self, config: TbdConfig | None = None) -> None:
        self.cfg = config or TbdConfig()

    def predict(
        self,
        state: AnalyticBeliefState,
        delta_t_s: float,
        dynamics: Mapping[str, PropertyDynamics] | None = None,
    ) -> AnalyticPrediction:
        """Returns a PREDICTED state; the corrected (input) state is never mutated."""
        if not math.isfinite(delta_t_s) or delta_t_s < 0:
            raise ValueError(f"delta_t_s must be a finite non-negative number of seconds, got {delta_t_s}")
        default = PropertyDynamics(0.0, self.cfg.analytic_process_noise_per_s)
        estimates: dict[str, PropertyEstimate] = {}
        retained: list[float] = []
        for name, est in state.estimates.items():
            dyn = (dynamics or {}).get(name, default)
            grown = est.variance + dyn.process_noise_per_s * delta_t_s
            estimates[name] = PropertyEstimate(
                est.mean + dyn.rate_per_s * delta_t_s,
                grown,
                est.units,
                KnowledgeStatus.PREDICTED if delta_t_s > 0 else est.status,
            )
            retained.append(math.log(est.variance / grown) if grown > 0 and est.variance > 0 else 0.0)
        # geometric mean of variance ratios composes exactly: predict(a) then predict(b) == predict(a+b)
        info = math.exp(sum(retained) / len(retained)) if retained else 1.0
        coverage = state.coverage * info
        predicted = replace(state, estimates=estimates, coverage=coverage, uo=max(0.0, 1.0 - coverage))
        return AnalyticPrediction(predicted, delta_t_s, info)


def normalised_surprise(predicted: PropertyEstimate, measured: float, measurement_variance: float) -> float:
    """Temporal surprise d(E_t, B_hat_t): |innovation| in predicted standard deviations (ch7)."""
    return abs(measured - predicted.mean) / math.sqrt(max(predicted.variance + measurement_variance, 1e-18))
