"""Belief-predictive value rankers behind the MCBR interface (ch16, ch18 re-evaluation, 2026-09-19).

Three ranking rules plug into the shared ``MCBRPlanner`` pipeline (InformationNeed -> candidates ->
feasibility filter -> value ranking -> ObservationPlan). They differ only in the predicted value:

    bayes_eig                  expected entropy reduction of the target beliefs      (standard EIG, A-B6b)
    mission_conditioned        expected reduction of the mission-relevant belief error (A-B11)
    hypothesis_discrimination  expected reduction of the probability of misclassifying the mission-relevant
                               hypotheses (quantity above its action threshold; binary hypotheses)  (A-B12)

Values come from ``PlanningRequest.predictive`` (BELIEF only). Without it the rankers fall back to the
analytic mission value of ``InformationGainEstimator`` (visibility-weighted channel gains); that fallback is
stated in every plan's model_version via the planner name.

Cost handling (``cost_mode``): ``none`` ranks by value only; ``subtract`` ranks by value - w * cost;
``ratio`` ranks by value / (cost_floor + w * cost). ``independent_modality_weight`` multiplies predicted
values of modalities that were already used while the belief reports model ignorance (U_E): an OOD belief
cannot trust the sensor model it has been relying on, so it prefers an independent modality.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from conrad.active.config import MCBRConfig
from conrad.active.gap import KnowledgeGap
from conrad.active.planner import MCBRPlanner, PlanningRequest, ScoredCandidate
from conrad.active.predictive import expected_values
from conrad.schemas.base import ConradModel
from conrad.schemas.ids import IdFactory

RankerKind = Literal["bayes_eig", "mission_conditioned", "hypothesis_discrimination"]
VALUE_KEY: dict[str, str] = {
    "bayes_eig": "entropy",
    "mission_conditioned": "abs_error",
    "hypothesis_discrimination": "misclassification",
}


class RankerConfig(ConradModel):
    kind: RankerKind
    cost_mode: Literal["none", "subtract", "ratio"] = "none"
    cost_weight: float = Field(default=1.0, ge=0)
    cost_floor: float = Field(default=0.05, gt=0)
    independent_modality_weight: float = Field(
        default=1.0, ge=0, le=1, description="value multiplier for already-used modalities when U_E is high"
    )
    epistemic_threshold: float = Field(default=0.5, ge=0, le=1)
    stop_min_value: float | None = Field(
        default=None, ge=0, description="NOT_WORTH_COST when the best predicted value is below this"
    )


class PredictiveRanker:
    """Scorer + stop rule for one ``RankerConfig``; predicted outcomes are cached per request."""

    def __init__(self, cfg: RankerConfig) -> None:
        self.cfg = cfg
        self._cache_key: tuple[object, ...] | None = None
        self._cache: dict[int, float] = {}

    def value(self, c: ScoredCandidate, req: PlanningRequest) -> float:
        if req.predictive is None:
            return float(c.mission_value)
        key = (id(req), req.need.need_id, len(req.prior_views))
        if self._cache_key != key:
            self._cache_key, self._cache = key, {}
        idx = c.raw.index
        if idx not in self._cache:
            model = req.predictive
            v = expected_values(model, model.predict(c.raw.pose, c.raw.sensor))[VALUE_KEY[self.cfg.kind]]
            if (
                model.epistemic >= self.cfg.epistemic_threshold
                and c.raw.sensor.modality in model.used_modalities
            ):
                v *= self.cfg.independent_modality_weight
            self._cache[idx] = float(v)
        return self._cache[idx]

    def __call__(self, gap: KnowledgeGap, c: ScoredCandidate, req: PlanningRequest) -> float:
        v = self.value(c, req)
        w = self.cfg.cost_weight
        if self.cfg.cost_mode == "subtract":
            return v - w * c.cost
        if self.cfg.cost_mode == "ratio":
            return v / (self.cfg.cost_floor + w * c.cost)
        return v

    def stop(self, best: ScoredCandidate, req: PlanningRequest) -> bool:
        return self.cfg.stop_min_value is not None and self.value(best, req) < self.cfg.stop_min_value


def ranker_planner(
    ids: IdFactory, cfg: MCBRConfig, ranker: RankerConfig, name: str, *, with_stop: bool = False
) -> MCBRPlanner:
    r = PredictiveRanker(ranker)
    return MCBRPlanner(ids, cfg, r, name=name, value_gate=False, stop=r.stop if with_stop else None)


DEFAULT_RANKERS: dict[str, RankerConfig] = {
    "A-B6b_bayes_eig": RankerConfig(kind="bayes_eig"),
    "A-B11_mission_conditioned": RankerConfig(kind="mission_conditioned"),
    "A-B12_hypothesis_discrimination": RankerConfig(kind="hypothesis_discrimination"),
}
