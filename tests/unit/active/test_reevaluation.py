"""MCBR re-evaluation: immutable partitions, purpose guard, predictive rankers, frozen production planner."""

import math
from collections.abc import Sequence

import numpy as np
import pytest
import yaml

from conrad.active import MCBRConfig, PriorView
from conrad.active.candidates import SensorOption
from conrad.active.predictive import (
    HypothesisBelief,
    PredictedOutcome,
    ScalarBelief,
    expected_values,
    hypothesis_misclassification_reduction,
    scalar_abs_error_reduction,
    scalar_misclassification_reduction,
)
from conrad.active.production import FROZEN_PATH, PRODUCTION, load_frozen, planner_digest, production_planner
from conrad.active.rankers import RankerConfig, ranker_planner
from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.active_mcbr_reeval import Budget, check_seeds
from conrad.evaluation.decision_experiments.fixtures import unc
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.schemas.decision import PlanStatus, QuestionType
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from tests.unit.active.test_mcbr import _request


# ---------------------------------------------------------------------------------------------- partitions
def test_partition_file_is_pinned_and_disjoint():
    loaded = P.load()
    assert loaded["digest"] == P.PARTITIONS_SHA256
    for domain in ("abstract", "mission"):  # disjoint within a domain; a dev seed may be dev in both
        seen: set[int] = set()
        for part in P.Partition:
            purpose = P.Purpose.DESIGN if part not in P.HELD_OUT else P.Purpose.FINAL_EVALUATION
            if part is P.Partition.VALIDATION:
                purpose = P.Purpose.SELECTION
            seeds = set(P.split(domain, part, purpose).world_seeds)
            assert seeds and not (seeds & seen)
            seen |= seeds


def test_contaminated_seed_is_development_only():
    assert P.partition_of("abstract", 2026201) is P.Partition.DEVELOPMENT
    assert P.partition_of("mission", 2026201) is P.Partition.DEVELOPMENT
    assert P.contamination_label(2026201) == P.CONTAMINATED_LABEL
    ood = P.split("abstract", "ood_test", "final_evaluation").families
    dev = P.split("abstract", "development", "design").families
    assert not set(ood) & set(dev)


def test_changed_partition_file_is_refused(tmp_path):
    raw = yaml.safe_load(P.PARTITIONS_PATH.read_text(encoding="utf-8"))
    raw["abstract"]["world_seeds"]["final_test"]["range"] = [3300000, 3300001]
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(P.PartitionIntegrityError):
        P.load(f)


def test_overlapping_partitions_are_refused():
    raw = yaml.safe_load(P.PARTITIONS_PATH.read_text(encoding="utf-8"))
    raw["abstract"]["world_seeds"]["validation"]["explicit"] = [3300005]
    with pytest.raises(P.PartitionIntegrityError):
        P.validate(raw)


@pytest.mark.parametrize("purpose", ["design", "tuning", "selection"])
@pytest.mark.parametrize("part", ["final_test", "ood_test"])
def test_design_and_tuning_cannot_read_held_out(purpose, part):
    with pytest.raises(P.PartitionAccessError):
        P.split("abstract", part, purpose)


def test_design_scope_blocks_held_out_even_under_final_purpose():
    with P.purpose_scope("design"), pytest.raises(P.PartitionAccessError):
        P.split("mission", "final_test", "final_evaluation")
    with P.purpose_scope("tuning"):
        assert P.split("abstract", "validation", "tuning").world_seeds
    with pytest.raises(P.PartitionAccessError):
        P.split("abstract", "development", "final_evaluation")


def test_stage_seed_check_refuses_contaminated_seed():
    with pytest.raises(P.PartitionAccessError):
        check_seeds("e002", [2026201])
    check_seeds("selection", [3200000])


# ---------------------------------------------------------------------------------------------- predictive
def test_preposterior_quantities():
    assert scalar_abs_error_reduction(1.0, 1e6) == pytest.approx(0.0, abs=1e-6)
    assert scalar_abs_error_reduction(1.0, 1e-6) == pytest.approx(math.sqrt(2 / math.pi), rel=1e-3)
    # a sector far from the threshold has nothing to discriminate; one at the threshold has the most
    far = scalar_misclassification_reduction(0.95, 0.001, 0.5, 0.1)
    near = scalar_misclassification_reduction(0.5, 0.05, 0.5, 0.1)
    assert near > 0.2 > far >= 0.0
    assert hypothesis_misclassification_reduction(0.5, 0.5) == pytest.approx(0.0)
    assert hypothesis_misclassification_reduction(0.5, 0.85) == pytest.approx(0.35)


class _Model:
    """A minimal but complete implementation of the ``PredictiveBelief`` protocol."""

    def __init__(self, sees: Sequence[str]) -> None:
        self.scalars: tuple[ScalarBelief, ...] = (
            ScalarBelief("a", 0.5, 0.3, 0.5, weight=1.0),
            ScalarBelief("b", 0.5, 0.3, 0.5, weight=0.0),
        )
        self.hypotheses: tuple[HypothesisBelief, ...] = (HypothesisBelief("H", 0.5),)
        self.epistemic = 0.1
        self.used_modalities: frozenset[str] = frozenset()
        self._sees = sees

    def predict(self, pose: Pose, sensor: SensorOption) -> PredictedOutcome:
        return PredictedOutcome(noise_std=dict.fromkeys(self._sees, 0.1))


def test_mission_conditioning_ignores_irrelevant_but_eig_does_not():
    rel = expected_values(_Model(["a"]), PredictedOutcome(noise_std={"a": 0.1}))
    irr = expected_values(_Model(["b"]), PredictedOutcome(noise_std={"b": 0.1}))
    assert irr["abs_error"] == 0.0 and irr["misclassification"] == 0.0
    assert irr["entropy"] == pytest.approx(rel["entropy"])
    assert irr["entropy"] > 0
    assert rel["abs_error"] > 0 and rel["misclassification"] > 0


def test_ranker_prefers_the_view_that_measures_the_relevant_element():
    ids = IdFactory(11)

    class Side(_Model):
        def predict(self, pose, sensor):  # only candidates with y > 0 measure the relevant element
            return PredictedOutcome(noise_std={"a": 0.1} if pose.position_m[1] > 0 else {"b": 0.1})

    for kind in ("mission_conditioned", "hypothesis_discrimination"):
        req = _request(ids, unc(uo=0.9), QuestionType.EXTEND_COVERAGE)
        req.predictive = Side([])
        r = ranker_planner(ids, MCBRConfig(n_azimuth=8), RankerConfig(kind=kind), kind).plan(req)
        assert r.plan.status is PlanStatus.PLAN and r.plan.primary_action is not None
        assert r.plan.primary_action.pose.position_m[1] > 0


def test_ranker_stop_rule_and_fallback_without_predictive():
    ids = IdFactory(12)
    req = _request(ids, unc(uo=0.9), QuestionType.EXTEND_COVERAGE)
    cfg = RankerConfig(kind="mission_conditioned", stop_min_value=1e-3)
    req.predictive = _Model([])
    req.predictive.predict = lambda pose, sensor: PredictedOutcome(noise_std={"a": 1e6})  # seen, useless
    stopped = ranker_planner(ids, MCBRConfig(n_azimuth=6), cfg, "x", with_stop=True).plan(req)
    assert stopped.plan.status is PlanStatus.NOT_WORTH_COST
    fallback = ranker_planner(ids, MCBRConfig(n_azimuth=6), cfg, "x", with_stop=True).plan(
        _request(
            ids,
            unc(uo=0.9),
            QuestionType.EXTEND_COVERAGE,
            views=[PriorView(position_m=(9, 9, 0), modality="RGB")],
        )
    )
    assert fallback.plan.status is PlanStatus.PLAN  # analytic mission value when no predictive belief


def test_budget_defaults_are_matched_caps():
    b = Budget.of({"max_observations": 3})
    assert b.max_observations == 3 and b.energy_j > 0 and b.time_s > 0


# ---------------------------------------------------------------------------------------------- production
def test_frozen_production_planner_loads_and_is_the_runtime_default():
    frozen = load_frozen()
    assert planner_digest(frozen["planner"]) == frozen["config_digest"]
    assert frozen["partitions_digest"] == P.PARTITIONS_SHA256
    assert frozen["selection"]["partition"] == "validation"
    p = production_planner(IdFactory(13), MCBRConfig(n_azimuth=6))
    assert p.name.startswith(PRODUCTION)
    assert MissionRuntimeConfig().planner == PRODUCTION


def test_tampered_frozen_config_is_refused(tmp_path):
    raw = yaml.safe_load(FROZEN_PATH.read_text(encoding="utf-8"))
    raw["planner"]["selected"] = "something-else"
    f = tmp_path / "f.yaml"
    f.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(Exception, match="digest"):
        load_frozen(f)


def test_rng_is_not_needed_by_predictive_rankers():
    ids = IdFactory(14)
    req = _request(ids, unc(uo=0.9), QuestionType.EXTEND_COVERAGE)
    req.rng = None
    req.predictive = _Model(["a"])
    r = ranker_planner(ids, MCBRConfig(n_azimuth=6), RankerConfig(kind="bayes_eig"), "e").plan(req)
    assert r.plan.status is PlanStatus.PLAN
    assert np.isfinite(r.plan.expected_information_gain)
