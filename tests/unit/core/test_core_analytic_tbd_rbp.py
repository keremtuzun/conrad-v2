import dataclasses

import pytest

from conrad.core.rbp import AnalyticRBP
from conrad.core.state import AnalyticBeliefState, PropertyEstimate
from conrad.core.tbd import AnalyticTBD, PropertyDynamics, normalised_surprise
from conrad.schemas.belief import KnowledgeStatus, Relationship
from conrad.schemas.uncertainty import unknown_uncertainty
from conrad.schemas.world import Domain

STATE = AnalyticBeliefState(estimates={"p": PropertyEstimate(0.5, 0.01)}, coverage=0.8, uo=0.2)


def test_analytic_tbd_uses_delta_t_and_marks_predicted():
    tbd = AnalyticTBD()
    dyn = {"p": PropertyDynamics(rate_per_s=0.01, process_noise_per_s=1e-3)}
    a = tbd.predict(STATE, 10.0, dyn).state
    b = tbd.predict(STATE, 20.0, dyn).state
    assert a.estimates["p"].status is KnowledgeStatus.PREDICTED
    assert b.estimates["p"].variance > a.estimates["p"].variance > STATE.estimates["p"].variance
    assert b.uo > a.uo > STATE.uo
    assert STATE.estimates["p"].status is KnowledgeStatus.OBSERVED  # corrected state untouched


def test_analytic_tbd_static_property_keeps_certainty():
    pred = AnalyticTBD().predict(STATE, 1e6, {"p": PropertyDynamics(0.0, 0.0)}).state
    assert pred.estimates["p"].variance == STATE.estimates["p"].variance
    assert pred.uo == pytest.approx(STATE.uo)


def test_analytic_tbd_composes_over_split_intervals():
    tbd, dyn = AnalyticTBD(), {"p": PropertyDynamics(0.0, 1e-3)}
    once = tbd.predict(STATE, 30.0, dyn).state
    twice = tbd.predict(tbd.predict(STATE, 10.0, dyn).state, 20.0, dyn).state
    assert once.estimates["p"].variance == pytest.approx(twice.estimates["p"].variance)
    assert once.coverage == pytest.approx(twice.coverage)


def test_analytic_tbd_rejects_negative_dt_and_surprise():
    with pytest.raises(ValueError):
        AnalyticTBD().predict(STATE, -1.0)
    assert normalised_surprise(PropertyEstimate(0.0, 0.01), 0.3, 0.0) == pytest.approx(3.0)


def _rel(ids, s, t, rtype="ATTACHED", conf=0.9):
    return Relationship(
        relationship_id=ids.new(),
        relation_type=rtype,
        source_belief_id=s,
        target_belief_id=t,
        source_domain=Domain.TECHNICAL,
        target_domain=Domain.TECHNICAL,
        confidence=conf,
        uncertainty=unknown_uncertainty(),
        provenance_id=ids.new(),
    )


def test_analytic_rbp_labels_inferred_and_keeps_observational_uncertainty(ids):
    a, b = ids.new(), ids.new()
    target = AnalyticBeliefState(uo=0.9, coverage=0.1)
    out = AnalyticRBP().propagate({a: STATE, b: target}, [_rel(ids, a, b)])
    assert len(out) == 1 and out[0].belief_id == b
    est = out[0].state.estimates["p"]
    assert est.status is KnowledgeStatus.INFERRED
    assert out[0].state.uo == target.uo and out[0].state.coverage == target.coverage
    assert est.variance > STATE.estimates["p"].variance  # relational info is weaker than the source
    assert out[0].source_belief_ids == (a,)


def test_analytic_rbp_never_overwrites_direct_observation(ids):
    a, b = ids.new(), ids.new()
    observed = AnalyticBeliefState(estimates={"p": PropertyEstimate(0.1, 0.001)})
    out = AnalyticRBP().propagate({a: STATE, b: observed}, [_rel(ids, a, b)])
    assert out == []


def test_analytic_rbp_replaces_previous_inference_instead_of_refusing(ids):
    a, b = ids.new(), ids.new()
    rbp = AnalyticRBP()
    first = rbp.propagate({a: STATE, b: AnalyticBeliefState()}, [_rel(ids, a, b)])[0].state
    second = rbp.propagate({a: STATE, b: first}, [_rel(ids, a, b)])[0].state
    assert second.estimates["p"].variance == pytest.approx(first.estimates["p"].variance)
    zero = rbp.propagate({a: STATE, b: AnalyticBeliefState()}, [_rel(ids, a, b, conf=0.0)])
    assert zero == []
    assert dataclasses.is_dataclass(first)
