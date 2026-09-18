from __future__ import annotations

import numpy as np
import pytest

from conrad.schemas.ids import IdFactory
from conrad.schemas.world import ScenarioEvent
from conrad.twins.twin2e import (
    BASELINE_SWITCHES,
    CONFOUNDER_KINDS,
    baseline_config,
    build_small_scenario,
    make_confounded_variant,
    make_counterfactual_pair,
    validate_twin,
)
from conrad.twins.twin2e.priors import KIND_BENTHIC, KIND_BIOFOULING

SESSILE = (KIND_BIOFOULING, KIND_BENTHIC)


def _run(make_twin, scenario, cfg, steps=6, dt=600.0):
    tw = make_twin(scenario=scenario, config=cfg)
    for _ in range(steps):
        tw.step(dt)
    return tw


def _covers(tw):
    return np.array(
        [tw.entities[k].cover for k in sorted(tw.entities, key=str) if tw.entities[k].kind in SESSILE]
    )


def test_turbidity_toggle_changes_only_downstream(make_twin, cfg):
    scen = build_small_scenario(IdFactory(1), 21, cfg)
    base = float(scen.environment["turbidity"]["value"])
    pair = make_counterfactual_pair(scen, "environment.turbidity", base + 25.0)
    a, b = _run(make_twin, pair.factual, cfg), _run(make_twin, pair.counterfactual, cfg)
    # not downstream of turbidity: temperature and current are bit-identical
    assert np.array_equal(a.fields.local.temperature, b.fields.local.temperature)
    assert np.array_equal(a.fields.local.current, b.fields.local.current)
    # downstream: turbidity, light, and fouling cover via light limitation
    assert b.fields.local.turbidity.mean() > a.fields.local.turbidity.mean()
    assert b.fields.local.light.mean() < a.fields.local.light.mean() or a.fields.local.light.max() == 0.0
    assert not np.array_equal(_covers(a), _covers(b)) or a.fields.local.light.max() == 0.0


def test_temperature_toggle_leaves_turbidity_when_entity_feedback_off(make_twin, cfg):
    scen = build_small_scenario(IdFactory(2), 22, cfg)
    t = float(scen.environment["temperature"]["value"])
    pair = make_counterfactual_pair(scen, "environment.temperature", t + 6.0)
    c = baseline_config("one_way_field_to_entity", cfg)
    a, b = _run(make_twin, pair.factual, c), _run(make_twin, pair.counterfactual, c)
    assert np.array_equal(a.fields.local.turbidity, b.fields.local.turbidity)
    assert b.fields.local.temperature.mean() > a.fields.local.temperature.mean() + 5.0
    assert not np.array_equal(_covers(a), _covers(b))


def test_same_world_same_seed_counterfactual_is_identical_without_toggle(make_twin, cfg):
    scen = build_small_scenario(IdFactory(3), 23, cfg)
    t = scen.environment["temperature"]["value"]
    pair = make_counterfactual_pair(scen, "environment.temperature", t)
    a, b = _run(make_twin, pair.factual, cfg, 3), _run(make_twin, pair.counterfactual, cfg, 3)
    assert np.array_equal(_covers(a), _covers(b))
    assert pair.counterfactual.metadata["counterfactual_factor"] == "environment.temperature"


def test_confounded_variants(make_twin, cfg):
    ids = IdFactory(4)
    scen = build_small_scenario(ids, 24, cfg)
    hot = make_confounded_variant(scen, "HIGH_TEMPERATURE_STABLE_ECOLOGY", ids)
    sick = make_confounded_variant(scen, "NORMAL_TEMPERATURE_DISTURBED_ECOLOGY", ids)
    assert hot.environment["temperature"]["value"] > scen.environment["temperature"]["value"]
    assert sick.environment["temperature"]["value"] == scen.environment["temperature"]["value"]
    base, dis = _run(make_twin, scen, cfg, 10, 3600.0), _run(make_twin, sick, cfg, 10, 3600.0)
    assert _covers(dis).sum() < _covers(base).sum()
    for kind in CONFOUNDER_KINDS:
        tw = _run(make_twin, make_confounded_variant(scen, kind, ids), cfg, 2)
        assert validate_twin(tw) == []


def _event(ids, etype, target=None, **params):
    return ScenarioEvent(
        event_id=ids.new(), time_s=0.0, event_type=etype, target_entity_id=target, parameters=params
    )


def test_disturbance_then_recovery(make_twin):
    ids = IdFactory(5)
    tw = make_twin(seed=25)
    e = next(x for x in tw.entities.values() if x.kind == KIND_BIOFOULING)
    c0, k0 = e.cover, e.condition
    tw.step(1.0, [_event(ids, "PHYSICAL_DISTURBANCE", e.entity_id, cover_loss_fraction=0.8)])
    assert e.cover < c0 and e.condition < k0
    k1 = e.condition
    tw.step(1.0, [_event(ids, "RECOVERY_EVENT", e.entity_id, condition_gain=0.2)])
    assert e.condition > k1
    assert [r["status"] for r in tw.event_log] == ["APPLIED", "APPLIED"]


def test_field_events_and_logging(make_twin):
    ids = IdFactory(6)
    tw = make_twin(seed=26)
    before = tw.fields.local.turbidity.mean()
    tw.step(
        1.0,
        [
            _event(ids, "TURBIDITY_SPIKE", delta_ntu=20.0, units="NTU"),
            _event(ids, "CORROSION_ONSET"),
            _event(ids, "FLOW_CHANGE", current_m_s=[0.5, 0.0, 0.0], units="m s-1"),
        ],
    )
    assert tw.fields.local.turbidity.mean() > before + 5.0
    status = {r["event_type"]: r["status"] for r in tw.event_log}
    assert status["CORROSION_ONSET"] == "NOT_ECOLOGICAL"
    assert status["FLOW_CHANGE"] == "APPLIED"
    assert tw.fields.local.current[0].max() > 0.45
    with pytest.raises(ValueError):
        tw.step(1.0, [_event(ids, "TURBIDITY_SPIKE", delta_ntu=1.0, units="mg/L")])


def test_temperature_anomaly_stresses_and_scheduled_events_fire_once(make_twin, cfg):
    ids = IdFactory(7)
    scen = build_small_scenario(
        ids, 27, cfg, events=((100.0, "TEMPERATURE_ANOMALY", None, {"delta_c": 8.0, "duration_s": 86400.0}),)
    )
    tw = make_twin(scenario=scen)
    t0 = tw.fields.local.temperature.mean()
    for _ in range(8):
        tw.step(3600.0)
    assert len(tw.event_log) == 1
    assert tw.fields.local.temperature.mean() > t0 + 3.0


def test_ablated_disturbances_are_logged_not_dropped(make_twin, cfg):
    ids = IdFactory(8)
    tw = make_twin(seed=28, config=baseline_config("no_disturbances", cfg))
    e = next(x for x in tw.entities.values() if x.kind == KIND_BIOFOULING)
    c0 = e.cover
    tw.step(1.0, [_event(ids, "MORTALITY_EVENT", e.entity_id, mortality_fraction=0.9)])
    assert tw.event_log[0]["status"] == "IGNORED_ABLATED_DISTURBANCES"
    assert e.cover == pytest.approx(c0, abs=0.01)


@pytest.mark.parametrize("name", sorted(BASELINE_SWITCHES))
def test_every_baseline_runs_and_validates(make_twin, cfg, name):
    tw = _run(make_twin, None, baseline_config(name, cfg), 3)
    assert validate_twin(tw) == []


def test_static_baselines_freeze_their_part(make_twin, cfg):
    sf = make_twin(config=baseline_config("static_fields", cfg))
    t0 = sf.fields.local.temperature.copy()
    sf.step(600.0)
    assert np.array_equal(sf.fields.local.temperature, t0)
    se = make_twin(config=baseline_config("static_ecology", cfg))
    c0 = _covers(se)
    se.step(600.0)
    assert np.array_equal(_covers(se), c0)


def test_single_scale_leaves_regional_untouched(make_twin, cfg):
    tw = make_twin(config=baseline_config("single_scale_coupled", cfg))
    r0 = tw.fields.regional.turbidity.copy()
    tw.step(600.0)
    assert np.array_equal(tw.fields.regional.turbidity, r0)
    full = make_twin()
    r1 = full.fields.regional.turbidity.copy()
    full.step(600.0)
    assert not np.array_equal(full.fields.regional.turbidity, r1)


def test_independent_noise_ecology_ignores_environment(make_twin, cfg):
    c = baseline_config("independent_noise_ecology", cfg)
    scen = build_small_scenario(IdFactory(9), 29, c)
    pair = make_counterfactual_pair(scen, "environment.turbidity", 80.0)
    a, b = _run(make_twin, pair.factual, c, 3), _run(make_twin, pair.counterfactual, c, 3)
    assert np.array_equal(_covers(a), _covers(b))
