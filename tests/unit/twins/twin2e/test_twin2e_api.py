from __future__ import annotations

import numpy as np
import pytest

from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import ClockDomainError, stamp
from conrad.schemas.world import Domain, DomainOwnership, Scenario, WorldEntity
from conrad.twins.twin2e import (
    IMPLEMENTATION_METADATA,
    LearnedResidualNotCalibratedError,
    Twin2E,
    Twin2ENotInitializedError,
    Twin2EScenarioError,
    build_small_scenario,
    populate_ecological_state,
    validate_twin,
)
from conrad.twins.twin2e.config import LearnedResidualConfig
from conrad.twins.twin2e.fields import FIELD_UNITS
from conrad.twins.twin2e.validators import state_digest


def test_metadata_contract():
    m = IMPLEMENTATION_METADATA
    assert m["implementation_status"] == "EXPERIMENTAL_CANDIDATE"
    assert m["claim_status"] == "IMPLEMENTED"
    assert any("does NOT claim" in a for a in m["assumptions"])
    assert "full_meife" in m["baselines"]


def test_initialize_step_truth(make_twin):
    tw = make_twin()
    for _ in range(4):
        tw.step(300.0)
    assert validate_twin(tw) == []
    truth = tw.get_truth(tw.now())
    assert len(truth) == len(tw.entities) == 5
    assert all(t.domain is Domain.ECOLOGICAL for t in truth)
    sessile = [t for t in truth if "cover" in t.state]
    assert sessile and all(0 <= t.state["cover"] <= 1 for t in sessile)
    assert all(t.state["exposed_to_units"]["temperature"] == "degC" for t in truth)


def test_truth_time_and_clock_domain_are_checked(make_twin):
    tw = make_twin()
    tw.step(60.0)
    with pytest.raises(ValueError):
        tw.get_truth(stamp(0.0, "SIM"))
    with pytest.raises(ClockDomainError):
        tw.get_truth(stamp(60.0, "OTHER"))


def test_uninitialized_twin_refuses(store):
    tw = Twin2E(IdFactory(1), store)
    with pytest.raises(Twin2ENotInitializedError):
        tw.step(1.0)
    with pytest.raises(Twin2ENotInitializedError):
        tw.reset(1)


def test_determinism_and_reset(make_twin):
    a, b = make_twin(seed=5), make_twin(seed=5)
    for _ in range(3):
        a.step(400.0)
        b.step(400.0)
    assert state_digest(a) == state_digest(b)
    d = state_digest(a)
    a.reset(5)
    for _ in range(3):
        a.step(400.0)
    assert state_digest(a) == d
    a.reset(6)
    for _ in range(3):
        a.step(400.0)
    assert state_digest(a) != d


def test_export_domain_state_stores_fields_with_units_and_geometry(make_twin, store):
    tw = make_twin()
    tw.step(120.0)
    ex = tw.export_domain_state()
    assert set(ex) >= {"Z", "Phi", "event_log", "prior_log", "switches", "config_digest"}
    for key, rec in ex["Phi"].items():
        name = key.split("/")[1]
        assert rec["units"] == FIELD_UNITS[name]
        assert rec["grid"]["frame_id"] == "WORLD"
        arr = store.get_array(rec["values_ref"]["digest"])
        assert tuple(arr.shape[-3:]) == tuple(rec["grid"]["shape"])
    assert ex["learned_residual_active"] is False
    assert all(p["label"] in {"ENGINEERING_ESTIMATE", "SYNTHETIC_ONLY", "DERIVED"} for p in ex["prior_log"])


def test_field_shapes_follow_config(make_twin, cfg):
    tw = make_twin()
    assert tw.fields.local.temperature.shape == tuple(cfg.local_grid.shape)
    assert tw.fields.regional.current.shape == (3, *cfg.regional_grid.shape)


def _scenario_with_foreign_state(ids, own_ecological: bool):
    t0 = stamp(0.0, "SIM")
    e = WorldEntity(
        id=ids.new(),
        entity_type="pipeline_segment",
        reference_frame="WORLD",
        created_at=t0,
        domain_ownership=DomainOwnership(ecological=own_ecological),
    )
    return Scenario(
        scenario_id=ids.new(),
        scenario_version="x",
        seed=1,
        world_entities=(e,),
        ecological_state={"entities": {str(e.id): {"initial_cover": 0.2}}},
    )


def test_refuses_state_for_non_ecological_entity():
    s = _scenario_with_foreign_state(IdFactory(2), own_ecological=False)
    with pytest.raises(Twin2EScenarioError):
        populate_ecological_state(s, np.random.default_rng(0))


def test_scenario_schema_refuses_unknown_entity():
    ids = IdFactory(3)
    with pytest.raises(ValueError):
        Scenario(
            scenario_id=ids.new(),
            scenario_version="x",
            seed=1,
            world_entities=(),
            ecological_state={"entities": {str(ids.new()): {}}},
        )


def test_environment_requires_units(cfg):
    ids = IdFactory(4)
    with pytest.raises(Twin2EScenarioError):
        build_small_scenario(ids, 1, cfg, environment={"temperature": 12.0})
    with pytest.raises(Twin2EScenarioError):
        build_small_scenario(ids, 1, cfg, environment={"temperature": {"value": 285.0, "units": "K"}})
    s = build_small_scenario(ids, 1, cfg, environment={"temperature": {"value": 12.0, "units": "degC"}})
    assert s.environment["temperature"]["value"] == 12.0
    assert not any(p["path"] == "environment.temperature" for p in s.ecological_state["prior_log"])


def test_prior_sampling_is_logged_and_can_be_disabled(cfg, store):
    ids = IdFactory(5)
    raw = build_small_scenario(ids, 1, cfg, populate=False)
    tw = Twin2E(IdFactory(6), store, cfg)
    tw.initialize(raw)
    log = tw.export_domain_state()["prior_log"]
    assert any(p["path"] == "environment.turbidity" for p in log)
    strict = Twin2E(IdFactory(6), store, cfg.model_copy(update={"allow_prior_sampling": False}))
    with pytest.raises(Twin2EScenarioError):
        strict.initialize(raw)


def test_learned_residual_disabled_and_refuses_without_calibration(cfg, store):
    bad = cfg.model_copy(update={"learned_residual": LearnedResidualConfig(enabled=True)})
    with pytest.raises(LearnedResidualNotCalibratedError):
        Twin2E(IdFactory(1), store, bad)
    no_fn = cfg.model_copy(
        update={"learned_residual": LearnedResidualConfig(enabled=True, calibration_data_ref="x")}
    )
    with pytest.raises(LearnedResidualNotCalibratedError):
        Twin2E(IdFactory(1), store, no_fn)


def test_biofouling_never_writes_technical_state(make_twin):
    tw = make_twin()
    tw.step(600.0)
    for t in tw.get_truth(tw.now()):
        assert not any("corrosion" in k for k in t.state)
