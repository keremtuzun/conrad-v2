from __future__ import annotations

import json
from uuid import UUID

import numpy as np
import pytest

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp
from conrad.schemas.world import DomainOwnership, Scenario, ScenarioEvent, WorldEntity
from conrad.twins.twin2t import (
    IMPLEMENTATION_METADATA,
    GeneratorKind,
    Twin2T,
    Twin2TConfig,
    build_small_scenario,
    populate_structural_state,
)
from conrad.twins.twin2t.priors import ALLOWED_PRIOR_SOURCES, SCENARIO_SUPPLIED
from conrad.twins.twin2t.scenario import StructuralScenarioError

DAY = 86400.0


def make_twin(tmp_path, scenario=None, seed=11, **cfg):
    twin = Twin2T(IdFactory(seed=seed + 100_000), ObjectStore(tmp_path / "obj"), Twin2TConfig(**cfg))
    twin.initialize(scenario or build_small_scenario(seed))
    return twin


def test_determinism_and_seed_sensitivity(tmp_path):
    runs = []
    for seed in (11, 11, 12):
        t = make_twin(tmp_path, build_small_scenario(5), seed=seed)
        t.reset(seed)
        for _ in range(6):
            t.step(30 * DAY)
        runs.append(json.dumps(t.export_domain_state()["truth_sequence"]))
    assert runs[0] == runs[1] and runs[0] != runs[2]


def test_truth_per_component_and_containers_excluded(tmp_path):
    t = make_twin(tmp_path)
    t.step(DAY)
    truths = t.get_truth(stamp(DAY, "sim"))
    types = {x.state["component_type"] for x in truths}
    assert "ASSET" not in types and "PIPELINE" not in types and "SEGMENT" in types
    assert all(x.domain.value == "TECHNICAL" for x in truths)
    concrete = [x for x in truths if x.state["material"] == "concrete"]
    assert concrete and not any(concrete[0].state["validity"].values())
    st = t.structural_truth(stamp(DAY, "sim"))
    assert set(st) == {
        "timestamp",
        "component_states",
        "mechanism_states",
        "relationships",
        "events",
        "environmental_context",
    }
    json.dumps(st)


def test_populate_logs_priors_and_keeps_supplied(tmp_path):
    raw = build_small_scenario(3, populate=False)
    seg = next(e for e in raw.world_entities if e.entity_type == "SEGMENT")
    ent = dict(raw.structural_state["entities"])
    ent[str(seg.id)] = {**ent[str(seg.id)], "wall_thickness_m": 0.0151}
    raw = Scenario(**{**dict(raw), "structural_state": {**raw.structural_state, "entities": ent}})
    full = populate_structural_state(raw, np.random.default_rng(0))
    assert full.structural_state["entities"][str(seg.id)]["wall_thickness_m"] == 0.0151
    for log in full.structural_state["prior_log"].values():
        for item in log:
            assert item["source"] in ALLOWED_PRIOR_SOURCES and item["citation"]
            assert item["name"] != "wall_thickness_m" or item["value"] != 0.0151
    t = make_twin(tmp_path, raw)
    plog = t.export_domain_state()["prior_log"]
    assert all(i["source"] != "MEASURED" for v in plog.values() for i in v)
    seg_params = t._asm.runtimes[seg.id].params.params  # truth-plane internals, test-only access
    assert seg_params["wall_thickness_m"].source == SCENARIO_SUPPLIED


def _entity(ids, etype, technical=True):
    return WorldEntity(
        id=ids.new(),
        entity_type=etype,
        reference_frame="WORLD",
        created_at=stamp(0, "sim"),
        domain_ownership=DomainOwnership(technical=technical),
    )


def test_refuses_state_for_absent_or_non_technical_entity(tmp_path):
    ids = IdFactory(seed=1)
    seg = _entity(ids, "SEGMENT")
    ghost = ids.new()
    bad = Scenario.model_construct(
        scenario_id=ids.new(),
        scenario_version="x",
        seed=1,
        metadata={},
        world_entities=(seg,),
        environment={},
        structural_state={"entities": {str(ghost): {"component_type": "SEGMENT"}}},
        ecological_state={},
        spatial_state={},
        events=(),
        robots=(),
        mission=None,
    )
    with pytest.raises(StructuralScenarioError):
        make_twin(tmp_path, bad)
    fish = _entity(ids, "SEGMENT", technical=False)
    s2 = Scenario(
        scenario_id=ids.new(),
        scenario_version="x",
        seed=1,
        world_entities=(seg, fish),
        structural_state={"entities": {str(fish.id): {"component_type": "SEGMENT"}}},
    )
    with pytest.raises(StructuralScenarioError):
        make_twin(tmp_path, s2)
    s3 = Scenario(
        scenario_id=ids.new(), scenario_version="x", seed=1, world_entities=(_entity(ids, "SPACESHIP"),)
    )
    with pytest.raises(StructuralScenarioError):
        make_twin(tmp_path, s3)


def test_missing_keys_sampled_for_foreign_scenario(tmp_path):
    ids = IdFactory(seed=4)
    seg = _entity(ids, "SEGMENT")
    s = Scenario(scenario_id=ids.new(), scenario_version="foreign", seed=4, world_entities=(seg,))
    t = make_twin(tmp_path, s)
    names = {i["name"] for i in t.export_domain_state()["prior_log"][str(seg.id)]}
    assert {"material", "wall_thickness_m", "stress_range_pa", "paris_C", "corrosion_A"} <= names


def test_scheduled_events_fire_and_unknown_types_ignored(tmp_path):
    base = build_small_scenario(7)
    seg = next(e.id for e in base.world_entities if e.entity_type == "SEGMENT")
    ids = IdFactory(seed=77)
    evs = (
        ScenarioEvent(event_id=ids.new(), time_s=10 * DAY, event_type="REPLACEMENT", target_entity_id=seg),
        ScenarioEvent(event_id=ids.new(), time_s=5 * DAY, event_type="FISH_SPAWN", target_entity_id=None),
    )
    t = make_twin(tmp_path, Scenario(**{**dict(base), "events": evs}))
    t.step(5 * DAY)
    assert not t.export_domain_state()["interventions"]
    t.step(6 * DAY)
    iv = t.export_domain_state()["interventions"]
    assert len(iv) == 1 and iv[0]["event_type"] == "REPLACEMENT" and UUID(iv[0]["entity_id"]) == seg
    t.step(DAY, [evs[0]])  # duplicate event id is not re-applied
    assert len(t.export_domain_state()["interventions"]) == 1


def test_generators_run_with_same_layout(tmp_path):
    for kind in GeneratorKind:
        t = make_twin(tmp_path, build_small_scenario(2), generator=kind.value)
        a0, m0 = t.truth_arrays()
        for _ in range(3):
            t.step(90 * DAY)
        a1, m1 = t.truth_arrays()
        assert a1.shape == a0.shape and (m1 == m0).all()
        assert (a1[~m1] == 0).all()
        if kind is GeneratorKind.RANDOM_STATIC_DEFECTS:
            assert np.array_equal(a0, a1)


def test_no_public_parameter_sharing_api():
    public = [n for n in dir(Twin2T) if not n.startswith("_")]
    assert not [n for n in public if "param" in n.lower() or "mechanism_model" in n.lower()]
    import conrad.twins.twin2t as pkg

    assert not [n for n in pkg.__all__ if "param" in n.lower() or "prior" in n.lower()]
    assert IMPLEMENTATION_METADATA["implementation_status"] == "EXPERIMENTAL_CANDIDATE"
    assert IMPLEMENTATION_METADATA["claim_status"] in ("NONE", "IMPLEMENTED")
