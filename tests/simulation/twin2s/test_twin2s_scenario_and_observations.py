"""Shared scenario consistency, observation contract, zero truth leakage, determinism, dynamics."""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pytest
from twin2s_sim_helpers import make_ctx, make_twin

from conrad.schemas.frames import WORLD, Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, SensorHealth
from conrad.schemas.timebase import stamp
from conrad.schemas.world import Scenario, ScenarioEvent
from conrad.sim.scenarios.pipeline_inspection import build_pipeline_inspection_scenario
from conrad.twins.twin2s.families import OOD_FAMILIES, WORLD_FAMILIES, family_split, generate_scenario
from conrad.twins.twin2s.world import EVENT_DISPLACE, EVENT_REMOVE


def test_shared_scenario_sections_keyed_by_same_world_entity_ids(t2s_scenario):
    s = t2s_scenario
    ids = {e.id for e in s.world_entities}
    spatial = {UUID(k) for k in s.spatial_state["entities"]}
    structural = s.structural_state["entities"]
    assert spatial <= ids and {UUID(k) for k in structural} <= spatial
    for v in structural.values():
        assert set(v) == {"material", "component_type"}
    types = {v["component_type"] for v in structural.values()}
    assert {"pipeline_segment", "weld", "support"} <= types
    assert all(UUID(k) in ids for k in s.ecological_state["entities"])
    mods = {x.modality for x in s.robots[0].sensors}
    assert mods == {"RGB", "SONAR", "DEPTH_RANGE", "IMU", "PRESSURE_DEPTH"}
    assert s.mission is not None and set(s.mission.target_entity_ids) <= ids
    # round-trips through the frozen contract (validator re-checks section keys)
    assert Scenario.model_validate_json(s.model_dump_json()) == s


def test_scenario_is_deterministic_and_seed_sensitive():
    a = build_pipeline_inspection_scenario(3, IdFactory(3))
    b = build_pipeline_inspection_scenario(3, IdFactory(3))
    c = build_pipeline_inspection_scenario(4, IdFactory(3))
    assert a.content_digest() == b.content_digest() != c.content_digest()


@pytest.mark.parametrize("family", WORLD_FAMILIES)
def test_every_world_family_builds(family):
    s = generate_scenario(5, IdFactory(5), family)
    assert s.metadata["split"] == family_split(family)
    assert (family in OOD_FAMILIES) == (s.metadata["split"] == "ood")
    assert len(s.spatial_state["entities"]) >= 5


def test_observations_carry_no_truth(t2s_scenario, tmp_path):
    tw = make_twin(t2s_scenario, tmp_path)
    true_ids = {str(e.id) for e in t2s_scenario.world_entities}
    true = t2s_scenario.robots[0].initial_pose
    est = Pose(
        frame_id=WORLD,
        position_m=(true.position_m[0] + 0.3, *true.position_m[1:]),
        orientation_wxyz=true.orientation_wxyz,
    )
    samples = []
    for sensor in t2s_scenario.robots[0].sensors:
        for k in range(3):
            samples += tw.generate_observation(make_ctx(t2s_scenario, sensor, 0.5 + 0.2 * k, true, est))
    mods = {s.observation.modality for s in samples}
    assert {
        Modality.RGB,
        Modality.SONAR,
        Modality.DEPTH_RANGE,
        Modality.POINT_CLOUD,
        Modality.IMU,
        Modality.PRESSURE_DEPTH,
    } <= mods
    for s in samples:
        js = s.observation.model_dump_json()
        assert not any(u in js for u in true_ids), "hidden world-entity id leaked into an Observation"
        assert s.observation.robot_pose_estimate == est  # never the true pose
        assert set(s.observation.sensor_context) == {"settings", "self_reported_health", "encoding"}
        assert s.supervision is not None and s.supervision.observation_id == s.observation.observation_id
        assert s.supervision.lineage.startswith("pipeline_inspection/")
    depth = next(s for s in samples if s.observation.modality == Modality.DEPTH_RANGE)
    tgt = depth.supervision.targets
    ent = tw.store.get_array(tgt["arrays"]["entity_index"])
    assert ent.shape == (48, 64) and (ent >= 0).any()
    assert set(tgt["entity_id_table"]) <= true_ids  # identity lives ONLY in supervision


def test_same_seed_gives_byte_identical_digests(t2s_scenario, tmp_path):
    def run(root):
        tw = make_twin(t2s_scenario, root)
        out = []
        for sensor in t2s_scenario.robots[0].sensors[:3]:
            out += [
                s.observation.payload_ref.digest
                for s in tw.generate_observation(make_ctx(t2s_scenario, sensor))
            ]
        return out

    a, b = run(tmp_path / "a"), run(tmp_path / "b")
    assert a == b and len(a) == 4
    tw = make_twin(t2s_scenario, tmp_path / "c")
    tw.reset(12345)
    depth = next(x for x in t2s_scenario.robots[0].sensors if x.modality == "DEPTH_RANGE")
    other = tw.generate_observation(make_ctx(t2s_scenario, depth))[0].observation.payload_ref.digest
    assert other not in a  # a different noise seed changes the measurement


def test_degradation_changes_measurements_and_health(t2s_scenario, tmp_path):
    tw = make_twin(t2s_scenario, tmp_path)
    sensors = {x.modality: x for x in t2s_scenario.robots[0].sensors}
    rgb = [
        tw.store.get_array(
            tw.generate_observation(make_ctx(t2s_scenario, sensors["RGB"], degradation=d))[
                0
            ].observation.payload_ref
        ).astype(float)
        for d in ({}, {"turbidity": 1.0, "blur": 1.0})
    ]
    assert np.abs(np.diff(rgb[1], axis=1)).mean() < np.abs(np.diff(rgb[0], axis=1)).mean()  # blurrier
    depth = tw.generate_observation(
        make_ctx(t2s_scenario, sensors["DEPTH_RANGE"], degradation={"dropout": 0.5})
    )
    assert depth[0].observation.sensor_health == SensorHealth.DEGRADED
    img = tw.store.get_array(depth[0].observation.payload_ref)
    assert np.isnan(img).mean() > 0.4
    assert tw.generate_observation(make_ctx(t2s_scenario, sensors["SONAR"], degradation={"fault": 1.0})) == []
    with pytest.raises(ValueError):
        tw.generate_observation(make_ctx(t2s_scenario, sensors["SONAR"], degradation={"bogus": 1.0}))


def test_step_dynamics_events_truth_and_reset(t2s_scenario, tmp_path):
    tw = make_twin(t2s_scenario, tmp_path)
    groups = t2s_scenario.spatial_state["groups"]
    seg = UUID(groups["segments"][0])
    ids = IdFactory(1)
    tw.step(
        2.0,
        [
            ScenarioEvent(
                event_id=ids.new(),
                time_s=2.0,
                event_type=EVENT_DISPLACE,
                target_entity_id=seg,
                parameters={"offset_m": [0.0, 0.5, 0.0]},
            ),
            ScenarioEvent(
                event_id=ids.new(),
                time_s=2.0,
                event_type=EVENT_REMOVE,
                target_entity_id=UUID(groups["rocks"][0]),
            ),
        ],
    )
    truth = {t.world_entity_id: t for t in tw.get_truth(stamp(2.0, "SIM"))}
    assert truth[seg].state["offset_m"] == [0.0, 0.5, 0.0]
    assert truth[UUID(groups["rocks"][0])].state["active"] is False
    assert len(tw.export_domain_state()["change_log"]) == 2
    if groups["dynamic"]:
        assert truth[UUID(groups["dynamic"][0])].state["dynamic"] is True
    tw.reset(11)
    truth0 = {t.world_entity_id: t for t in tw.get_truth(stamp(0.0, "SIM"))}
    assert truth0[seg].state["offset_m"] == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        tw.step(-1.0)
