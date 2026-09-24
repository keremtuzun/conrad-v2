"""The mission constructs and updates a persisted spatial Model2T child."""

import math
from copy import deepcopy
from uuid import UUID

import numpy as np
import pytest

from conrad.active.candidates import SensorOption, look_at
from conrad.domains.technical.evidence import structured_evidence
from conrad.domains.technical.spatial_mission import MODEL_VERSION, SpatialMissionModel2T
from conrad.orchestration.association import StructuralAssociator, registry_of
from conrad.orchestration.children import build_children
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.orchestration.mission_predictive import SpatialMissionPredictive, mission_predictive_provider
from conrad.persistence.db import make_engine, migrate
from conrad.persistence.replay_store import ReplayIntegrityError
from conrad.persistence.repository import Repository
from conrad.schemas.frames import WORLD, Pose, quat_from_euler, quat_to_matrix
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import TimeStamp
from conrad.settings import load_settings
from conrad.sim.mission.replay import replay_run, validate_spatial_replay_contract
from conrad.sim.mission.run import prepare, replay_inputs
from conrad.sim.mission.spatial_support import capsule_basis
from conrad.sim.mission.world import MissionWorld
from tests.integration.test_spatial_mission_truth import _options


def test_spatial_model2t_child_initializes_and_persists_unknown_head(tmp_path):
    options = _options()
    assert options.spatial_sensor_model is not None
    sensor_model = options.spatial_sensor_model.model_copy(
        update={
            "position_uncertainty_m": 0.0,
            "orientation_uncertainty_rad": 0.0,
            "footprint_uncertainty_m": 0.0,
        }
    )
    options = options.model_copy(update={"survey_sigma_m": 0.0, "spatial_sensor_model": sensor_model})
    world = MissionWorld.build(401, "SPATIAL-BELIEF-DEV", options, UUID(int=61), tmp_path / "objects")
    db = tmp_path / "beliefs.sqlite"
    migrate(db)
    repo = Repository(make_engine(db))
    cfg = MissionRuntimeConfig(
        model2t_backend="spatial_v1",
        model2t_spatial={
            "axial_cells": 2,
            "sectors": 4,
            "sensor": sensor_model.model_dump(mode="json"),
            "thresholds": {
                "corrosion_degraded_m": 0.002,
                "corrosion_severe_m": 0.006,
                "corrosion_failed_m": 0.012,
                "crack_degraded_m": 0.003,
                "crack_severe_m": 0.01,
                "crack_failed_m": 0.03,
            },
            "required_looks": 1,
        },
    )
    children = build_children(
        world.context, cfg, IdFactory(401), world.store, repo, UUID(int=61), "sim", 0, False
    )
    assert isinstance(children.m2t, SpatialMissionModel2T)
    target = world.context.critical_component_ids[0]
    head = next(m for m in children.m2t.export_beliefs() if m.world_entity_id == target)
    assert head.model_version == MODEL_VERSION
    assert head.technical is not None and head.technical.condition is None
    assert repo.head(head.belief_id) is not None

    suite = world.hardware.suite
    assert suite is not None
    primitive = world.t2s.world.entities[world.t2s.world.index_of(world.target)].primitive
    a, b = np.asarray(primitive.a), np.asarray(primitive.b)  # type: ignore[attr-defined]
    radius = float(primitive.radius)  # type: ignore[attr-defined]
    _, normal, _ = capsule_basis(a, b)
    origin = (a + b) / 2 + (radius + 2.0) * normal
    forward = -normal
    yaw = math.atan2(float(forward[1]), float(forward[0])) - math.radians(options.structural.mount_yaw_deg)
    rotation = quat_from_euler(0.0, 0.0, yaw)
    body = origin - quat_to_matrix(rotation) @ np.asarray(options.structural.mount_position_m)
    pose = Pose(frame_id=WORLD, position_m=tuple(float(x) for x in body), orientation_wxyz=rotation)
    now = TimeStamp(time_ns=1_000_000_000, clock_domain="sim")
    observations = suite._structural(1.0, now, pose, pose, IdFactory(99).new())
    associator = StructuralAssociator(world.context, cfg.association)
    encoder_ids = IdFactory(500)
    matched = []
    for obs in observations:
        repo.put_observation(obs)
        ev, provenance = structured_evidence(obs, encoder_ids, None)
        ev, _ = associator.associate(obs, ev)
        repo.put_provenance(UUID(int=61), [provenance])
        repo.put_evidence(ev)
        if registry_of(ev) == target:
            matched.append(ev)
    assert matched
    children.m2t.ingest(matched)
    messages = children.m2t.update_beliefs(now)
    spatial = next(m for m in messages if m.world_entity_id == target)
    assert spatial.model_version == MODEL_VERSION
    assert spatial.technical is not None and spatial.technical.direct_support > 0
    assert any(c.observation_count > 0 for c in spatial.technical.local_cells)
    assert any(c.observation_count == 0 for c in spatial.technical.local_cells)
    assert spatial.technical.condition != "INTACT"
    assert repo.head(spatial.belief_id) is not None

    spec = world.context.sensor(world.context.structural_sensor_ids[0])
    boresight = spec.model_copy(
        update={"mount_pose": Pose(frame_id=spec.mount_pose.frame_id, position_m=(0, 0, 0))}
    )
    provider = mission_predictive_provider(children.m2t, children.m2s, boresight, 0.3)
    assert isinstance(provider, SpatialMissionPredictive)
    predictive = provider(children.m2t.export_beliefs())
    assert predictive is not None and len(predictive.cell_prior) == 8
    center = (a + b) / 2
    candidate = look_at(
        (float(origin[0]), float(origin[1]), float(origin[2])),
        (float(center[0]), float(center[1]), float(center[2])),
        WORLD,
    )
    option = SensorOption(sensor_id=spec.sensor_id, modality=spec.modality, min_range_m=0.3, max_range_m=4.0)
    predicted_weights = predictive.cell_weights(candidate, option)
    assert np.any(predicted_weights > 0)
    assert np.any(predicted_weights == 0)
    no_map = SpatialMissionPredictive(children.m2t, None, boresight, 0.3, provider.cfg)
    no_map_predictive = no_map(children.m2t.export_beliefs())
    assert no_map_predictive is not None
    assert not np.any(no_map_predictive.cell_weights(candidate, option) > 0)

    inputs = replay_inputs(
        "SPATIAL-BELIEF-DEV",
        load_settings("configs/sim/mission_test_small.yaml"),
        world,
        cfg.model_dump(mode="json"),
        options.model_dump(mode="json"),
        "spatial-development",
    )
    assert len(inputs["git_commit"]) == 40
    validate_spatial_replay_contract(inputs)
    for key, value in (
        ("sensor_config_digest", "0" * 64),
        ("support", "structural-observation-v3"),
        ("model2t", "model2t-spatial-v2"),
    ):
        edited = deepcopy(inputs)
        edited["spatial_versions"][key] = value
        with pytest.raises(ReplayIntegrityError, match="version mismatch"):
            validate_spatial_replay_contract(edited)

    short_cfg = cfg.model_copy(update={"duration_s": 4.0, "control_period_s": 0.1, "model2e_enabled": False})
    session = prepare(
        "GOLDEN-SMOKE",
        load_settings("configs/sim/mission_test_small.yaml"),
        run_id="SPATIAL-DEV-SMOKE",
        runs_root=tmp_path / "runs",
        stored_world=options,
        stored_runtime=short_cfg,
    )
    session.run()
    result = session.finish()
    replay = replay_run(tmp_path / "runs" / "SPATIAL-DEV-SMOKE", tmp_path / "replayed")
    assert result["run_id"] == "SPATIAL-DEV-SMOKE"
    assert replay["equal"]
