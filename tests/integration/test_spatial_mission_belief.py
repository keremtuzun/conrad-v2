"""The mission constructs and updates a persisted spatial Model2T child."""

import json
import math
import sqlite3
from copy import deepcopy
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from conrad.active.candidates import SensorOption, look_at
from conrad.decision.context import DecisionContext, MissionRequirement
from conrad.decision.egdc import EGDC
from conrad.domains.technical.evidence import structured_evidence
from conrad.domains.technical.spatial_mission import MODEL_VERSION, SpatialMissionModel2T
from conrad.orchestration.association import StructuralAssociator, registry_of
from conrad.orchestration.children import build_children
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.orchestration.mission_predictive import SpatialMissionPredictive, mission_predictive_provider
from conrad.persistence.db import make_engine, migrate
from conrad.persistence.replay_store import ReplayIntegrityError
from conrad.persistence.repository import Repository
from conrad.schemas.belief import Availability, BeliefSnapshot
from conrad.schemas.decision import MissionPhase, MissionState
from conrad.schemas.frames import WORLD, Pose, matrix_to_quat, quat_from_euler, quat_to_matrix
from conrad.schemas.ids import IdFactory
from conrad.schemas.structural_sensor import StructuralSensorModel
from conrad.schemas.structural_support import CapsuleSurfaceSupport, ParameterAuthority
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain
from conrad.settings import load_settings
from conrad.sim.mission.options import SpatialTruthOptions
from conrad.sim.mission.replay import compare, replay_run, validate_spatial_replay_contract
from conrad.sim.mission.run import prepare, replay_inputs, validate_spatial_mission_selection
from conrad.sim.mission.spatial_support import capsule_basis
from conrad.sim.mission.world import MissionWorld
from tests.integration.test_spatial_mission_truth import _options
from tests.leakage.test_dynamic_leakage import TWIN_ONLY_KEYS, _keys, _runtime_texts


@pytest.mark.parametrize(("survey_sigma_m", "credited"), [(0.0, True), (0.02, False)])
def test_spatial_observation_reaches_model2t_through_mission_perception(tmp_path, survey_sigma_m, credited):
    options = _options()
    assert options.spatial_sensor_model is not None
    sensor = options.spatial_sensor_model.model_copy(
        update={
            "position_uncertainty_m": 0.0,
            "orientation_uncertainty_rad": 0.0,
            "footprint_uncertainty_m": 0.0,
        }
    )
    options = options.model_copy(
        update={
            "family": "pipeline_with_supports",
            "survey_sigma_m": survey_sigma_m,
            "ecological_enabled": False,
            "spatial_sensor_model": sensor,
        }
    )
    cfg = MissionRuntimeConfig(
        duration_s=1.0,
        model2e_enabled=False,
        model2t_backend="spatial_v1",
        model2t_spatial={
            "axial_cells": 2,
            "sectors": 4,
            "sensor": sensor.model_dump(mode="json"),
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
    session = prepare(
        "GOLDEN-SMOKE",
        load_settings("configs/sim/mission_test_small.yaml"),
        run_id="SPATIAL-PERCEPTION-DEV",
        runs_root=tmp_path,
        stored_world=options,
        stored_runtime=cfg,
        capture=False,
    )
    try:
        suite = session.world.hardware.suite
        assert suite is not None
        pose = Pose(
            frame_id=WORLD,
            position_m=(0.13, 2.52, 1.51),
            orientation_wxyz=quat_from_euler(0.0, 0.0, math.pi),
            covariance_6x6=(0.0,) * 36,
        )
        now = TimeStamp(time_ns=1_000_000_000, clock_domain="SIM")
        observations = suite._structural(1.0, now, pose, pose, IdFactory(99).new())
        spatial = [obs for obs in observations if obs.structural_support is not None]
        assert spatial
        if not credited:
            assert all(
                obs.structural_support is not None
                and obs.structural_support.frame_id == "CAPSULE_UNREGISTERED"
                for obs in spatial
            )
            assert all(
                obs.structural_support is not None and obs.structural_support.axial_uncertainty_m is None
                for obs in spatial
            )
        session.runtime.perception.process(spatial, now)
        target = session.world.context.critical_component_ids[0]
        head = next(m for m in session.runtime.m2t.export_beliefs() if m.world_entity_id == target)
        assert head.technical is not None
        assert any(cell.observation_count > 0 for cell in head.technical.local_cells) is credited
        if credited:
            assert any(row["registry_id"] == str(target) for row in session.runtime.perception.structural_log)
        else:
            assert head.technical.condition is None
            assert isinstance(session.runtime.m2t, SpatialMissionModel2T)
            unresolved = session.runtime.m2t.spatial.unresolved
            assert unresolved and len(unresolved) == len(set(unresolved))
    finally:
        session.finish()


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
    validate_spatial_mission_selection(options, cfg)
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
    assert spec.parameters["min_incidence_cos"] == world.t2s.cfg.observed.min_incidence_cos
    assert spec.parameters["min_quality"] == world.t2s.cfg.observed.min_quality
    assert spec.parameters["water_attenuation_per_m"] == world.t2s.cfg.water_attenuation_per_m
    boresight = spec.model_copy(
        update={"mount_pose": Pose(frame_id=spec.mount_pose.frame_id, position_m=(0, 0, 0))}
    )
    provider = mission_predictive_provider(children.m2t, children.m2s, boresight, 0.3)
    assert isinstance(provider, SpatialMissionPredictive)
    predictive = provider(children.m2t.export_beliefs())
    assert predictive is not None and len(predictive.cell_prior) == 8
    assert len(predictive.candidate_regions) == 4
    center = (a + b) / 2
    candidate = look_at(
        (float(origin[0]), float(origin[1]), float(origin[2])),
        (float(center[0]), float(center[1]), float(center[2])),
        WORLD,
    )
    option = SensorOption(sensor_id=spec.sensor_id, modality=spec.modality, min_range_m=0.3, max_range_m=4.0)
    predicted_weights = predictive.cell_weights(candidate, option)
    # This exact pose has already supplied its certifiable tiles. More looks
    # there must not be mistaken for new spatial coverage.
    assert not np.any(predicted_weights > 0)
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
        ("registration", "unsafe-assumed-exact-v0"),
        ("visibility_certificate", "capsule-sdf-old-v0"),
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
    assert replay["observation_payloads"]["equal"]
    assert replay["evidence_payloads"]["equal"]
    assert replay["trajectories"]["equal"]

    replayed = tmp_path / "replayed" / "SPATIAL-DEV-SMOKE"
    with sqlite3.connect(replayed / "conrad.sqlite") as con:
        observation_id, payload = con.execute(
            "SELECT observation_id, payload_json FROM observations ORDER BY observation_id LIMIT 1"
        ).fetchone()
        edited = json.loads(payload)
        edited["replay_mutation"] = True
        con.execute(
            "UPDATE observations SET payload_json = ? WHERE observation_id = ?",
            (json.dumps(edited), observation_id),
        )
    comparison = compare(tmp_path / "runs" / "SPATIAL-DEV-SMOKE", replayed)
    assert comparison["events"]["equal"]
    assert not comparison["observation_payloads"]["equal"]
    assert not comparison["equal"]

    run_dir = tmp_path / "runs" / "SPATIAL-DEV-SMOKE"
    truth = json.loads((run_dir / "truth" / "truth_record.json").read_text(encoding="utf-8"))
    world_ids = set(truth["meta"]["world_entity_ids"])
    assert world_ids
    scanned = 0
    for where, payload in _runtime_texts(run_dir):
        scanned += 1
        assert not any(world_id in payload for world_id in world_ids), where
        assert "visibility:" not in payload, where
        for line in payload.splitlines() or [payload]:
            try:
                keys = _keys(json.loads(line), set())
            except json.JSONDecodeError:
                continue
            assert not keys.intersection(TWIN_ONLY_KEYS), where
    assert scanned > 100


@pytest.mark.slow
@pytest.mark.parametrize(
    ("state", "expected", "full_sweep"),
    [
        pytest.param(None, "INTACT", True, id="healthy"),
        pytest.param(None, None, False, id="partial-healthy"),
        pytest.param("outside-support", None, False, id="defect-outside-measured-support"),
        pytest.param("occluded-defect", None, False, id="occluded-local-defect"),
        pytest.param({"corrosion_depth_m": 0.008}, "SEVERE", True, id="covered-corrosion"),
        pytest.param({"crack_length_m": 0.02, "crack_depth_m": 0.006}, "SEVERE", True, id="covered-crack"),
        pytest.param("subresolution", "INTACT", True, id="subresolution-qualified-intact"),
        pytest.param("heterogeneous-healthy", "INTACT", True, id="heterogeneous-healthy"),
        pytest.param("multiple-defects", "SEVERE", True, id="separated-corrosion-and-crack"),
        pytest.param("edge-defect", "SEVERE", True, id="required-domain-edge-defect"),
        pytest.param("uniform-same-mean", "INTACT", True, id="uniform-same-mean"),
        pytest.param("local-same-mean", "SEVERE", True, id="local-same-mean"),
    ],
)
def test_required_surface_identifies_healthy_and_resolvable_defects(tmp_path, state, expected, full_sweep):
    base = _options()
    assert base.spatial_sensor_model is not None
    sensor = base.spatial_sensor_model.model_copy(
        update={
            "position_uncertainty_m": 0.0,
            "orientation_uncertainty_rad": 0.0,
            "footprint_uncertainty_m": 0.0,
        }
    )
    truth_config: dict[str, Any] = {"axial_cells": 2, "sectors": 4}
    if state == "uniform-same-mean":
        truth_config["base"] = {"corrosion_depth_m": 0.001}
    elif state == "heterogeneous-healthy":
        truth_config["cell_states"] = [{"corrosion_depth_m": 0.0002 * (index % 4)} for index in range(8)]
    elif state == "multiple-defects":
        truth_config["cell_states"] = [
            {"corrosion_depth_m": 0.008}
            if index == 3
            else {"crack_length_m": 0.02, "crack_depth_m": 0.006}
            if index == 6
            else {}
            for index in range(8)
        ]
    elif state == "edge-defect":
        truth_config["patches"] = [
            {
                "axial_start_fraction": 0.055,
                "axial_end_fraction": 0.12,
                "angle_start_rad": 3.7,
                "angle_end_rad": 4.2,
                "state": {"corrosion_depth_m": 0.008},
            }
        ]
    elif state == "local-same-mean":
        truth_config["cell_states"] = [
            {"corrosion_depth_m": 0.008} if index == 6 else {} for index in range(8)
        ]
    elif state == "outside-support":
        truth_config["patches"] = [
            {
                "axial_start_fraction": 0.3,
                "axial_end_fraction": 0.38,
                "angle_start_rad": 0.3,
                "angle_end_rad": 0.8,
                "state": {"corrosion_depth_m": 0.008},
            }
        ]
    elif state == "occluded-defect":
        truth_config["patches"] = [
            {
                "axial_start_fraction": 0.3,
                "axial_end_fraction": 0.38,
                "angle_start_rad": 3.9,
                "angle_end_rad": 4.2,
                "state": {"corrosion_depth_m": 0.008},
            }
        ]
    elif state == "subresolution":
        truth_config["patches"] = [
            {
                "axial_start_fraction": 0.3,
                "axial_end_fraction": 0.305,
                "angle_start_rad": 3.7,
                "angle_end_rad": 3.75,
                "state": {"corrosion_depth_m": 0.008},
            }
        ]
    elif state is not None:
        truth_config["patches"] = [
            {
                "axial_start_fraction": 0.3,
                "axial_end_fraction": 0.38,
                "angle_start_rad": 3.7,
                "angle_end_rad": 4.2,
                "state": state,
            }
        ]
    truth = SpatialTruthOptions.model_validate(truth_config)
    changes: dict[str, Any] = {
        "survey_sigma_m": 0.0,
        "family": "pipeline_with_supports",
        "spatial_truth": truth,
        "spatial_sensor_model": sensor,
    }
    if state == "occluded-defect":
        changes["family"] = "straight_pipeline"
        changes["occlusion"] = base.occlusion.model_copy(
            update={
                "enabled": True,
                "azimuth_offset_deg": (0.0, 0.0),
                "window_half_deg": (20.0, 20.0),
                "standoff_m": (0.5, 0.5),
                "half_width_m": (0.4, 0.4),
                "half_length_fraction": (0.5, 0.5),
                "axial_shift_fraction": (0.0, 0.0),
            }
        )
    options = base.model_copy(update=changes)
    world = MissionWorld.build(401, "SPATIAL-HEALTHY-DEV", options, UUID(int=162), tmp_path / "objects")
    clear_world = None
    if state == "occluded-defect":
        clear_options = options.model_copy(
            update={"occlusion": options.occlusion.model_copy(update={"enabled": False})}
        )
        clear_world = MissionWorld.build(
            401, "SPATIAL-CLEAR-DEV", clear_options, UUID(int=164), tmp_path / "clear_objects"
        )
    if state in ("uniform-same-mean", "local-same-mean"):
        field = world.t2t.spatial_fields[world.target]
        assert sum(cell.corrosion_depth_m for cell in field.base) / len(field.base) == pytest.approx(0.001)
        coarse = StructuralSensorModel(
            footprint_width_m=field.grid.length_m,
            footprint_height_m=2 * math.pi * field.grid.radius_m,
            minimum_resolvable_corrosion_m=0.05,
            minimum_resolvable_crack_m=0.05,
            range_min_m=0.3,
            range_max_m=4.0,
            noise_sigma_m=0.0,
            aggregation_kernel="AREA_MEAN",
            authority=ParameterAuthority.ENGINEERING_ESTIMATE,
        )
        whole = CapsuleSurfaceSupport(
            sensor_model_version=coarse.version,
            sensor_config_digest=coarse.digest,
            frame_id="CAPSULE_DESIGN",
            axial_start_m=0.0,
            axial_end_m=field.grid.length_m,
            angle_start_rad=0.0,
            angle_end_rad=2 * math.pi,
            aggregation_kernel="AREA_MEAN",
        )
        assert field.measure(whole, coarse).corrosion_depth_m == pytest.approx(0.001)
    if state in ("outside-support", "occluded-defect", "subresolution"):
        assert world.t2t.spatial_fields[world.target].worst_local().corrosion_depth_m == 0.008
    assert world.hardware.suite is not None
    cfg = MissionRuntimeConfig(
        model2t_backend="spatial_v1",
        model2t_spatial={
            "axial_cells": 2,
            "sectors": 4,
            "sensor": sensor.model_dump(mode="json"),
            "thresholds": {
                "corrosion_degraded_m": 0.002,
                "corrosion_severe_m": 0.006,
                "corrosion_failed_m": 0.012,
                "crack_degraded_m": 0.003,
                "crack_severe_m": 0.01,
                "crack_failed_m": 0.03,
            },
            "required_looks": 1,
            "required_domain": {"axial_fraction": [0.05, 0.95], "sectors": [2, 3]},
        },
    )
    validate_spatial_mission_selection(options, cfg)
    db = tmp_path / "healthy.sqlite"
    migrate(db)
    repo = Repository(make_engine(db))
    children = build_children(
        world.context, cfg, IdFactory(401), world.store, repo, UUID(int=162), "sim", 0, False
    )
    assert isinstance(children.m2t, SpatialMissionModel2T)
    primitive = world.t2s.world.entities[world.t2s.world.index_of(world.target)].primitive
    a, b = np.asarray(primitive.a), np.asarray(primitive.b)  # type: ignore[attr-defined]
    radius = float(primitive.radius)  # type: ignore[attr-defined]
    d, u, v = capsule_basis(a, b)
    spec = world.context.sensor(world.context.structural_sensor_ids[0])
    mount = quat_to_matrix(spec.mount_pose.orientation_wxyz)
    associator = StructuralAssociator(world.context, cfg.association)
    ids = IdFactory(163)
    now = TimeStamp(time_ns=1_000_000_000, clock_domain="sim")
    accepted = 0
    xs = (0.5, 1.5, 2.5, 3.5) if full_sweep else (1.5,)
    angles = (
        np.arange(0, 2 * math.pi, math.pi / 8)
        if full_sweep
        else (4.05 if state == "occluded-defect" else math.pi,)
    )
    for x in xs:
        for angle in angles:
            normal = math.cos(angle) * u + math.sin(angle) * v
            origin = a + x * d + (radius + (2.0 if state == "occluded-defect" else 0.5)) * normal
            forward = -normal
            side = np.cross(np.array([0.0, 0.0, 1.0]), forward)
            side /= np.linalg.norm(side)
            sensor_rot = np.column_stack((forward, side, np.cross(forward, side)))
            body_rot = sensor_rot @ mount.T
            body = origin - body_rot @ np.asarray(spec.mount_pose.position_m)
            pose = Pose(
                frame_id=WORLD,
                position_m=tuple(float(value) for value in body),
                orientation_wxyz=matrix_to_quat(body_rot),
                covariance_6x6=(0.0,) * 36,
            )
            trace = ids.new()
            observations = world.hardware.suite._structural(1.0, now, pose, pose, trace)
            if clear_world is not None:
                assert clear_world.hardware.suite is not None
                clear_observations = clear_world.hardware.suite._structural(1.0, now, pose, pose, trace)
                assert any(
                    obs.inline_values is not None and obs.inline_values[0] >= 0.008
                    for obs in clear_observations
                )
                assert not observations
            for observation in observations:
                repo.put_observation(observation)
                ev, provenance = structured_evidence(observation, ids, None)
                ev, _ = associator.associate(observation, ev)
                repo.put_provenance(UUID(int=162), [provenance])
                repo.put_evidence(ev)
                if registry_of(ev) == world.context.critical_component_ids[0]:
                    children.m2t.ingest([ev])
                    accepted += 1
    if state == "occluded-defect":
        assert accepted == 0
    else:
        assert accepted > (100 if full_sweep else 0)
    updated = children.m2t.update_beliefs(now)
    target = next(
        m
        for m in (children.m2t.export_beliefs() if state == "occluded-defect" else updated)
        if m.world_entity_id == world.context.critical_component_ids[0]
    )
    assert target.technical is not None
    if full_sweep:
        assert target.technical.direct_support == pytest.approx(1.0)
    else:
        assert target.technical.direct_support < 1.0
    assert target.technical.condition == expected
    if not full_sweep:
        assert target.uncertainty.epistemic == 0.0
        assert target.uncertainty.observational == 1.0
    if state == "occluded-defect":
        assert target.independent_observation_count == 0
    else:
        assert target.independent_observation_count > (1 if full_sweep else 0)
    requirement = MissionRequirement(
        requirement_id=ids.new(),
        description="condition over declared capsule inspection domain",
        domain=Domain.TECHNICAL,
        target_belief_ids=(target.belief_id,),
        target_entity_ids=(world.context.critical_component_ids[0],),
        properties=("condition",),
        consequence=0.9,
    )
    context = DecisionContext(
        timestamp=now,
        trace_id=ids.new(),
        mission=MissionState(
            mission_id=world.context.mission_id,
            phase=MissionPhase.INSPECTING,
            timestamp=now,
            objectives_total=1,
            objectives_done=0,
        ),
        mission_spec=world.context.spec,
        requirements=(requirement,),
        snapshot=BeliefSnapshot(
            snapshot_id=ids.new(),
            created_time_ns=now.time_ns,
            messages=(target,),
            domain_availability={Domain.TECHNICAL.value: Availability.AVAILABLE},
            provenance={"revisions": {str(target.belief_id): target.revision}},
        ),
        available_modalities=(spec.modality,),
    )
    assert context.beliefs(Domain.TECHNICAL)[0].technical is not None
    decision = EGDC(ids)
    assert decision.decide(context).record is not None
    if not full_sweep:
        assert decision.last_graph is not None
        assessment = next(
            a for a in decision.last_graph.assessments if a.requirement_id == requirement.requirement_id
        )
        assert assessment.causes[0].value == "OBSERVATIONAL"
