"""Development-only matched spatial structural sensing on the rebuilt Unity player."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from conrad.active.candidates import look_at
from conrad.domains.technical.evidence import structured_evidence
from conrad.domains.technical.spatial_local import LocalThresholds, SpatialModel2T
from conrad.domains.technical.spatial_mission import MODEL_VERSION as SPATIAL_MODEL_VERSION
from conrad.orchestration.association import StructuralAssociator, registry_of
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.schemas.capsule_surface import CapsuleSurfaceGrid, capsule_basis
from conrad.schemas.frames import WORLD, Pose, matrix_to_quat, quat_to_matrix
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import TimeStamp
from conrad.settings import load_settings
from conrad.sim.mission.unity_run import prepare_unity, replay_unity_run
from conrad.sim.mission.unity_world import UnityMissionWorld, UnityWorldOptions
from conrad.sim.unity.player import find_player
from conrad.twins.twin2s.raycast import sensor_world_pose
from tests.integration.test_spatial_mission_truth import _options


@pytest.mark.parametrize("occluded", [False, True], ids=["clear", "view-occluder"])
@pytest.mark.parametrize("noisy", [False, True], ids=["deterministic", "sensor-noise"])
@pytest.mark.parametrize("bounded_survey", [False, True], ids=["exact-survey", "bounded-survey"])
def test_matched_kernel_unity_spatial_support_and_measurements(tmp_path, occluded, noisy, bounded_survey):
    _exercise_kernel_unity_parity(tmp_path, occluded, noisy, bounded_survey, 402 if noisy else 401)


def _exercise_kernel_unity_parity(tmp_path, occluded, noisy, bounded_survey, world_seed):
    player = find_player()
    if player is None:
        pytest.skip("rebuilt Unity player unavailable")
    opts = _options()
    assert opts.spatial_sensor_model is not None
    model = opts.spatial_sensor_model.model_copy(
        update={
            "position_uncertainty_m": 0.0,
            "orientation_uncertainty_rad": 0.0,
            "footprint_uncertainty_m": 0.0,
            "noise_sigma_m": 0.001 if noisy else 0.0,
        }
    )
    changes: dict[str, Any] = {
        "survey_sigma_m": 0.001 if bounded_survey else 0.0,
        "survey_endpoint_bound_m": 0.002 if bounded_survey else None,
        "ecological_enabled": False,
        "spatial_sensor_model": model,
    }
    if occluded:
        changes["occlusion"] = opts.occlusion.model_copy(
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
    opts = opts.model_copy(update=changes)
    world = UnityMissionWorld.build(
        world_seed,
        "SPATIAL-UNITY-PARITY-DEV",
        opts,
        UUID(int=810),
        tmp_path / "unity_run",
        UnityWorldOptions(graphics=True, player_path=str(player)),
    )
    try:
        primitive = world.t2s.world.entities[world.t2s.world.index_of(world.target)].primitive
        a, b = np.asarray(primitive.a), np.asarray(primitive.b)  # type: ignore[attr-defined]
        radius = float(primitive.radius)  # type: ignore[attr-defined]
        _, normal, _ = capsule_basis(a, b)
        stamp = TimeStamp(time_ns=1_000_000_000, clock_domain="sim")
        registry_id = world.context.critical_component_ids[0]
        design = world.context.component(registry_id)
        grid = CapsuleSurfaceGrid(
            float(np.linalg.norm(np.asarray(design.p1_m) - np.asarray(design.p0_m))),
            design.radius_m,
            2,
            4,
        )
        thresholds = LocalThresholds(0.002, 0.006, 0.012, 0.003, 0.01, 0.03)
        kernel_belief = SpatialModel2T(grid, model, thresholds, registry_id, require_depth=True)
        unity_belief = SpatialModel2T(grid, model, thresholds, registry_id, require_depth=True)
        kernel_associator = StructuralAssociator(world.context, MissionRuntimeConfig().association)
        unity_associator = StructuralAssociator(world.context, MissionRuntimeConfig().association)
        kernel_evidence_ids, unity_evidence_ids = IdFactory(18), IdFactory(18)
        kernel = replace(
            world.suite,
            ids=IdFactory(15),
            rng=np.random.default_rng(16),
            spatial_visibility=None,
            record=lambda _kind, _row: None,
        )
        parity_rows = []
        support_counts = []
        unity = replace(
            world.suite,
            ids=IdFactory(15),
            rng=np.random.default_rng(16),
            record=lambda kind, row: parity_rows.append(row) if kind == "spatial_visibility_parity" else None,
        )
        views = (
            (1.0, 0.50, 2.0, 0.0),
            (-1.0, 0.50, 2.0, 0.0),
            (1.0, 0.25, 1.5, 0.5),
            (-1.0, 0.75, 2.5, 0.5),
        )
        mount = world.suite.sensors.structural.mount_pose
        for side, axial_fraction, standoff, height in views:
            target = a + axial_fraction * (b - a) + side * radius * normal
            desired_origin = target + side * standoff * normal + np.array([0.0, 0.0, height])
            sensor_rotation = quat_to_matrix(
                look_at(tuple(desired_origin), tuple(target), WORLD).orientation_wxyz
            )
            body_rotation = sensor_rotation @ quat_to_matrix(mount.orientation_wxyz).T
            rotation = matrix_to_quat(body_rotation)
            body = desired_origin - body_rotation @ np.asarray(mount.position_m)
            pose = Pose(
                frame_id=WORLD,
                position_m=tuple(float(x) for x in body),
                orientation_wxyz=rotation,
                covariance_6x6=(0.0,) * 36,
            )
            rot, origin = sensor_world_pose(pose, world.suite.sensors.structural.mount_pose)
            trace = IdFactory(17).new()
            left = kernel._spatial_structural(stamp, pose, pose, trace, rot, origin)
            right = unity._spatial_structural(stamp, pose, pose, trace, rot, origin)
            support_counts.append(len(left))
            assert [o.structural_support for o in left] == [o.structural_support for o in right]
            assert [o.inline_values for o in left] == [o.inline_values for o in right]
            if bounded_survey and left:
                assert all(
                    o.structural_support is not None
                    and o.structural_support.axial_uncertainty_m is not None
                    and o.structural_support.axial_uncertainty_m > 0
                    for o in left
                )
            for observations, belief, associator, evidence_ids in (
                (left, kernel_belief, kernel_associator, kernel_evidence_ids),
                (right, unity_belief, unity_associator, unity_evidence_ids),
            ):
                for observation in observations:
                    evidence, _ = structured_evidence(observation, evidence_ids, None)
                    evidence, _ = associator.associate(observation, evidence)
                    assert registry_of(evidence) == registry_id
                    accepted = belief.ingest(evidence)
                    if not bounded_survey:
                        assert accepted
            assert [kernel_belief.coverage_fraction(i) for i in range(grid.n_cells)] == pytest.approx(
                [unity_belief.coverage_fraction(i) for i in range(grid.n_cells)], abs=1e-12
            )
            assert [kernel_belief.cell_condition(i) for i in range(grid.n_cells)] == [
                unity_belief.cell_condition(i) for i in range(grid.n_cells)
            ]
            assert kernel_belief.condition() is unity_belief.condition()
        assert parity_rows
        assert sum(support_counts) > 0
        if bounded_survey:
            assert sum(kernel_belief.coverage_fraction(i) for i in range(grid.n_cells)) > 0
        if occluded:
            assert 0 in support_counts
        else:
            assert all(support_counts)
        assert all(r["kernel_visible_unity_hidden"] == 0 for r in parity_rows)
    finally:
        world.close()


def test_short_spatial_unity_mission_replays(tmp_path):
    player = find_player()
    if player is None:
        pytest.skip("rebuilt Unity player unavailable")
    opts = _options()
    assert opts.spatial_sensor_model is not None
    sensor = opts.spatial_sensor_model.model_copy(
        update={
            "position_uncertainty_m": 0.0,
            "orientation_uncertainty_rad": 0.0,
            "footprint_uncertainty_m": 0.0,
        }
    )
    opts = opts.model_copy(
        update={"survey_sigma_m": 0.0, "ecological_enabled": False, "spatial_sensor_model": sensor}
    )
    runtime = MissionRuntimeConfig(
        duration_s=4.0,
        control_period_s=0.1,
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
    session = prepare_unity(
        "I3-UNITY",
        load_settings("configs/sim/mission_test_small.yaml"),
        run_id="SPATIAL-UNITY-SMOKE-DEV",
        runs_root=tmp_path / "runs",
        seed=401,
        uopts=UnityWorldOptions(graphics=True, player_path=str(player)),
        stored_world=opts,
        stored_runtime=runtime,
    )
    try:
        session.run()
        result = session.finish()
    except BaseException:
        session.abort()
        raise
    head = next(
        m
        for m in session.runtime.m2t.export_beliefs()
        if m.world_entity_id == session.world.context.critical_component_ids[0]
    )
    assert head.model_version == SPATIAL_MODEL_VERSION
    assert result["run_id"] == "SPATIAL-UNITY-SMOKE-DEV"
    replay = replay_unity_run(tmp_path / "runs" / "SPATIAL-UNITY-SMOKE-DEV", tmp_path / "replay")
    assert replay["equal"] and replay["same_player_binary"]
