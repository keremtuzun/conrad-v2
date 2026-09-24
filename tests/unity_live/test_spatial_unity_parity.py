"""Development-only matched spatial structural sensing on the rebuilt Unity player."""

from __future__ import annotations

import math
from dataclasses import replace
from uuid import UUID

import numpy as np
import pytest

from conrad.domains.technical.spatial_mission import MODEL_VERSION as SPATIAL_MODEL_VERSION
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.schemas.capsule_surface import capsule_basis
from conrad.schemas.frames import WORLD, Pose, quat_from_euler, quat_to_matrix
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import TimeStamp
from conrad.settings import load_settings
from conrad.sim.mission.unity_run import prepare_unity, replay_unity_run
from conrad.sim.mission.unity_world import UnityMissionWorld, UnityWorldOptions
from conrad.sim.unity.player import find_player
from conrad.twins.twin2s.raycast import sensor_world_pose
from tests.integration.test_spatial_mission_truth import _options


def test_matched_kernel_unity_spatial_support_and_measurements(tmp_path):
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
        }
    )
    opts = opts.model_copy(
        update={"survey_sigma_m": 0.0, "ecological_enabled": False, "spatial_sensor_model": model}
    )
    world = UnityMissionWorld.build(
        401,
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
        kernel = replace(
            world.suite,
            ids=IdFactory(15),
            rng=np.random.default_rng(16),
            spatial_visibility=None,
            record=lambda _kind, _row: None,
        )
        parity_rows = []
        unity = replace(
            world.suite,
            ids=IdFactory(15),
            rng=np.random.default_rng(16),
            record=lambda kind, row: parity_rows.append(row) if kind == "spatial_visibility_parity" else None,
        )
        for side in (1.0, -1.0):
            desired_origin = (a + b) / 2 + side * (radius + 2.0) * normal
            forward = -side * normal
            yaw = math.atan2(float(forward[1]), float(forward[0])) - math.radians(
                opts.structural.mount_yaw_deg
            )
            rotation = quat_from_euler(0.0, 0.0, yaw)
            body = desired_origin - quat_to_matrix(rotation) @ np.asarray(opts.structural.mount_position_m)
            pose = Pose(frame_id=WORLD, position_m=tuple(float(x) for x in body), orientation_wxyz=rotation)
            rot, origin = sensor_world_pose(pose, world.suite.sensors.structural.mount_pose)
            trace = IdFactory(17).new()
            left = kernel._spatial_structural(stamp, pose, pose, trace, rot, origin)
            right = unity._spatial_structural(stamp, pose, pose, trace, rot, origin)
            assert left and right
            assert [o.structural_support for o in left] == [o.structural_support for o in right]
            assert [o.inline_values for o in left] == [o.inline_values for o in right]
        assert parity_rows and all(r["kernel_visible_unity_hidden"] == 0 for r in parity_rows)
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
