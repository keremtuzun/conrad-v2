"""Opt-in mission Twin2T local truth and evolution, without a sensor fixture."""

import math
from uuid import UUID

import numpy as np
import pytest

from conrad.schemas.frames import WORLD, Pose, quat_from_euler, quat_to_matrix
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import TimeStamp
from conrad.sim.mission.options import MissionWorldOptions
from conrad.sim.mission.spatial_support import capsule_basis
from conrad.sim.mission.world import MissionWorld


def _options() -> MissionWorldOptions:
    return MissionWorldOptions.model_validate(
        {
            "twin2t_truth_model": "spatial_v1",
            "spatial_truth": {
                "axial_cells": 2,
                "sectors": 4,
                "base": {},
                "cell_rates": [
                    {"corrosion_m_per_s": 0.001},
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                ],
                "patches": [
                    {
                        "axial_start_fraction": 0.2,
                        "axial_end_fraction": 0.4,
                        "angle_start_rad": 0.2,
                        "angle_end_rad": 0.5,
                        "state": {"crack_length_m": 0.02, "crack_depth_m": 0.003},
                    }
                ],
            },
            "spatial_sensor_model": {
                "footprint_width_m": 1.0,
                "footprint_height_m": 1.0,
                "axial_resolution_m": 0.2,
                "lateral_resolution_m": 0.2,
                "minimum_resolvable_corrosion_m": 0.05,
                "minimum_resolvable_crack_m": 0.05,
                "minimum_detectable_crack_length_m": 0.01,
                "minimum_detectable_crack_depth_m": 0.001,
                "range_min_m": 0.3,
                "range_max_m": 4.0,
                "noise_sigma_m": 0.0,
                "authority": "ENGINEERING_ESTIMATE",
            },
        }
    )


def test_mission_twin2t_loads_local_field_and_advances_cells_independently(tmp_path):
    world = MissionWorld.build(401, "SPATIAL-DEV", _options(), UUID(int=55), tmp_path / "objects")
    initial = world.t2t.spatial_fields[world.target]
    assert len(initial.base) == 8
    assert initial.worst_local().crack_depth_m == 0.003
    assert initial.base[0].corrosion_depth_m == 0.0
    assert "spatial_fields" not in world.context.model_dump_json()
    world.advance(1.0)
    evolved = world.t2t.spatial_fields[world.target]
    assert evolved.base[0].corrosion_depth_m == pytest.approx(0.001)
    assert evolved.base[1].corrosion_depth_m == 0.0
    assert evolved.patches[0].state.crack_depth_m == 0.003


def test_legacy_world_has_no_local_truth_and_spatial_requires_explicit_config(tmp_path):
    legacy = MissionWorld.build(401, "LEGACY-DEV", MissionWorldOptions(), UUID(int=56), tmp_path / "legacy")
    assert legacy.t2t.spatial_fields == {}
    with pytest.raises(ValueError, match="spatial_v1 requires"):
        MissionWorldOptions(twin2t_truth_model="spatial_v1")


def test_mission_structural_sensor_derives_support_from_pose_and_scene(tmp_path):
    world = MissionWorld.build(401, "SPATIAL-SENSOR-DEV", _options(), UUID(int=57), tmp_path / "sensor")
    suite = world.hardware.suite
    assert suite is not None
    primitive = world.t2s.world.entities[world.t2s.world.index_of(world.target)].primitive
    a, b = np.asarray(primitive.a), np.asarray(primitive.b)  # type: ignore[attr-defined]
    radius = float(primitive.radius)  # type: ignore[attr-defined]
    _, normal, _ = capsule_basis(a, b)
    desired_origin = (a + b) / 2 + (radius + 2.0) * normal
    forward = -normal
    yaw = math.atan2(float(forward[1]), float(forward[0])) - math.radians(world.opts.structural.mount_yaw_deg)
    rotation = quat_from_euler(0.0, 0.0, yaw)
    body_position = desired_origin - quat_to_matrix(rotation) @ np.asarray(
        world.opts.structural.mount_position_m
    )
    pose = Pose(frame_id=WORLD, position_m=tuple(float(x) for x in body_position), orientation_wxyz=rotation)
    observations = suite._structural(
        1.0, TimeStamp(time_ns=1_000_000_000, clock_domain="sim"), pose, pose, IdFactory(99).new()
    )
    spatial = [o for o in observations if o.structural_support is not None]
    assert spatial
    assert all(o.sensor_context["structural_sensor_version"] == "structural-sensor-v2" for o in spatial)
    assert all("world_id" not in o.canonical_json() for o in spatial)


def test_flown_off_axis_view_credits_only_certified_spatial_support(tmp_path):
    opts = _options().model_copy(
        update={"family": "pipeline_with_supports", "survey_sigma_m": 0.0, "ecological_enabled": False}
    )
    world = MissionWorld.build(2026201, "SPATIAL-OFF-AXIS-DEV", opts, UUID(int=987), tmp_path)
    suite = world.hardware.suite
    assert suite is not None
    # The vehicle's inspection controller holds yaw, not pitch. Its boresight
    # passes above the pipe while the upper surface remains inside the FOV.
    pose = Pose(
        frame_id=WORLD,
        position_m=(0.13, 2.52, 1.51),
        orientation_wxyz=quat_from_euler(0.0, 0.0, math.pi),
        covariance_6x6=(0.0,) * 36,
    )
    observations = suite._structural(
        65.0, TimeStamp(time_ns=65_000_000_000, clock_domain="sim"), pose, pose, IdFactory(99).new()
    )
    assert observations
    assert all(o.structural_support is not None for o in observations)
