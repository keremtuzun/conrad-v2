"""Visibility clipping checks for pose-derived synthetic support geometry."""

import math
from uuid import UUID

import numpy as np
import pytest

from conrad.schemas.capsule_surface import CapsuleSurfaceGrid, surface_coordinates, surface_point
from conrad.schemas.frames import Pose
from conrad.schemas.structural_sensor import StructuralSensorModelV2
from conrad.schemas.structural_support import ParameterAuthority
from conrad.schemas.world import SensorSpec
from conrad.sim.mission.spatial_support import (
    capsule_visibility_certificate,
    pose_visible_capsule_supports,
    visible_capsule_supports,
)
from conrad.twins.twin2s.config import Twin2SConfig
from conrad.twins.twin2s.sdf import Capsule, Sphere
from conrad.twins.twin2s.world import SpatialEntity, SpatialWorld


def _model() -> StructuralSensorModelV2:
    return StructuralSensorModelV2(
        footprint_width_m=2.0,
        footprint_height_m=2 * math.pi,
        axial_resolution_m=1.0,
        lateral_resolution_m=math.pi / 2,
        minimum_resolvable_corrosion_m=0.1,
        minimum_resolvable_crack_m=0.1,
        minimum_detectable_crack_length_m=0.01,
        minimum_detectable_crack_depth_m=0.001,
        range_min_m=0.3,
        range_max_m=4.0,
        noise_sigma_m=0,
        position_uncertainty_m=0,
        orientation_uncertainty_rad=0,
        footprint_uncertainty_m=0,
        authority=ParameterAuthority.ENGINEERING_ESTIMATE,
    )


def test_near_side_visibility_never_credits_hidden_far_side():
    grid = CapsuleSurfaceGrid(2.0, 1.0, 2, 4)

    def near_side(points, normals):
        del points
        return normals[:, 1] < 0

    supports = visible_capsule_supports(
        grid,
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        _model(),
        near_side,
        frame_id="CAPSULE_DESIGN",
    )
    assert len(supports) == 4
    assert all(s.angle_end_rad <= math.pi / 2 or s.angle_start_rad >= 3 * math.pi / 2 for s in supports)
    assert all(s.occlusion_clipped for s in supports)


def test_partial_occluder_removes_affected_axial_cells():
    grid = CapsuleSurfaceGrid(2.0, 1.0, 2, 4)

    def partial(points, normals):
        return (points[:, 0] < 1.0) & (normals[:, 1] < 0)

    supports = visible_capsule_supports(
        grid,
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        _model(),
        partial,
        frame_id="CAPSULE_DESIGN",
    )
    assert len(supports) == 2
    assert all(s.axial_end_m <= 1.0 for s in supports)


def test_geometry_oracle_shape_and_axis_mismatch_fail_closed():
    grid = CapsuleSurfaceGrid(2.0, 1.0, 2, 4)
    with pytest.raises(ValueError, match="axis"):
        visible_capsule_supports(
            grid,
            np.zeros(3),
            np.array([3.0, 0.0, 0.0]),
            _model(),
            lambda p, n: np.ones(len(p), dtype=bool),
            frame_id="CAPSULE_DESIGN",
        )


def test_pose_ray_limits_nominal_footprint_and_rejects_miss():
    grid = CapsuleSurfaceGrid(2.0, 1.0, 2, 4)
    a, b = np.zeros(3), np.array([2.0, 0.0, 0.0])
    origin = np.array([0.5, -3.0, 0.0])

    def visible(points, normals):
        del points
        return normals[:, 1] < 0

    supports = pose_visible_capsule_supports(
        grid,
        a,
        b,
        origin,
        np.array([0.0, 1.0, 0.0]),
        _model(),
        visible,
        frame_id="CAPSULE_DESIGN",
    )
    assert supports
    assert all(s.axial_end_m <= 1.0 for s in supports)
    assert (
        pose_visible_capsule_supports(
            grid,
            a,
            b,
            origin,
            np.array([1.0, 0.0, 0.0]),
            _model(),
            visible,
            frame_id="CAPSULE_DESIGN",
        )
        == ()
    )
    with pytest.raises(ValueError, match="wrong shape"):
        visible_capsule_supports(
            grid,
            np.zeros(3),
            np.array([2.0, 0.0, 0.0]),
            _model(),
            lambda p, n: np.ones(1, dtype=bool),
            frame_id="CAPSULE_DESIGN",
        )


def test_off_axis_pipe_requires_continuous_field_of_view_certificate():
    grid = CapsuleSurfaceGrid(2.0, 1.0, 2, 4)
    a, b = np.zeros(3), np.array([2.0, 0.0, 0.0])
    origin = np.array([1.0, -3.0, 1.2])
    forward = np.array([0.0, 1.0, 0.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    sensor = SensorSpec(
        sensor_id=UUID(int=6),
        modality="STRUCTURED",
        frame_id="SENSOR",
        mount_pose=Pose(frame_id="ROBOT", position_m=(0, 0, 0), orientation_wxyz=(1, 0, 0, 0)),
        rate_hz=1.0,
        parameters={"hfov_deg": 100.0, "vfov_deg": 80.0, "min_range_m": 0.3, "max_range_m": 4.0},
    )
    world = SpatialWorld(
        [SpatialEntity(UUID(int=1), "pipeline_segment", Capsule((0, 0, 0), (2, 0, 0), 1.0))],
        (-5, -5, -5),
        (5, 5, 5),
    )
    visible = lambda points, normals: np.ones(len(points), dtype=bool)  # noqa: E731
    model = _model().model_copy(update={"axial_resolution_m": 0.2, "lateral_resolution_m": 0.2})
    kwargs = (grid, a, b, origin, forward, model, visible)
    assert not pose_visible_capsule_supports(*kwargs, frame_id="CAPSULE_DESIGN")
    certificate = capsule_visibility_certificate(
        world, 0, a, b, 1.0, origin, rotation, sensor, Twin2SConfig()
    )
    supports = pose_visible_capsule_supports(*kwargs, frame_id="CAPSULE_DESIGN", certify=certificate)
    assert supports
    assert all(
        certificate(s.axial_start_m, s.axial_end_m, s.angle_start_rad, s.angle_end_rad) for s in supports
    )


def test_continuous_certificate_refuses_narrow_unproven_occlusion():
    axis_a, axis_b = np.zeros(3), np.array([2.0, 0.0, 0.0])
    origin = np.array([1.0, -3.0, 0.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    sensor = SensorSpec(
        sensor_id=UUID(int=4),
        modality="STRUCTURED",
        frame_id="SENSOR",
        mount_pose=Pose(frame_id="ROBOT", position_m=(0, 0, 0), orientation_wxyz=(1, 0, 0, 0)),
        rate_hz=1.0,
        parameters={"hfov_deg": 100.0, "vfov_deg": 80.0, "min_range_m": 0.3, "max_range_m": 4.0},
    )
    target = SpatialEntity(UUID(int=1), "pipeline_segment", Capsule((0, 0, 0), (2, 0, 0), 1.0))
    clear = SpatialWorld([target], (-5, -5, -5), (5, 5, 5))
    occluded = SpatialWorld(
        [target, SpatialEntity(UUID(int=2), "rock", Sphere((1.07, -2.0, 0.0), 0.03))],
        (-5, -5, -5),
        (5, 5, 5),
    )
    kwargs = (0, axis_a, axis_b, 1.0, origin, rotation, sensor, Twin2SConfig())
    rect = (0.9, 1.1, 0.0, 0.1)
    assert capsule_visibility_certificate(clear, *kwargs)(*rect)
    assert not capsule_visibility_certificate(occluded, *kwargs)(*rect)


def test_subdivided_certificate_can_prove_visible_resolution_cell():
    radius = 0.28
    a, b = np.zeros(3), np.array([2.0, 0.0, 0.0])
    angle = 3.5
    normal = np.array([0.0, -math.cos(angle), -math.sin(angle)])
    origin = np.array([1.0, 0.0, 0.0]) + (radius + 0.5) * normal
    forward = -normal
    side = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    side /= np.linalg.norm(side)
    rotation = np.column_stack((forward, side, np.cross(forward, side)))
    sensor = SensorSpec(
        sensor_id=UUID(int=5),
        modality="STRUCTURED",
        frame_id="SENSOR",
        mount_pose=Pose(frame_id="ROBOT", position_m=(0, 0, 0), orientation_wxyz=(1, 0, 0, 0)),
        rate_hz=1.0,
        parameters={"hfov_deg": 100.0, "vfov_deg": 80.0, "min_range_m": 0.3, "max_range_m": 4.0},
    )
    world = SpatialWorld(
        [SpatialEntity(UUID(int=1), "pipeline_segment", Capsule((0, 0, 0), (2, 0, 0), radius))],
        (-5, -5, -5),
        (5, 5, 5),
    )
    certificate = capsule_visibility_certificate(
        world, 0, a, b, radius, origin, rotation, sensor, Twin2SConfig()
    )
    assert certificate(0.9, 1.1, 3.14, 3.86)


@pytest.mark.parametrize("angle", [-4 * math.pi, -0.01, 0.0, 0.01, math.pi, 2 * math.pi + 0.3])
def test_world_surface_coordinates_round_trip_and_wrap(angle):
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 4.0, 5.0])
    point = surface_point(a, b, 0.4, 1.2, angle)
    x, wrapped, radial = surface_coordinates(a, b, point)
    assert x == pytest.approx(1.2)
    assert wrapped == pytest.approx(angle % (2 * math.pi), abs=1e-12)
    assert radial == pytest.approx(0.4)
