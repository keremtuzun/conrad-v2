"""Visibility clipping checks for pose-derived synthetic support geometry."""

import math

import numpy as np
import pytest

from conrad.schemas.capsule_surface import CapsuleSurfaceGrid
from conrad.schemas.structural_sensor import StructuralSensorModelV2
from conrad.schemas.structural_support import ParameterAuthority
from conrad.sim.mission.spatial_support import visible_capsule_supports


def _model() -> StructuralSensorModelV2:
    return StructuralSensorModelV2(
        footprint_width_m=2.0,
        footprint_height_m=2 * math.pi,
        axial_resolution_m=1.0,
        lateral_resolution_m=math.pi / 2,
        minimum_resolvable_corrosion_m=0.1,
        minimum_resolvable_crack_m=0.1,
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
    with pytest.raises(ValueError, match="wrong shape"):
        visible_capsule_supports(
            grid,
            np.zeros(3),
            np.array([2.0, 0.0, 0.0]),
            _model(),
            lambda p, n: np.ones(1, dtype=bool),
            frame_id="CAPSULE_DESIGN",
        )
