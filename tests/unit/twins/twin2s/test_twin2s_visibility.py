"""Ray casting speed/correctness and visibility-oracle sanity."""

from __future__ import annotations

import time
from uuid import UUID

import numpy as np

from conrad.schemas.frames import ROBOT, WORLD, Pose
from conrad.schemas.world import SensorSpec
from conrad.twins.twin2s.config import RaycastConfig, Twin2SConfig
from conrad.twins.twin2s.raycast import pinhole_directions, sphere_trace
from conrad.twins.twin2s.sdf import Box, Sphere
from conrad.twins.twin2s.visibility import VisibilityOracle, VisibilityReason
from conrad.twins.twin2s.world import SpatialEntity, SpatialWorld


def _world():
    return SpatialWorld(
        [
            SpatialEntity(UUID(int=1), "rock", Sphere((2.0, 0.0, 0.0), 0.5)),  # occluder
            SpatialEntity(UUID(int=2), "pipeline_segment", Box((4.5, 0.0, 0.0), (0.5, 2.0, 2.0))),  # target
        ],
        (-20, -20, -20),
        (20, 20, 20),
    )


def _sensor():
    return SensorSpec(
        sensor_id=UUID(int=9),
        modality="DEPTH_RANGE",
        frame_id="S",
        rate_hz=5.0,
        mount_pose=Pose(frame_id=ROBOT, position_m=(0.0, 0.0, 0.0)),
        parameters={"width_px": 64, "height_px": 48, "hfov_deg": 70.0, "max_range_m": 12.0},
    )


ORIGIN = Pose(frame_id=WORLD, position_m=(0.0, 0.0, 0.0))


def test_sphere_trace_range_and_entity():
    w = _world()
    dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0.3, 0]])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    h = sphere_trace(w, np.zeros(3), dirs, 12.0, RaycastConfig())
    assert abs(h.range_m[0] - 1.5) < 1e-6 and h.entity_index[0] == 0
    assert not h.hit[1] and h.entity_index[1] == -1
    assert h.entity_index[2] == 1 and abs(h.points_m[2, 0] - 4.0) < 1e-6


def test_64x48_depth_raycast_is_fast():
    w = _world()
    dirs = pinhole_directions(64, 48, 70.0)
    t0 = time.perf_counter()
    h = sphere_trace(w, np.zeros(3), dirs, 12.0, RaycastConfig())
    assert time.perf_counter() - t0 < 0.5
    assert h.hit.any()


def test_occluded_point_is_not_visible_and_reasons():
    oracle = VisibilityOracle(_world(), Twin2SConfig())
    pts = np.array(
        [
            [4.0, 0.0, 0.0],  # target face directly behind the occluding rock
            [4.0, 1.2, 0.0],  # target face, clear line of sight
            [-8.5, 0.0, 0.0],  # behind the sensor
            [4.0, 0.0, 30.0],  # outside the vertical FOV / range
        ]
    )
    normals = np.array([[-1.0, 0, 0]] * 4)
    r = oracle.visibility(_sensor(), ORIGIN, pts, normals)
    assert r.reason[0] == VisibilityReason.OCCLUDED and not r.visible[0] and r.score[0] == 0.0
    assert r.visible[1] and 0.0 < r.score[1] <= 1.0
    assert r.reason[2] == VisibilityReason.OUT_OF_FOV
    assert r.reason[3] in (VisibilityReason.OUT_OF_FOV, VisibilityReason.OUT_OF_RANGE)
    # factor decomposition is stored, not only a Boolean
    assert np.isfinite(r.range_factor[1]) and np.isfinite(r.environment_factor[1])


def test_turbidity_lowers_observability():
    oracle = VisibilityOracle(_world(), Twin2SConfig())
    p = np.array([[4.0, 1.2, 0.0]])
    clear = oracle.visibility(_sensor(), ORIGIN, p).score[0]
    murky = oracle.visibility(_sensor(), ORIGIN, p, turbidity=1.0).score[0]
    assert murky < clear
