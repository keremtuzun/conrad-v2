"""SDF correctness, primitive round-trip and octree export."""

from __future__ import annotations

import math
from uuid import UUID

import numpy as np
import pytest

from conrad.twins.twin2s.config import OctreeConfig
from conrad.twins.twin2s.octree import export_octree
from conrad.twins.twin2s.sdf import Box, Capsule, Cylinder, Ellipsoid, PolyCapsule, Sphere, SphereBlob, Torus
from conrad.twins.twin2s.terrain import HeightfieldTerrain, primitive_from_dict
from conrad.twins.twin2s.world import Motion, SpatialEntity, SpatialWorld

P = np.array


def test_sphere_capsule_cylinder_box_torus_exact_values():
    assert Sphere((0, 0, 0), 1.0).sdf(P([[2.0, 0, 0], [0, 0, 0]])) == pytest.approx([1.0, -1.0])
    cap = Capsule((0, 0, 0), (2, 0, 0), 0.5)
    assert cap.sdf(P([[1.0, 1.0, 0], [3.0, 0, 0], [1.0, 0, 0]])) == pytest.approx([0.5, 0.5, -0.5])
    cyl = Cylinder((0, 0, 0), (0, 0, 2), 1.0)
    assert cyl.sdf(P([[2.0, 0, 1], [0, 0, 3], [2.0, 0, 3], [0, 0, 1]])) == pytest.approx(
        [1.0, 1.0, math.sqrt(2), -1.0]
    )
    box = Box((0, 0, 0), (1, 2, 3))
    assert box.sdf(P([[2.0, 0, 0], [0, 0, 0], [2, 3, 0]])) == pytest.approx([1.0, -1.0, math.sqrt(2)])
    tor = Torus((0, 0, 0), (0, 0, 1), 2.0, 0.5)
    assert tor.sdf(P([[2.0, 0, 0], [3.0, 0, 0], [0, 0, 0]])) == pytest.approx([-0.5, 0.5, 1.5])


def test_polycapsule_bend_and_blob():
    bend = PolyCapsule(((0, 0, 0), (1, 0, 0), (1, 1, 0)), 0.1)
    assert bend.sdf(P([[1.0, 0.5, 0], [0.5, 0.5, 0]])) == pytest.approx([-0.1, 0.4])
    blob = SphereBlob(((0, 0, 0), (3, 0, 0)), (1.0, 0.5))
    assert blob.sdf(P([[3.0, 0, 0], [1.5, 0, 0]])) == pytest.approx([-0.5, 0.5])


def test_conservative_sdfs_are_sign_exact_and_lower_bounds():
    rng = np.random.default_rng(0)
    ell = Ellipsoid((0, 0, 0), (2.0, 1.0, 0.5))
    pts = rng.uniform(-3, 3, (2000, 3))
    inside = (pts[:, 0] / 2) ** 2 + pts[:, 1] ** 2 + (pts[:, 2] / 0.5) ** 2 < 1
    assert np.array_equal(ell.sdf(pts) < 0, inside)
    ter = HeightfieldTerrain(
        0.0, (0.05, 0.0), ((0.1, 0.5, 0.3, 0.2),), ((0.4, 0, 0, 0.5, 1.0),), (-5, -5, -2), (5, 5, 2)
    )
    d = ter.sdf(pts)
    assert np.array_equal(d < 0, pts[:, 2] < ter.height(pts[:, :2]))
    # 1-Lipschitz lower bound: |sdf| never exceeds the vertical distance to the surface
    assert np.all(np.abs(d) <= np.abs(pts[:, 2] - ter.height(pts[:, :2])) + 1e-12)


@pytest.mark.parametrize(
    "prim",
    [
        Sphere((1, 2, 3), 0.4),
        Ellipsoid((0, 0, 0), (1, 2, 3), (0.9, 0.1, 0.0, 0.4)),
        Capsule((0, 0, 0), (1, 1, 1), 0.2),
        PolyCapsule(((0, 0, 0), (1, 0, 0), (1, 2, 0)), 0.3),
        Cylinder((0, 0, 0), (0, 1, 0), 0.5),
        Box((1, 1, 1), (0.5, 0.2, 0.3)),
        Torus((0, 0, 0), (1, 0, 0), 0.3, 0.02),
        SphereBlob(((0, 0, 0),), (0.3,)),
    ],
)
def test_primitive_round_trip(prim):
    back = primitive_from_dict(prim.to_dict())
    pts = np.random.default_rng(1).uniform(-2, 2, (50, 3))
    np.testing.assert_allclose(back.sdf(pts), prim.sdf(pts), atol=1e-12)


def test_world_union_nearest_entity_and_motion():
    a, b = UUID(int=1), UUID(int=2)
    w = SpatialWorld(
        [
            SpatialEntity(a, "rock", Sphere((0, 0, 0), 1.0)),
            SpatialEntity(b, "dynamic_object", Sphere((5, 0, 0), 1.0), motion=Motion((1.0, 0, 0))),
        ],
        (-10, -10, -10),
        (10, 10, 10),
    )
    d, idx = w.sdf_with_index(P([[0.0, 0, 0], [5.0, 0, 0], [2.5, 0, 0]]))
    assert list(idx) == [0, 1, 0] and d[0] == pytest.approx(-1.0)
    w.time_s = 2.0
    assert w.occupied(P([[7.0, 0, 0]]))[0] and not w.occupied(P([[5.0, 0, 0]]))[0]


def test_octree_export_levels_and_occupancy():
    w = SpatialWorld([SpatialEntity(UUID(int=1), "rock", Sphere((0, 0, 0), 0.6))], (-1, -1, -1), (1, 1, 1))
    o = export_octree(w, OctreeConfig())
    assert set(np.unique(o.size_m)) <= {0.25, 0.125, 0.0625}
    # surface cells are refined to the finest level; interior cells may stay coarse
    assert np.all(o.size_m[o.surface] == 0.0625)
    np.testing.assert_array_equal(o.occupied, np.linalg.norm(o.centers_m, axis=1) < 0.6)
    vol = float((o.size_m[o.occupied] ** 3).sum())
    assert vol == pytest.approx(4 / 3 * math.pi * 0.6**3, rel=0.1)
    with pytest.raises(ValueError):
        export_octree(w, OctreeConfig(refinement_voxels_m=(0.1,)))
