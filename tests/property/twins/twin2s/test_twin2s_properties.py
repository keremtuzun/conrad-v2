"""Property tests: every primitive SDF is 1-Lipschitz (sphere-tracing safe); exact ones match surface distance."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from conrad.schemas.ids import IdFactory
from conrad.twins.twin2s.families import generate_scenario
from conrad.twins.twin2s.sdf import Box, Capsule, Cylinder, Ellipsoid, PolyCapsule, Sphere, SphereBlob, Torus
from conrad.twins.twin2s.terrain import HeightfieldTerrain
from conrad.twins.twin2s.world import SpatialWorld

coord = st.floats(-2.0, 2.0, allow_nan=False)
pos = st.floats(0.1, 1.5, allow_nan=False)
vec = st.tuples(coord, coord, coord)

PRIMS = st.one_of(
    st.builds(Sphere, vec, pos),
    st.builds(Ellipsoid, vec, st.tuples(pos, pos, pos)),
    st.builds(Capsule, vec, vec, pos),
    st.builds(lambda a, b, c, r: PolyCapsule((a, b, c), r), vec, vec, vec, pos),
    st.builds(lambda a, r: Cylinder(a, (a[0] + 1.0, a[1], a[2] + 0.5), r), vec, pos),
    st.builds(Box, vec, st.tuples(pos, pos, pos)),
    st.builds(lambda c, R, r: Torus(c, (0.3, 0.2, 1.0), R + r, r), vec, pos, st.floats(0.02, 0.3)),
    st.builds(lambda a, b, r: SphereBlob((a, b), (r, r / 2)), vec, vec, pos),
    st.builds(
        lambda amp, k, h: HeightfieldTerrain(
            0.0, (0.03, -0.02), ((amp, k, 0.5 * k, 0.1),), ((h, 0.0, 0.0, 0.7, 1.0),), (-5, -5, -5), (5, 5, 5)
        ),
        st.floats(0.0, 0.3),
        st.floats(0.1, 2.0),
        st.floats(-0.5, 0.5),
    ),
)


@settings(max_examples=60, deadline=None)
@given(PRIMS, st.integers(0, 2**31 - 1))
def test_sdf_is_one_lipschitz(prim, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-4, 4, (200, 3))
    q = p + rng.normal(0, 0.3, (200, 3))
    lhs = np.abs(prim.sdf(p) - prim.sdf(q))
    assert np.all(lhs <= np.linalg.norm(p - q, axis=1) + 1e-9)


@settings(max_examples=40, deadline=None)
@given(st.builds(Sphere, vec, pos), st.integers(0, 2**31 - 1))
def test_exact_sdf_zero_set_is_surface(sphere, seed):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(100, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    on = np.asarray(sphere.center) + sphere.radius * d
    assert np.allclose(sphere.sdf(on), 0.0, atol=1e-12)


@settings(max_examples=6, deadline=None)
@given(st.integers(0, 10_000), st.sampled_from(["straight_pipeline", "bent_pipeline", "cluttered_field"]))
def test_generated_world_is_deterministic_and_loadable(seed, family):
    a = generate_scenario(seed, IdFactory(seed), family)
    b = generate_scenario(seed, IdFactory(seed), family)
    assert a.content_digest() == b.content_digest()
    w = SpatialWorld.from_spatial_state(a.spatial_state)
    spawn = np.asarray(a.robots[0].initial_pose.position_m)[None, :]
    assert w.sdf(spawn)[0] > 0.0  # the robot never spawns inside geometry
