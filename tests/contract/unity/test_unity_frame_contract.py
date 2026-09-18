"""ch28 frame contract tests for the documented Unity adapter.

* T_A_from_B @ T_B_from_A approximately equals identity.
* Unit-axis probes match the documented Unity adapter.
* Round-trip pose preserves position and orientation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conrad.adapters.unity import DEFAULT_MAPPER, UNITY_WORLD, UnityFrameMapper
from conrad.schemas.frames import (
    WORLD,
    FrameConvention,
    FrameError,
    Handedness,
    Pose,
    Transform,
    quat_from_euler,
    quat_to_matrix,
)

M = DEFAULT_MAPPER


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=np.float64)


def _same_rotation(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return abs(abs(sum(x * y for x, y in zip(a, b, strict=True))) - 1.0) < 1e-9


def test_axis_map_is_orthonormal_and_flips_handedness() -> None:
    s = M.unity_from_conrad
    assert np.allclose(s @ s.T, np.eye(3))
    assert M.determinant == pytest.approx(-1.0)


def test_t_a_from_b_times_t_b_from_a_is_identity() -> None:
    assert np.allclose(M.unity_from_conrad @ M.conrad_from_unity, np.eye(3), atol=1e-12)
    assert np.allclose(M.conrad_from_unity @ M.unity_from_conrad, np.eye(3), atol=1e-12)
    t = Transform(
        parent_frame=WORLD,
        child_frame="ROBOT",
        translation_m=(1.0, -2.0, 3.0),
        rotation_wxyz=quat_from_euler(0.2, -0.4, 1.1),
    )
    ident = t.compose(t.inverse().model_copy(update={"child_frame": "WORLD2"}))
    assert np.allclose(ident.translation, 0.0, atol=1e-12)
    assert np.allclose(ident.rotation, np.eye(3), atol=1e-12)
    # the same holds after mapping the transform into Unity
    tu = M.transform_to_unity(t, UNITY_WORLD, "UNITY_ROBOT")
    ident_u = tu.compose(tu.inverse().model_copy(update={"child_frame": "UNITY_WORLD2"}))
    assert np.allclose(ident_u.translation, 0.0, atol=1e-12)
    assert np.allclose(ident_u.rotation, np.eye(3), atol=1e-12)


@pytest.mark.parametrize(
    ("conrad_axis", "unity_axis", "meaning"),
    [
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "forward: Conrad +X is Unity +Z"),
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), "left: Conrad +Y is Unity -X"),
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), "up: Conrad +Z is Unity +Y"),
    ],
)
def test_unit_axis_probes(
    conrad_axis: tuple[float, float, float], unity_axis: tuple[float, float, float], meaning: str
) -> None:
    assert M.point_to_unity(conrad_axis) == pytest.approx(unity_axis), meaning
    assert M.point_to_conrad(unity_axis) == pytest.approx(conrad_axis), meaning


def test_quaternion_probe_yaw_left_is_negative_about_unity_up() -> None:
    yaw = 0.3
    q_u = M.quat_to_unity(quat_from_euler(0.0, 0.0, yaw))
    # documented closed form: (w, x, y, z)_c -> (w, y, -z, -x)_u
    assert q_u == pytest.approx((math.cos(yaw / 2), 0.0, -math.sin(yaw / 2), 0.0))
    # Behavioural probe with Unity's own rule: a positive angle about +Y turns +Z (forward) towards +X (right).
    forward_after = quat_to_matrix(q_u) @ np.array([0.0, 0.0, 1.0])
    assert forward_after[0] < 0  # turned LEFT (towards -X), which is Conrad +yaw


def test_quaternion_probes_roll_and_pitch() -> None:
    a = 0.2
    # Conrad roll about +X(forward) -> Unity rotation about +Z(forward) with opposite sign
    assert M.quat_to_unity(quat_from_euler(a, 0, 0)) == pytest.approx(
        (math.cos(a / 2), 0, 0, -math.sin(a / 2))
    )
    # Conrad pitch about +Y(left) -> Unity rotation about +X(right) with the same sign (left = -right, det = -1)
    assert M.quat_to_unity(quat_from_euler(0, a, 0)) == pytest.approx(
        (math.cos(a / 2), math.sin(a / 2), 0, 0)
    )


def test_angular_velocity_is_axial() -> None:
    assert M.angular_velocity_to_unity((0.0, 0.0, 1.0)) == pytest.approx((0.0, -1.0, 0.0))
    assert M.angular_velocity_to_unity((1.0, 0.0, 0.0)) == pytest.approx((0.0, 0.0, -1.0))
    assert M.angular_velocity_to_unity((0.0, 1.0, 0.0)) == pytest.approx((1.0, 0.0, 0.0))


@settings(max_examples=50, deadline=None)
@given(
    st.tuples(*[st.floats(-3, 3)] * 3),
    st.tuples(*[st.floats(-5, 5)] * 3),
    st.tuples(*[st.floats(-50, 50)] * 3),
)
def test_rotation_vector_and_kinematics_consistency(
    euler: tuple[float, float, float], omega: tuple[float, float, float], point: tuple[float, float, float]
) -> None:
    q_c = quat_from_euler(*euler)
    r_c = quat_to_matrix(q_c)
    r_u = quat_to_matrix(M.quat_to_unity(q_c))
    s = M.unity_from_conrad
    # rotating then mapping == mapping then rotating
    p = np.asarray(point)
    assert np.allclose(s @ (r_c @ p), r_u @ (s @ p), atol=1e-9)
    assert np.linalg.det(r_u) == pytest.approx(1.0)
    # Rdot = [w]x R must hold in both conventions with the axial mapping
    w_c = np.asarray(omega)
    w_u = np.asarray(M.angular_velocity_to_unity(omega))
    assert np.allclose(s @ (_skew(w_c) @ r_c) @ s.T, _skew(w_u) @ r_u, atol=1e-9)
    # torque is axial: tau = r x F maps consistently
    f = np.array([0.3, -1.2, 2.0])
    tau_u = np.cross(s @ p, s @ f)
    assert np.allclose(tau_u, np.asarray(M.axial_to_unity(tuple(np.cross(p, f)))), atol=1e-9)


@settings(max_examples=50, deadline=None)
@given(st.tuples(*[st.floats(-3, 3)] * 3), st.tuples(*[st.floats(-100, 100)] * 3))
def test_round_trip_pose_preserves_position_and_orientation(
    euler: tuple[float, float, float], position: tuple[float, float, float]
) -> None:
    pose = Pose(frame_id=WORLD, position_m=position, orientation_wxyz=quat_from_euler(*euler))
    back = M.pose_to_conrad(M.pose_to_unity(pose), WORLD)
    assert back.frame_id == WORLD
    assert back.position_m == pytest.approx(pose.position_m, abs=1e-9)
    assert _same_rotation(back.orientation_wxyz, pose.orientation_wxyz)
    assert M.pose_to_unity(pose).frame_id == UNITY_WORLD
    assert M.angular_velocity_to_conrad(M.angular_velocity_to_unity(position)) == pytest.approx(position)


def test_other_declared_conventions_are_supported_not_assumed() -> None:
    # A Unity-like declared convention maps to itself.
    unity_like = UnityFrameMapper(
        FrameConvention(handedness=Handedness.LEFT, up_axis="+Y", forward_axis="+Z")
    )
    assert np.allclose(unity_like.unity_from_conrad, np.eye(3))
    assert unity_like.determinant == pytest.approx(1.0)
    # NED-style right-handed frame: +X forward, +Z down.
    ned = UnityFrameMapper(FrameConvention(up_axis="-Z", forward_axis="+X"))
    assert ned.point_to_unity((0.0, 0.0, 1.0)) == pytest.approx((0.0, -1.0, 0.0))
    assert ned.point_to_unity((0.0, 1.0, 0.0)) == pytest.approx((1.0, 0.0, 0.0))  # +Y is starboard in NED
    assert ned.determinant == pytest.approx(-1.0)


def test_invalid_conventions_fail() -> None:
    with pytest.raises(FrameError):
        UnityFrameMapper(FrameConvention(up_axis="+X", forward_axis="+X"))
    with pytest.raises(FrameError):
        UnityFrameMapper(FrameConvention(units="ft"))
