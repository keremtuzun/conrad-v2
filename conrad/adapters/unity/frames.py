"""Conrad <-> Unity frame conversion (ch28 frame contract, ch21 Unity V2).

Conventions handled here and nowhere else:

* Conrad WORLD / ROBOT (simulation default, ``FrameConvention()``): right-handed, +X forward,
  +Y left, +Z up.
* Unity: left-handed, +X right, +Y up, +Z forward.

``UNITY_FROM_CONRAD`` is the 3x3 axis map ``S`` with ``p_unity = S @ p_conrad``. ``det(S) = -1``
because handedness flips, therefore:

* points / polar vectors (position, velocity, force, acceleration): ``v_u = S v_c``
* rotation matrices: ``R_u = S R_c S^T`` (a proper rotation again)
* quaternions (w, x, y, z): scalar unchanged, vector part ``det(S) * S @ q_vec``
* axial vectors (angular velocity, torque): ``w_u = det(S) * S @ w_c``

For the default convention this gives ``(x, y, z)_c -> (-y, z, x)_u`` for points and
``(w, x, y, z)_c -> (w, y, -z, -x)_u`` for quaternions. The physical-world convention is OPEN
(ADR-0002); the mapper is therefore built from a declared :class:`FrameConvention` and never assumed.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import numpy as np

from conrad.schemas.frames import (
    FrameConvention,
    FrameError,
    Handedness,
    Pose,
    Quat,
    Transform,
    Vec3,
    matrix_to_quat,
    quat_to_matrix,
)

UNITY_FRAME_CONVENTION = "UNITY_LH_X_RIGHT_Y_UP_Z_FORWARD"
UNITY_WORLD = "UNITY_WORLD"


def _axis(spec: str) -> np.ndarray:
    sign = 1.0 if spec[0] == "+" else -1.0
    vec = np.zeros(3, dtype=np.float64)
    vec["XYZ".index(spec[1])] = sign
    return vec


def _vec3(a: np.ndarray) -> Vec3:
    return (float(a[0]), float(a[1]), float(a[2]))


class UnityFrameMapper:
    """Linear axis map between one declared Conrad convention and Unity's fixed convention."""

    def __init__(self, convention: FrameConvention | None = None) -> None:
        conv = convention if convention is not None else FrameConvention()
        if conv.units != "m":
            raise FrameError(f"Unity adapter requires metres at the boundary, got {conv.units!r}")
        forward = _axis(conv.forward_axis)
        up = _axis(conv.up_axis)
        if abs(float(forward @ up)) > 1e-12:
            raise FrameError("forward_axis and up_axis must be orthogonal")
        # 'right' expressed in Conrad coordinates: forward x up for a right-handed frame.
        right = np.cross(forward, up) if conv.handedness is Handedness.RIGHT else np.cross(up, forward)
        # Rows are the Unity axes (right, up, forward) expressed in Conrad coordinates.
        self._s = np.vstack([right, up, forward])
        self._det = float(np.linalg.det(self._s))
        if abs(abs(self._det) - 1.0) > 1e-9:
            raise FrameError("axis map is not orthonormal")
        self.convention = conv

    @property
    def unity_from_conrad(self) -> np.ndarray:
        return self._s.copy()

    @property
    def conrad_from_unity(self) -> np.ndarray:
        return self._s.T.copy()

    @property
    def determinant(self) -> float:
        return self._det

    # -- polar vectors / points -------------------------------------------------------------
    def vector_to_unity(self, v: Vec3) -> Vec3:
        return _vec3(self._s @ np.asarray(v, dtype=np.float64))

    def vector_to_conrad(self, v: Vec3) -> Vec3:
        return _vec3(self._s.T @ np.asarray(v, dtype=np.float64))

    point_to_unity = vector_to_unity
    point_to_conrad = vector_to_conrad

    # -- axial vectors ----------------------------------------------------------------------
    def axial_to_unity(self, w: Vec3) -> Vec3:
        return _vec3(self._det * (self._s @ np.asarray(w, dtype=np.float64)))

    def axial_to_conrad(self, w: Vec3) -> Vec3:
        return _vec3(self._det * (self._s.T @ np.asarray(w, dtype=np.float64)))

    angular_velocity_to_unity = axial_to_unity
    angular_velocity_to_conrad = axial_to_conrad

    # -- rotations --------------------------------------------------------------------------
    def rotation_to_unity(self, r_conrad: np.ndarray) -> np.ndarray:
        return self._s @ r_conrad @ self._s.T

    def rotation_to_conrad(self, r_unity: np.ndarray) -> np.ndarray:
        return self._s.T @ r_unity @ self._s

    def quat_to_unity(self, q_wxyz: Quat) -> Quat:
        """Conrad (w,x,y,z) -> Unity (w,x,y,z). Unity's own struct order is (x,y,z,w); the wire uses wxyz."""
        return _canonical(
            matrix_to_quat(self.rotation_to_unity(quat_to_matrix(q_wxyz))), self._s, q_wxyz, self._det
        )

    def quat_to_conrad(self, q_wxyz: Quat) -> Quat:
        return _canonical(
            matrix_to_quat(self.rotation_to_conrad(quat_to_matrix(q_wxyz))), self._s.T, q_wxyz, self._det
        )

    # -- poses ------------------------------------------------------------------------------
    def pose_to_unity(self, pose: Pose, unity_frame: str = UNITY_WORLD) -> Pose:
        return Pose(
            frame_id=unity_frame,
            position_m=self.point_to_unity(pose.position_m),
            orientation_wxyz=self.quat_to_unity(pose.orientation_wxyz),
        )

    def pose_to_conrad(self, pose: Pose, conrad_frame: str) -> Pose:
        return Pose(
            frame_id=conrad_frame,
            position_m=self.point_to_conrad(pose.position_m),
            orientation_wxyz=self.quat_to_conrad(pose.orientation_wxyz),
        )

    def transform_to_unity(self, t: Transform, parent: str, child: str) -> Transform:
        return Transform(
            parent_frame=parent,
            child_frame=child,
            translation_m=self.point_to_unity(t.translation_m),
            rotation_wxyz=self.quat_to_unity(t.rotation_wxyz),
        )


def _canonical(q_from_matrix: Quat, s: np.ndarray, q_in: Quat, det: float) -> Quat:
    """Keep the closed-form sign (w preserved) so q and -q do not flip between calls."""
    w = q_in[0]
    vec = det * (s @ np.asarray(q_in[1:], dtype=np.float64))
    norm = float(np.sqrt(w * w + float(vec @ vec)))
    closed = (w / norm, float(vec[0]) / norm, float(vec[1]) / norm, float(vec[2]) / norm)
    # The closed form and the matrix route must describe the same rotation; this is cheap and
    # guards any future edit of either path.
    dot = abs(sum(a * b for a, b in zip(closed, q_from_matrix, strict=True)))
    if dot < 1.0 - 1e-6:
        raise FrameError("quaternion conversion paths disagree")
    return closed


DEFAULT_MAPPER = UnityFrameMapper()
