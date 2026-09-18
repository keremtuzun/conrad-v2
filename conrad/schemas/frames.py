"""Coordinate frames. Every spatial quantity names its frame (ch2 Coordinate frames, ch28).

Convention: ``T_A_from_B`` maps coordinates expressed in B to coordinates expressed in A.
Quaternion order is (w, x, y, z). SI units at all software boundaries.

WORLD handedness / vertical axis are declared per scenario through :class:`FrameConvention`; the
physical-world convention remains OPEN (ADR-0002) and is never silently assumed.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from pydantic import Field, field_validator, model_validator

from conrad.schemas.base import ConradModel

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

WORLD = "WORLD"
ROBOT = "ROBOT"

_AXIS_PATTERN = "^[+-][XYZ]$"


class FrameError(ValueError):
    """Frames were mixed, missing or not connected."""


class Handedness(str, Enum):
    RIGHT = "RIGHT"
    LEFT = "LEFT"


class FrameConvention(ConradModel):
    """Declared convention of a root frame. ``authority`` records who decided it."""

    handedness: Handedness = Handedness.RIGHT
    up_axis: str = Field(default="+Z", pattern=_AXIS_PATTERN)
    forward_axis: str = Field(default="+X", pattern=_AXIS_PATTERN)
    units: str = "m"
    # The physical convention is OPEN until the frame-contract ADR is accepted.
    authority: str = "SIMULATION_DEFAULT"


def _normalize(q: Quat) -> Quat:
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-12:
        raise FrameError("zero-norm quaternion")
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_to_matrix(q: Quat) -> np.ndarray:
    w, x, y, z = _normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(m: np.ndarray) -> Quat:
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(m[i, i] - m[j, j] - m[k, k] + 1.0, 1e-18)) * 2
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (m[j, i] + m[i, j]) / s
        v[k] = (m[k, i] + m[i, k]) / s
        q = ((m[k, j] - m[j, k]) / s, v[0], v[1], v[2])
    q = _normalize(q)
    return q if q[0] >= 0 else (-q[0], -q[1], -q[2], -q[3])


def quat_from_euler(roll: float, pitch: float, yaw: float) -> Quat:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return _normalize(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def euler_from_quat(q: Quat) -> Vec3:
    w, x, y, z = _normalize(q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (roll, pitch, yaw)


class Transform(ConradModel):
    """``T_parent_from_child``: p_parent = R @ p_child + t."""

    parent_frame: str = Field(min_length=1)
    child_frame: str = Field(min_length=1)
    translation_m: Vec3
    rotation_wxyz: Quat = (1.0, 0.0, 0.0, 0.0)

    @field_validator("rotation_wxyz")
    @classmethod
    def _unit(cls, q: Quat) -> Quat:
        return _normalize(q)

    @model_validator(mode="after")
    def _distinct(self) -> Transform:
        if self.parent_frame == self.child_frame:
            raise FrameError("transform parent and child frames are identical")
        return self

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_matrix(self.rotation_wxyz)

    @property
    def translation(self) -> np.ndarray:
        return np.asarray(self.translation_m, dtype=np.float64)

    def inverse(self) -> Transform:
        r_inv = self.rotation.T
        t_inv = -r_inv @ self.translation
        return Transform(
            parent_frame=self.child_frame,
            child_frame=self.parent_frame,
            translation_m=(float(t_inv[0]), float(t_inv[1]), float(t_inv[2])),
            rotation_wxyz=matrix_to_quat(r_inv),
        )

    def compose(self, other: Transform) -> Transform:
        """``self`` = T_A_from_B, ``other`` = T_B_from_C  ->  T_A_from_C."""
        if self.child_frame != other.parent_frame:
            raise FrameError(
                f"cannot compose {self.parent_frame}<-{self.child_frame} with "
                f"{other.parent_frame}<-{other.child_frame}"
            )
        r = self.rotation @ other.rotation
        t = self.rotation @ other.translation + self.translation
        return Transform(
            parent_frame=self.parent_frame,
            child_frame=other.child_frame,
            translation_m=(float(t[0]), float(t[1]), float(t[2])),
            rotation_wxyz=matrix_to_quat(r),
        )


class FramedPoint(ConradModel):
    frame_id: str = Field(min_length=1)
    xyz_m: Vec3

    def array(self) -> np.ndarray:
        return np.asarray(self.xyz_m, dtype=np.float64)


class Pose(ConradModel):
    """Pose of a body expressed in ``frame_id`` with optional 6x6 covariance (x,y,z,roll,pitch,yaw)."""

    frame_id: str = Field(min_length=1)
    position_m: Vec3
    orientation_wxyz: Quat = (1.0, 0.0, 0.0, 0.0)
    covariance_6x6: tuple[float, ...] | None = None

    @field_validator("orientation_wxyz")
    @classmethod
    def _unit(cls, q: Quat) -> Quat:
        return _normalize(q)

    @field_validator("covariance_6x6")
    @classmethod
    def _cov_len(cls, c: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if c is not None and len(c) != 36:
            raise ValueError("covariance_6x6 must have 36 row-major entries")
        return c

    def position_sigma_m(self) -> float | None:
        """RMS positional 1-sigma, or None when covariance is unknown (never a fabricated zero)."""
        if self.covariance_6x6 is None:
            return None
        c = self.covariance_6x6
        return math.sqrt(max(c[0] + c[7] + c[14], 0.0) / 3.0)

    def as_transform(self, child_frame: str) -> Transform:
        return Transform(
            parent_frame=self.frame_id,
            child_frame=child_frame,
            translation_m=self.position_m,
            rotation_wxyz=self.orientation_wxyz,
        )


class SpatialSupport(ConradModel):
    """Region of space a piece of evidence/belief speaks about: a framed centre plus extent."""

    frame_id: str = Field(min_length=1)
    center_m: Vec3
    half_extent_m: Vec3 = (0.0, 0.0, 0.0)
    position_sigma_m: float | None = Field(default=None, ge=0)


def transform_point(p: FramedPoint, t_a_from_b: Transform) -> FramedPoint:
    if p.frame_id != t_a_from_b.child_frame:
        raise FrameError(f"point is in {p.frame_id!r} but transform expects {t_a_from_b.child_frame!r}")
    out = t_a_from_b.rotation @ p.array() + t_a_from_b.translation
    return FramedPoint(frame_id=t_a_from_b.parent_frame, xyz_m=(float(out[0]), float(out[1]), float(out[2])))


class FrameGraph:
    """Tree of frames rooted at WORLD. Lookup composes explicit transforms; nothing is assumed."""

    def __init__(self, root: str = WORLD) -> None:
        self.root = root
        self._to_parent: dict[str, Transform] = {}

    def set(self, t_parent_from_child: Transform) -> None:
        if t_parent_from_child.child_frame == self.root:
            raise FrameError("root frame cannot have a parent")
        self._to_parent[t_parent_from_child.child_frame] = t_parent_from_child

    def _root_from(self, frame: str) -> Transform | None:
        acc: Transform | None = None
        seen = {frame}
        while frame != self.root:
            if frame not in self._to_parent:
                raise FrameError(f"frame {frame!r} is not connected to {self.root!r}")
            t = self._to_parent[frame]
            acc = t if acc is None else t.compose(acc)
            frame = t.parent_frame
            if frame in seen:
                raise FrameError("cycle in frame graph")
            seen.add(frame)
        return acc

    def lookup(self, target: str, source: str) -> Transform:
        """Return T_target_from_source."""
        if target == source:
            raise FrameError("identity lookup requested; frames are identical")
        root_from_source = self._root_from(source)
        root_from_target = self._root_from(target)
        if root_from_target is None:
            assert root_from_source is not None
            return root_from_source
        target_from_root = root_from_target.inverse()
        return target_from_root if root_from_source is None else target_from_root.compose(root_from_source)
