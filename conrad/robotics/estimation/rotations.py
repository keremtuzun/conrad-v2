"""Quaternion helpers, (w, x, y, z), body->world. Shared by the robotics stack and the sim kernel."""

from __future__ import annotations

import math

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    return q / n


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_exp(rotvec: np.ndarray) -> np.ndarray:
    """Unit quaternion of a rotation vector."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-10:
        return quat_normalize(np.array([1.0, 0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2]]))
    axis = rotvec / angle
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_log(q: np.ndarray) -> np.ndarray:
    """Rotation vector of a unit quaternion (shortest path)."""
    if q[0] < 0:
        q = -q
    v = q[1:]
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        return 2.0 * v
    return 2.0 * math.atan2(n, float(q[0])) * v / n


def quat_error_body(q_desired: np.ndarray, q_actual: np.ndarray) -> np.ndarray:
    """Rotation vector, in the actual body frame, that takes actual to desired."""
    return quat_log(quat_mul(quat_conj(q_actual), q_desired))


def yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
