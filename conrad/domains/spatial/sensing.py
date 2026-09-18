"""Belief-side sensor geometry: where an observation's rays start and point, from the ESTIMATED robot pose,
the robot's own sensor configuration (``SensorSpec`` mount + intrinsics) and the pose covariance.

Nothing here knows the true pose. The pinhole / fan conventions are the declared sensor model of the
robot configuration (sensor frame: +X boresight, +Y left, +Z up; image u grows right, v grows down).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from conrad.domains.spatial.config import SpatialConfig
from conrad.domains.spatial.keys import FloatArr
from conrad.schemas.frames import Pose, quat_to_matrix
from conrad.schemas.world import SensorSpec


class SpatialSensingError(ValueError):
    """An observation cannot be placed in the map (no pose estimate, unknown sensor, bad payload)."""


@dataclass(frozen=True)
class PoseUncertainty:
    sigma_pos_m: float
    sigma_rot_rad: float
    declared: bool  # False when the covariance was missing and the configured prior was used

    def at_lever(self, lever_m: FloatArr) -> FloatArr:
        """Isotropic 1-sigma displacement of a point ``lever_m`` metres from the robot origin."""
        lever = np.asarray(lever_m, dtype=np.float64)
        return np.asarray(np.sqrt(self.sigma_pos_m**2 + (2.0 / 3.0) * (self.sigma_rot_rad * lever) ** 2))


@dataclass(frozen=True)
class SensorFrame:
    rotation: FloatArr  # R_world_from_sensor
    origin: FloatArr  # sensor origin, WORLD
    robot_position: FloatArr
    pose_sigma: PoseUncertainty


def pose_uncertainty(pose: Pose, cfg: SpatialConfig) -> PoseUncertainty:
    """Covariance -> RMS position / rotation sigmas. A missing covariance is never treated as zero."""
    if not cfg.pose.use_pose_covariance:
        return PoseUncertainty(0.0, 0.0, pose.covariance_6x6 is not None)
    if pose.covariance_6x6 is None:
        return PoseUncertainty(cfg.pose.unknown_pose_sigma_m, cfg.pose.unknown_pose_sigma_rad, False)
    c = np.asarray(pose.covariance_6x6, dtype=np.float64).reshape(6, 6)
    pos = math.sqrt(max(float(np.trace(c[:3, :3])), 0.0) / 3.0)
    rot = math.sqrt(max(float(np.trace(c[3:, 3:])), 0.0) / 3.0)
    return PoseUncertainty(pos, rot, True)


def sensor_frame(pose: Pose | None, spec: SensorSpec, cfg: SpatialConfig) -> SensorFrame:
    if pose is None:
        raise SpatialSensingError("observation has no robot_pose_estimate; it cannot be placed")
    if pose.frame_id != cfg.grid.frame_id:
        raise SpatialSensingError(f"pose is in {pose.frame_id!r}; the map frame is {cfg.grid.frame_id!r}")
    r_wr = quat_to_matrix(pose.orientation_wxyz)
    t_wr = np.asarray(pose.position_m, dtype=np.float64)
    r_rs = quat_to_matrix(spec.mount_pose.orientation_wxyz)
    origin = r_wr @ np.asarray(spec.mount_pose.position_m, dtype=np.float64) + t_wr
    return SensorFrame(r_wr @ r_rs, origin, t_wr, pose_uncertainty(pose, cfg))


def pinhole_directions(width: int, height: int, hfov_deg: float) -> FloatArr:
    """Unit ray directions (H*W, 3) in the sensor frame, row-major."""
    f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
    u = (np.arange(width) + 0.5 - 0.5 * width) / f
    v = (np.arange(height) + 0.5 - 0.5 * height) / f
    uu, vv = np.meshgrid(u, v)
    d = np.stack([np.ones_like(uu), -uu, -vv], axis=-1).reshape(-1, 3)
    return np.asarray(d / np.linalg.norm(d, axis=1, keepdims=True), dtype=np.float64)


def fan_directions(azimuths_rad: FloatArr, elevations_rad: FloatArr) -> FloatArr:
    """(n_az, n_el, 3) unit directions of a sonar fan in the sensor frame."""
    a, e = np.meshgrid(azimuths_rad, elevations_rad, indexing="ij")
    return np.asarray(np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1))


@dataclass(frozen=True)
class SensorFov:
    hfov_rad: float
    vfov_rad: float
    max_range_m: float
    min_range_m: float


def sensor_fov(spec: SensorSpec) -> SensorFov:
    p = spec.parameters
    if "hfov_deg" not in p or "max_range_m" not in p:
        raise SpatialSensingError(f"sensor modality {spec.modality} has no viewing geometry")
    hfov = math.radians(float(p["hfov_deg"]))
    if "vfov_deg" in p:
        vfov = math.radians(float(p["vfov_deg"]))
    elif "width_px" in p and "height_px" in p:
        vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * float(p["height_px"]) / float(p["width_px"]))
    else:
        raise SpatialSensingError(f"sensor {spec.sensor_id} declares no vertical field of view")
    return SensorFov(hfov, vfov, float(p["max_range_m"]), float(p.get("min_range_m", 0.0)))


def range_noise_sigma(spec: SensorSpec, ranges_m: FloatArr, cfg: SpatialConfig) -> FloatArr:
    """Declared range noise of the sensor model (datasheet-style), growing with range."""
    base = float(spec.parameters.get("range_noise_sigma_m", cfg.sensor.default_range_noise_sigma_m))
    return np.asarray(base * (1.0 + cfg.sensor.range_noise_growth_per_m * np.asarray(ranges_m)))
