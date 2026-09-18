"""Synthetic sensors/observations for Model2S unit tests (not a test module).

Test-side geometry is analytic (planes / panels), computed here, never inside the belief package.
"""

from __future__ import annotations

import math

import numpy as np

from conrad.domains.spatial.config import spatial_config
from conrad.domains.spatial.model import Model2S
from conrad.domains.spatial.sensing import fan_directions, pinhole_directions
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.frames import ROBOT, WORLD, Pose, quat_from_euler
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.timebase import stamp
from conrad.schemas.world import SensorSpec

W, H = 48, 36


def depth_spec(ids: IdFactory, w: int = W, h: int = H, max_range: float = 10.0) -> SensorSpec:
    return SensorSpec(
        sensor_id=ids.new(),
        modality="DEPTH_RANGE",
        frame_id="SENSOR_DEPTH_RANGE",
        mount_pose=Pose(frame_id=ROBOT, position_m=(0.0, 0.0, 0.0)),
        rate_hz=5.0,
        parameters={
            "width_px": w,
            "height_px": h,
            "hfov_deg": 60.0,
            "max_range_m": max_range,
            "range_noise_sigma_m": 0.01,
        },
    )


def sonar_spec(ids: IdFactory) -> SensorSpec:
    return SensorSpec(
        sensor_id=ids.new(),
        modality="SONAR",
        frame_id="SENSOR_SONAR",
        mount_pose=Pose(frame_id=ROBOT, position_m=(0.0, 0.0, 0.0)),
        rate_hz=5.0,
        parameters={
            "n_beams": 32,
            "n_range_bins": 64,
            "hfov_deg": 60.0,
            "vfov_deg": 20.0,
            "max_range_m": 8.0,
            "min_range_m": 0.3,
        },
    )


def pose(x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0, sigma: float | None = 0.0) -> Pose:
    cov = None if sigma is None else tuple(float(c) for c in np.diag([sigma**2] * 3 + [0.0] * 3).ravel())
    return Pose(
        frame_id=WORLD, position_m=(x, y, z), orientation_wxyz=quat_from_euler(0, 0, yaw), covariance_6x6=cov
    )


def panel_ranges(
    dirs: np.ndarray,
    origin: np.ndarray,
    wall_x: float,
    half_y: float = 50.0,
    half_z: float = 50.0,
    hole_y: tuple[float, float] | None = None,
) -> np.ndarray:
    """Range along each WORLD ray to the plane x = wall_x inside |y|<=half_y, |z|<=half_z (NaN elsewhere)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wall_x - origin[0]) / dirs[:, 0]
    p = origin + t[:, None] * dirs
    ok = (t > 0) & (np.abs(p[:, 1]) <= half_y) & (np.abs(p[:, 2]) <= half_z)
    if hole_y is not None:
        ok &= ~((p[:, 1] >= hole_y[0]) & (p[:, 1] < hole_y[1]))
    return np.where(ok, t, np.nan)


def depth_image(spec: SensorSpec, robot: Pose, wall_x: float, **kw: object) -> np.ndarray:
    p = spec.parameters
    d = pinhole_directions(p["width_px"], p["height_px"], p["hfov_deg"])
    yaw = 2 * math.atan2(robot.orientation_wxyz[3], robot.orientation_wxyz[0])
    rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    r = panel_ranges(d @ rot.T, np.asarray(robot.position_m), wall_x, **kw)  # type: ignore[arg-type]
    r = np.where(r > p["max_range_m"], np.nan, r)
    return r.reshape(p["height_px"], p["width_px"]).astype(np.float32)


def sonar_image(spec: SensorSpec, wall_x: float) -> np.ndarray:
    """Polar image of a wall at x = wall_x seen from the origin, 5 elevation rays per beam."""
    p = spec.parameters
    nb, nr, rmin, rmax = p["n_beams"], p["n_range_bins"], p["min_range_m"], p["max_range_m"]
    az = np.radians(np.linspace(p["hfov_deg"] / 2, -p["hfov_deg"] / 2, nb))
    el = np.radians(np.linspace(-p["vfov_deg"] / 2, p["vfov_deg"] / 2, 5))
    d = fan_directions(az, el)
    t = wall_x / d[..., 0]
    img = np.zeros((nr, nb))
    bins = np.clip(((t - rmin) / (rmax - rmin) * nr).astype(int), 0, nr - 1)
    for b in range(nb):
        for e in range(len(el)):
            if t[b, e] <= rmax:
                img[bins[b, e], b] += 0.3
    return img.astype(np.float32)


def make_obs(
    ids: IdFactory,
    store: ObjectStore,
    spec: SensorSpec,
    robot: Pose | None,
    arr: np.ndarray,
    t: float,
    modality: Modality | None = None,
) -> Observation:
    return Observation(
        observation_id=ids.new(),
        mission_id=ids.new(),
        run_id=ids.new(),
        trace_id=ids.new(),
        sensor_id=spec.sensor_id,
        modality=modality or Modality(spec.modality),
        timestamp=stamp(t, "SIM"),
        sensor_frame=spec.frame_id,
        robot_pose_estimate=robot,
        payload_ref=store.put_array(arr),
        sensor_context={"settings": dict(spec.parameters)},
    )


def new_model(
    ids: IdFactory,
    store: ObjectStore,
    specs: list[SensorSpec],
    cls: type[Model2S] = Model2S,
    **overrides: object,
) -> Model2S:
    m = cls(ids, store, spatial_config(overrides)) if overrides else cls(ids, store)
    m.initialize({"sensors": specs})
    return m


def line(x0: float, x1: float, y: float = 0.1, z: float = 0.1, step: float = 0.25) -> np.ndarray:
    xs = np.arange(x0, x1, step) + 0.125
    return np.stack([xs, np.full_like(xs, y), np.full_like(xs, z)], axis=1)
