"""Observation payload -> per-resolution cell-mass function, per modality. BELIEF PLANE.

DEPTH_RANGE ("HxW" range image, NaN = no return/dropout) and POINT_CLOUD ("Nx3", sensor frame) give
returns with ray free space. SONAR ("range_bins x beams" polar intensity) gives returns whose elevation
is ambiguous inside the vertical beam: each detection is spread over the whole elevation arc (so sonar
never produces a thin surface) and its U_A reflects that arc; free space is carved only in beams with a
detection and only before the first one. RGB carries no range and is not a geometric measurement here.

The pose blur uses the pose sigma at the observation's median lever arm. A finer level whose kernel cannot
represent that blur returns None: the map then gives it the coarser level's masses (no fake detail).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math

import numpy as np

from conrad.domains.spatial.config import SpatialConfig
from conrad.domains.spatial.integrate import (
    CellMasses,
    MassFn,
    combine,
    empty_partial,
    free_masses,
    hit_masses,
    pose_blur,
)
from conrad.domains.spatial.keys import FloatArr
from conrad.domains.spatial.sensing import (
    SensorFrame,
    SpatialSensingError,
    fan_directions,
    pinhole_directions,
    range_noise_sigma,
    sensor_fov,
)
from conrad.schemas.world import SensorSpec


def _blur_sigma(frame: SensorFrame, points: FloatArr) -> float:
    if len(points) == 0:
        return float(frame.pose_sigma.sigma_pos_m)
    lever = float(np.median(np.linalg.norm(points - frame.robot_position, axis=1)))
    return float(frame.pose_sigma.at_lever(np.array([lever]))[0])


def _representable(res: float, sigma: float, cfg: SpatialConfig) -> bool:
    return (
        res >= cfg.grid.base_voxel_m
        or cfg.pose.splat_sigma_k * sigma <= cfg.pose.max_splat_radius_cells * res
    )


def ray_mass_fn(
    frame: SensorFrame,
    dirs_world: FloatArr,
    ranges: FloatArr,
    spec: SensorSpec,
    cfg: SpatialConfig,
    quality: float,
) -> MassFn:
    fov = sensor_fov(spec)
    r = np.asarray(ranges, dtype=np.float64)
    valid = np.isfinite(r) & (r > fov.min_range_m) & (r <= fov.max_range_m)
    d, rv = dirs_world[valid], r[valid]
    pts = frame.origin[None, :] + rv[:, None] * d
    s_meas = range_noise_sigma(spec, rv, cfg)
    sigma_b = _blur_sigma(frame, pts)
    no_ret = ~np.isfinite(r)
    k = cfg.sensor.free_margin_sigma_k

    def masses(res: float, tail_only: bool) -> CellMasses | None:
        if not _representable(res, sigma_b, cfg):
            return None
        ua = np.maximum(s_meas / (s_meas + res), 1.0 - quality)
        hits = hit_masses(res, pts, s_meas, np.full(len(rv), quality), ua, cfg)
        ends = rv - np.maximum(k * s_meas, 0.5 * res)
        dirs_f = d
        if cfg.sensor.free_space_on_no_return and no_ret.any():
            dirs_f = np.concatenate([d, dirs_world[no_ret]])
            ends = np.concatenate([ends, np.full(int(no_ret.sum()), fov.max_range_m)])
        start = np.full(len(ends), fov.min_range_m)
        if tail_only:
            start = np.maximum(start, ends - cfg.refinement.fine_free_tail_m)
        n = len(ends)
        frees = free_masses(
            res, frame.origin, dirs_f, start, ends, np.full(n, quality), np.full(n, 1 - quality), cfg
        )
        return pose_blur(combine(hits, frees, cfg), res, sigma_b, cfg)

    return masses


def depth_mass_fn(
    frame: SensorFrame, image: FloatArr, spec: SensorSpec, cfg: SpatialConfig, quality: float
) -> MassFn:
    p = spec.parameters
    h, w = int(p["height_px"]), int(p["width_px"])
    if image.shape != (h, w):
        raise SpatialSensingError(f"range image shape {image.shape} disagrees with sensor {(h, w)}")
    dirs = pinhole_directions(w, h, float(p["hfov_deg"])) @ frame.rotation.T
    return ray_mass_fn(frame, dirs, image.reshape(-1).astype(np.float64), spec, cfg, quality)


def point_cloud_mass_fn(
    frame: SensorFrame, points_sensor: FloatArr, spec: SensorSpec, cfg: SpatialConfig, quality: float
) -> MassFn:
    pts = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
    rng = np.linalg.norm(pts, axis=1)
    ok = rng > 1e-9
    dirs = (pts[ok] / rng[ok, None]) @ frame.rotation.T
    return ray_mass_fn(frame, dirs, rng[ok], spec, cfg, quality)


def sonar_detections(image: FloatArr, cfg: SpatialConfig) -> np.ndarray:
    """Boolean (bins x beams) detections: absolute floor plus a CFAR-style multiple of the median."""
    sc = cfg.sensor
    thr = max(sc.sonar_detect_threshold, sc.sonar_cfar_k * float(np.median(image)))
    return np.asarray(image > thr)


def sonar_mass_fn(
    frame: SensorFrame, image: FloatArr, spec: SensorSpec, cfg: SpatialConfig, quality: float
) -> MassFn:
    p, sc = spec.parameters, cfg.sensor
    nb, nr = int(p["n_beams"]), int(p["n_range_bins"])
    if image.shape != (nr, nb):
        raise SpatialSensingError(f"sonar image shape {image.shape} disagrees with sensor {(nr, nb)}")
    fov = sensor_fov(spec)
    dr = (fov.max_range_m - fov.min_range_m) / nr
    az = np.radians(np.linspace(0.5 * float(p["hfov_deg"]), -0.5 * float(p["hfov_deg"]), nb))
    beam_w = fov.hfov_rad / nb
    det = sonar_detections(np.asarray(image, dtype=np.float64), cfg)
    bins, beams = np.nonzero(det)
    r_hit = fov.min_range_m + (bins + 0.5) * dr
    first = np.where(det.any(axis=0), det.argmax(axis=0), -1)
    fb = np.nonzero(first >= 0)[0]
    r_first = fov.min_range_m + first[fb] * dr
    sigma_b = float(frame.pose_sigma.at_lever(np.array([float(np.median(r_hit)) if r_hit.size else 0.0]))[0])

    def masses(res: float, tail_only: bool) -> CellMasses | None:
        if not _representable(res, sigma_b, cfg):
            return None
        if bins.size == 0:
            return combine(empty_partial(), empty_partial(), cfg)
        n_el = max(sc.sonar_elevation_min_samples, math.ceil(float(r_hit.max()) * fov.vfov_rad / res) + 1)
        el = np.linspace(-0.5 * fov.vfov_rad, 0.5 * fov.vfov_rad, n_el)
        dirs_s = fan_directions(az, el)  # (nb, n_el, 3)
        pts = (frame.origin + r_hit[:, None, None] * (dirs_s[beams] @ frame.rotation.T)).reshape(-1, 3)
        rr = np.repeat(r_hit, n_el)
        s_meas = np.sqrt(dr**2 / 12.0 + (rr * beam_w) ** 2 / 12.0)
        arc = rr * fov.vfov_rad
        ua = np.maximum.reduce([s_meas / (s_meas + res), arc / (arc + res), np.full(len(rr), 1.0 - quality)])
        mass = np.full(len(rr), quality * sc.sonar_mass_scale / n_el)
        hits = hit_masses(res, pts, s_meas, mass, ua, cfg)
        frees = empty_partial()
        if sc.sonar_free_space and fb.size:
            d_free = (dirs_s[fb] @ frame.rotation.T).reshape(-1, 3)
            ends = np.repeat(r_first - dr, n_el)
            start = np.full(len(ends), fov.min_range_m)
            if tail_only:
                start = np.maximum(start, ends - cfg.refinement.fine_free_tail_m)
            n = len(ends)
            wf = np.full(n, quality * sc.sonar_free_mass_scale)
            frees = free_masses(res, frame.origin, d_free, start, ends, wf, np.full(n, 1 - quality), cfg)
        return pose_blur(combine(hits, frees, cfg), res, sigma_b, cfg)

    return masses
