"""Sensor models O_m = H_m(S, V, eta_m): depth/range, point cloud, forward sonar, RGB-like. TRUTH PLANE.

Fidelity level 2 (synthetic geometric sensor data) with level-3-style optical/acoustic degradation knobs.
Every coefficient is a SIMULATION DEFAULT (SYNTHETIC_ONLY), not a calibrated sensor model.

Degradation keys (SensingContext.degradation, all >= 0, 0 = clean):
    turbidity, backscatter, blur, low_light, range_noise_scale, dropout, speckle, multipath,
    attenuation, calibration_bias_m, fault (>= 1 -> sensor FAULT, nothing sensed).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from conrad.schemas.frames import Pose
from conrad.schemas.world import SensorSpec
from conrad.twins.twin2s.config import RaycastConfig
from conrad.twins.twin2s.raycast import (
    RayHits,
    fan_directions,
    pinhole_directions,
    sensor_world_pose,
    sphere_trace,
)
from conrad.twins.twin2s.sdf import Arr
from conrad.twins.twin2s.world import SEMANTIC_CLASSES, SpatialWorld

DEGRADATION_KEYS = (
    "turbidity",
    "backscatter",
    "blur",
    "low_light",
    "range_noise_scale",
    "dropout",
    "speckle",
    "multipath",
    "attenuation",
    "calibration_bias_m",
    "fault",
)

# Per-class optical albedo (RGB) and acoustic reflectivity. Synthetic rendering constants only.
ALBEDO: dict[str, tuple[float, float, float]] = {
    "water": (0.0, 0.0, 0.0),
    "seafloor": (0.55, 0.50, 0.38),
    "rock": (0.42, 0.40, 0.38),
    "pipeline_segment": (0.62, 0.55, 0.30),
    "pipeline_bend": (0.62, 0.55, 0.30),
    "weld": (0.75, 0.72, 0.65),
    "support": (0.70, 0.70, 0.68),
    "frame": (0.55, 0.30, 0.20),
    "cable": (0.15, 0.15, 0.15),
    "debris": (0.35, 0.28, 0.25),
    "biological": (0.30, 0.55, 0.30),
    "dynamic_object": (0.60, 0.45, 0.40),
    "anchor": (0.60, 0.60, 0.58),
}
REFLECTIVITY: dict[str, float] = {
    "water": 0.0,
    "seafloor": 0.25,
    "rock": 0.5,
    "pipeline_segment": 0.9,
    "pipeline_bend": 0.9,
    "weld": 0.95,
    "support": 0.7,
    "frame": 0.85,
    "cable": 0.6,
    "debris": 0.6,
    "biological": 0.2,
    "dynamic_object": 0.15,
    "anchor": 0.7,
}
WATER_RGB = np.array([0.10, 0.35, 0.45])


def check_degradation(deg: dict[str, float]) -> None:
    unknown = set(deg) - set(DEGRADATION_KEYS)
    if unknown:
        raise ValueError(f"unknown degradation keys {sorted(unknown)}")
    if any(v < 0 for v in deg.values()):
        raise ValueError("degradation values must be >= 0")


@dataclass(frozen=True)
class Render:
    """Measured arrays (observation plane) plus truth arrays (supervision plane only)."""

    measured: dict[str, NDArray[np.generic]]
    truth: dict[str, NDArray[np.generic]]


def _trace(
    world: SpatialWorld, spec: SensorSpec, pose: Pose, dirs_s: Arr, max_r: float, rc: RaycastConfig
) -> tuple[RayHits, Arr, Arr]:
    rot, origin = sensor_world_pose(pose, spec.mount_pose)
    return sphere_trace(world, origin, dirs_s @ rot.T, max_r, rc), rot, origin


def render_depth(
    world: SpatialWorld,
    spec: SensorSpec,
    pose: Pose,
    deg: dict[str, float],
    rng: np.random.Generator,
    rc: RaycastConfig,
) -> Render:
    p = spec.parameters
    w, h, max_r = int(p["width_px"]), int(p["height_px"]), float(p["max_range_m"])
    dirs = pinhole_directions(w, h, float(p["hfov_deg"]))
    hits, _, _ = _trace(world, spec, pose, dirs, max_r, rc)
    true_r = np.where(hits.hit, hits.range_m, np.nan)
    sigma = float(p["range_noise_sigma_m"]) * (1.0 + deg.get("range_noise_scale", 0.0))
    noisy = true_r + rng.normal(0.0, 1.0, size=true_r.shape) * sigma * (1.0 + 0.05 * np.nan_to_num(true_r))
    noisy = noisy + deg.get("calibration_bias_m", 0.0)
    drop_p = min(float(p["dropout_prob"]) + deg.get("dropout", 0.0) + 0.3 * deg.get("turbidity", 0.0), 1.0)
    dropped = rng.random(true_r.shape) < drop_p
    meas = np.where(dropped | ~hits.hit | (noisy > max_r), np.nan, noisy)
    # A depth image stores ray length along each pixel direction (range image).
    img = meas.reshape(h, w).astype(np.float32)
    valid = ~np.isnan(meas)
    pts_s = (dirs[valid] * meas[valid, None]).astype(np.float32)
    return Render(
        measured={"range_image": img, "points_sensor": pts_s},
        truth={
            "true_range": true_r.reshape(h, w).astype(np.float32),
            "entity_index": hits.entity_index.reshape(h, w),
            "visible_mask": hits.hit.reshape(h, w),
            "valid_mask": valid.reshape(h, w),
            "point_entity_index": hits.entity_index[valid],
            "point_true_range": true_r[valid].astype(np.float32),
        },
    )


def render_rgb(
    world: SpatialWorld,
    spec: SensorSpec,
    pose: Pose,
    deg: dict[str, float],
    rng: np.random.Generator,
    rc: RaycastConfig,
) -> Render:
    p = spec.parameters
    w, h, max_r = int(p["width_px"]), int(p["height_px"]), float(p["max_range_m"])
    dirs = pinhole_directions(w, h, float(p["hfov_deg"]))
    hits, rot, _ = _trace(world, spec, pose, dirs, max_r, rc)
    turb = deg.get("turbidity", 0.0)
    c = 0.12 * (1.0 + 4.0 * turb + deg.get("attenuation", 0.0))
    light = float(p["light_intensity"]) * max(1.0 - deg.get("low_light", 0.0), 0.0)
    r = np.where(hits.hit, hits.range_m, max_r)
    alb = np.zeros((len(dirs), 3))
    cos_i = np.zeros(len(dirs))
    if hits.hit.any():
        classes = np.array([SEMANTIC_CLASSES.index(e.semantic_class) for e in world.entities])
        table = np.array([ALBEDO[k] for k in SEMANTIC_CLASSES])
        alb[hits.hit] = table[classes[hits.entity_index[hits.hit]]]
        n = world.normals(hits.points_m[hits.hit])
        cos_i[hits.hit] = np.clip(-np.einsum("ij,ij->i", n, dirs[hits.hit] @ rot.T), 0.0, 1.0)
    direct = light * alb * (cos_i / (1.0 + 0.05 * r**2) * np.exp(-2.0 * c * r))[:, None]
    veil = (0.15 + deg.get("backscatter", 0.0) + 0.5 * turb) * light * (1.0 - np.exp(-c * r))
    img = (direct + veil[:, None] * WATER_RGB).reshape(h, w, 3)
    blur = deg.get("blur", 0.0) + 0.5 * turb
    if blur > 0:
        img = gaussian_filter(img, sigma=(2.0 * blur, 2.0 * blur, 0.0), mode="nearest")
    img = img + rng.normal(0.0, 0.01 + 0.04 * deg.get("low_light", 0.0), size=img.shape)
    img8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    return Render(
        measured={"image": img8},
        truth={
            "entity_index": hits.entity_index.reshape(h, w),
            "visible_mask": hits.hit.reshape(h, w),
            "true_range": np.where(hits.hit, hits.range_m, np.nan).reshape(h, w).astype(np.float32),
        },
    )


def render_sonar(
    world: SpatialWorld,
    spec: SensorSpec,
    pose: Pose,
    deg: dict[str, float],
    rng: np.random.Generator,
    rc: RaycastConfig,
) -> Render:
    """Forward-looking sonar polar image (range bins x beams)."""
    p = spec.parameters
    nb, nr = int(p["n_beams"]), int(p["n_range_bins"])
    rmin, rmax = float(p["min_range_m"]), float(p["max_range_m"])
    dirs, beam = fan_directions(nb, float(p["hfov_deg"]), int(p["n_elevation_rays"]), float(p["vfov_deg"]))
    hits, rot, _ = _trace(world, spec, pose, dirs, rmax, rc)
    ok = hits.hit & (hits.range_m >= rmin)
    img = np.zeros((nr, nb))
    ent_img = np.full((nr, nb), -1, dtype=np.int64)
    if ok.any():
        refl = np.array([REFLECTIVITY[e.semantic_class] for e in world.entities])[hits.entity_index[ok]]
        n = world.normals(hits.points_m[ok])
        cos_i = np.abs(np.einsum("ij,ij->i", n, dirs[ok] @ rot.T))
        alpha = 0.02 * (1.0 + deg.get("attenuation", 0.0))
        r = hits.range_m[ok]
        inten = refl * (0.2 + 0.8 * cos_i) * np.exp(-2.0 * alpha * r) / (1.0 + 0.02 * r)
        bins = np.clip(((r - rmin) / (rmax - rmin) * nr).astype(np.int64), 0, nr - 1)
        np.add.at(img, (bins, beam[ok]), inten)
        order = np.argsort(inten, kind="stable")  # strongest contributor wins the truth label
        ent_img[bins[order], beam[ok][order]] = hits.entity_index[ok][order]
    true_img = img.copy()  # geometric returns without multipath ghosts or speckle
    mp = deg.get("multipath", 0.0)
    if ok.any() and mp > 0:
        ghost = np.clip(bins + max(int(0.1 * nr), 1), 0, nr - 1)
        np.add.at(img, (ghost, beam[ok]), mp * 0.5 * inten)
    s = float(p["speckle_sigma"]) + deg.get("speckle", 0.0)
    shape = 1.0 / max(s * s, 1e-6)
    img = img * rng.gamma(shape, 1.0 / shape, size=img.shape)
    img = img + np.abs(rng.normal(0.0, 0.01 * (1.0 + deg.get("turbidity", 0.0)), size=img.shape))
    return Render(
        measured={"polar_image": img.astype(np.float32)},
        truth={
            "true_intensity": true_img.astype(np.float32),
            "entity_index": ent_img,
            "visible_mask": ent_img >= 0,
        },
    )
