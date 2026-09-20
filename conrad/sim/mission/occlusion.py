"""World family ACTIVE_INSPECTION_OCCLUDED_V1: hidden critical surface + an occluding structure. TRUTH PLANE.

Everything here is a SYNTHETIC_ONLY simulation setting, never a measured physical value, and everything is
additive: it runs only when ``MissionWorldOptions.defect_variation`` / ``.occlusion`` are enabled, which only
the ``I4-OCCLUDED`` scenario does (``docs/audits/I4_WORLD_FAMILY.md``).

Two truth-side mechanisms:

* per-world sampling of the hidden defect (severity, patch tilt, patch extent) from declared ranges, so worlds
  differ in difficulty instead of all carrying the identical defect;
* one unregistered occluding structure alongside the target segment on the side the transit lane does not see:
  a longitudinal rack panel and, optionally, two posts to the seabed. It is a Twin2S entity only - not in the
  asset registry, no Twin2T / Twin2E state, named in no Observation - so the deployment can learn about it only
  through its own geometric sensors, exactly like the I5 lane obstacle. It is appended to the live
  ``SpatialWorld`` BEFORE the Unity scene is converted, so the Unity path carries the same geometry.

The rack is positioned in the frame of the target segment and of the defect patch. Nothing in the sampling
refers to a planner, to a candidate index, or to which planner would win.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.frames import matrix_to_quat
from conrad.schemas.ids import IdFactory
from conrad.sim.mission.options import DefectOptions, MissionWorldOptions, ViewOcclusionOptions
from conrad.twins.twin2s.sdf import Box
from conrad.twins.twin2s.world import SpatialEntity, SpatialWorld, class_index

DEFECT_STREAM = 0xC1
OCCLUSION_STREAM = 0xC2


def _u(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) / max(float(np.linalg.norm(v)), 1e-12)


def _draw(rng: np.random.Generator, span: tuple[float, float]) -> float:
    lo, hi = float(span[0]), float(span[1])
    return float(rng.uniform(min(lo, hi), max(lo, hi)))


def sample_defect(opts: MissionWorldOptions, seed: int) -> DefectOptions:
    """Per-world hidden defect drawn from the declared ranges (evaluation truth; never reaches the belief)."""
    v = opts.defect_variation
    rng = np.random.default_rng([seed, DEFECT_STREAM])
    return opts.defect.model_copy(
        update={
            "corrosion_depth_m": _draw(rng, v.corrosion_depth_m),
            "crack_length_m": _draw(rng, v.crack_length_m),
            "patch_tilt_deg": _draw(rng, v.patch_tilt_deg),
            "patch_half_angle_deg": _draw(rng, v.patch_half_angle_deg),
            "patch_axial_fraction": _draw(rng, v.patch_axial_fraction),
        }
    )


def patch_left(axis: list[np.ndarray], opts: MissionWorldOptions, lane: list[np.ndarray]) -> np.ndarray:
    """Untilted unit direction from the pipe axis towards the hidden defect patch.

    ``far`` / ``near`` keep their historical meaning (the WORLD +Y normal of the pipe heading, negated for
    ``near``). ``off_lane`` / ``lane`` are defined against the transit lane the mission actually flies, which
    is the physically meaningful statement "hidden from the nominal route": the sign of the WORLD +Y rule
    depends on the sampled pipeline heading, so with ``far`` the defect lands on the lane side in a share of
    the worlds (docs/audits/I4_WORLD_FAMILY.md section 1).
    """
    d = _u(axis[-1] - axis[0])
    left = np.array([-d[1], d[0], 0.0])
    if left[1] < 0:
        left = -left
    side = opts.defect.side
    if side == "near":
        return left * -1.0
    if side in ("off_lane", "lane"):
        mid = 0.5 * (np.asarray(axis[0], dtype=np.float64) + np.asarray(axis[-1], dtype=np.float64))
        to_lane = np.mean(np.asarray(lane, dtype=np.float64), axis=0) - mid
        to_lane[2] = 0.0
        towards_lane = float(left @ to_lane) > 0.0
        if (side == "off_lane") == towards_lane:
            return left * -1.0
    return left


def tilted(left: np.ndarray, tilt_deg: float) -> np.ndarray:
    """``left`` rotated ``tilt_deg`` towards +Z (the patch normal direction used by ``_patch_mask``)."""
    t = math.radians(tilt_deg)
    return _u(math.cos(t) * np.asarray(left, dtype=np.float64) + math.sin(t) * np.array([0.0, 0.0, 1.0]))


def _rotate_about(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    k = _u(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return _u(v * c + np.cross(k, v) * s + k * float(k @ v) * (1.0 - c))


def _plate(
    centre: np.ndarray, ex: np.ndarray, ey: np.ndarray, ez: np.ndarray, half: tuple[float, float, float]
) -> Box:
    m = np.column_stack([_u(ex), _u(ey), _u(ez)])
    return Box(
        center=(float(centre[0]), float(centre[1]), float(centre[2])),
        half_extents=(float(half[0]), float(half[1]), float(half[2])),
        rotation_wxyz=matrix_to_quat(m),
    )


def sample_occlusion(opts: ViewOcclusionOptions, seed: int) -> dict[str, float]:
    """The per-world placement parameters of the occluding structure (no planner ever reads these)."""
    rng = np.random.default_rng([seed, OCCLUSION_STREAM])
    return {
        "azimuth_offset_deg": _draw(rng, opts.azimuth_offset_deg),
        "window_half_deg": _draw(rng, opts.window_half_deg),
        "standoff_m": _draw(rng, opts.standoff_m),
        "half_width_m": _draw(rng, opts.half_width_m),
        "half_length_fraction": _draw(rng, opts.half_length_fraction),
        "axial_shift_fraction": _draw(rng, opts.axial_shift_fraction),
    }


def _window_direction(
    patch_dir: np.ndarray, d: np.ndarray, p: dict[str, float], opts: ViewOcclusionOptions
) -> np.ndarray:
    """Window direction: ``patch_dir`` rotated about the pipe axis by the sampled offset.

    The rotation sign is flipped when it would point the window below ``min_window_z``. A window aimed into
    the seabed is not reachable by any planner (the vehicle flies above the seabed and the shared candidate
    generator samples only non-negative elevations), so such a world would be unsolvable for every arm and
    would measure nothing. This is a reachability guard on the WORLD, identical for every planner; it never
    refers to a planner, a candidate index or a ranking rule.
    """
    both = [_rotate_about(patch_dir, d, s * math.radians(p["azimuth_offset_deg"])) for s in (1.0, -1.0)]
    if float(both[0][2]) >= opts.min_window_z:
        return both[0]
    return both[0] if float(both[0][2]) >= float(both[1][2]) else both[1]


def add_view_occluders(
    world: SpatialWorld,
    target: UUID,
    patch_dir: np.ndarray,
    opts: ViewOcclusionOptions,
    seed: int,
    seabed_z_m: float,
) -> dict[str, Any]:
    """Append the unregistered occluding structure; returns its truth record (evaluation only).

    ``patch_dir`` is the tilted unit direction from the pipe axis to the defect patch. TWO longitudinal rack
    panels flank the pipe on that side, leaving one angular WINDOW between them. The window is centred
    ``azimuth_offset_deg`` from the patch direction and is ``2 * window_half_deg`` wide, so a view that
    resolves the defect always exists (the family is never unsolvable) while most of the candidate ring is
    blocked. Each panel is pushed just outside the window: its centre sits at
    ``window_half_deg + atan(half_width / radial)`` from the window centre, so the window width is a declared
    quantity and does not depend on how wide or how far out the panels are.
    """
    i = world.index_of(target)
    prim = world.entities[i].primitive
    a = np.asarray(prim.a, dtype=np.float64)  # type: ignore[attr-defined]
    b = np.asarray(prim.b, dtype=np.float64)  # type: ignore[attr-defined]
    radius = float(prim.radius)  # type: ignore[attr-defined]
    d = _u(b - a)
    length = float(np.linalg.norm(b - a))
    p = sample_occlusion(opts, seed)
    window = _window_direction(np.asarray(patch_dir, dtype=np.float64), d, p, opts)
    half_len = max(0.15, p["half_length_fraction"] * length)
    mid = 0.5 * (a + b) + p["axial_shift_fraction"] * length * d
    radial = radius + p["standoff_m"] + opts.thickness_m
    span = math.radians(p["window_half_deg"]) + math.atan2(p["half_width_m"], radial)
    ids = IdFactory(seed).child("view_occluder")
    entities: list[SpatialEntity] = []
    panels: list[dict[str, Any]] = []
    for sign in (-1.0, 1.0):
        m = _rotate_about(window, d, sign * span)
        centre = mid + radial * m
        entities.append(
            SpatialEntity(
                entity_id=ids.new(),
                semantic_class=opts.semantic_class,
                primitive=_plate(
                    centre, d, np.cross(m, d), m, (half_len, p["half_width_m"], opts.thickness_m)
                ),
                material_id=opts.material_id,
            )
        )
        panels.append({"center_m": [float(x) for x in centre], "normal": [float(x) for x in m]})
    posts: list[dict[str, Any]] = []
    if opts.posts:
        top = float(mid[2]) + radial * float(window[2]) - opts.thickness_m
        bottom = min(seabed_z_m - 0.3, top - 0.2)
        h = 0.5 * (top - bottom)
        for sign in (-1.0, 1.0):
            foot = mid + radial * window + sign * (half_len - opts.post_inset_m) * d
            pc = np.array([foot[0], foot[1], bottom + h])
            entities.append(
                SpatialEntity(
                    entity_id=ids.new(),
                    semantic_class=opts.post_semantic_class,
                    primitive=Box(
                        center=(float(pc[0]), float(pc[1]), float(pc[2])),
                        half_extents=(opts.post_half_m, opts.post_half_m, float(h)),
                    ),
                    material_id=opts.material_id,
                )
            )
            posts.append({"center_m": [float(x) for x in pc], "half_z_m": float(h)})
    for e in entities:
        class_index(e.semantic_class)
        world.entities.append(e)
    return {
        "family": "ACTIVE_INSPECTION_OCCLUDED_V1",
        "entity_ids": [str(e.entity_id) for e in entities],
        "window_direction": [float(x) for x in window],
        "panel_half_extent_m": [half_len, p["half_width_m"], opts.thickness_m],
        "panel_span_deg": math.degrees(span),
        "panels": panels,
        "posts": posts,
        "sampled": p,
        "source_kind": "SYNTHETIC_ONLY",
    }
