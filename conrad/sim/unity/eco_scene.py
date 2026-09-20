"""Twin2E -> Unity scene extensions (gate I6), with a measured conversion report. TRUTH PLANE (simulator setup).

On the Python kernel path Twin2E drives the payload sensors in two ways (``conrad.sim.mission.sensing``):

* the structural inspection readings get ``turbidity`` and ``biofouling_cover`` degradation from
  ``Twin2E.observability_modifiers`` at the observed surface point;
* the environmental probe and the ecological survey are Twin2E observations.

Both are rendered on the Python side at the Unity TRUE pose on the Unity path too (``MissionSensorSuite``), exactly as on
the kernel. What Unity itself renders is converted here and loaded with ``CONFIGURE_SCENE``:

==================  =====================================================================================================
Twin2E truth        Unity
==================  =====================================================================================================
turbidity field     ``optics_grid``: beam attenuation c = c_clear + c_ntu * turbidity (Twin2E ``ObservationConfig``
                    constants) on a regular grid of cell centres; the RGB camera blends each pixel toward the water colour
                    with exp(-integral of c along the pixel ray). Refreshed every ``update_period_s`` (the field evolves).
biofouling cover    ``fouling_cover`` on the box/capsule primitives of each structure (mean Twin2E cover over the
                    structure's surface samples): a visual colour only, colliders unchanged (the kernel does not change
                    geometry either).
==================  =====================================================================================================

The range imager and the sonar are NOT attenuated, because the kernel's Twin2S geometric channel is not either.

Every approximation is measured, never assumed (like ``twin_scene`` for geometry): attenuation and transmission error of
the grid against Twin2E at random points and rays, and the cover error of the per-structure constant.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import Field

from conrad.schemas.base import ConradModel
from conrad.schemas.frames import WORLD, FramedPoint
from conrad.twins.twin2e import Twin2E
from conrad.twins.twin2s.terrain import HeightfieldTerrain
from conrad.twins.twin2s.visibility import sample_surface
from conrad.twins.twin2s.world import SpatialWorld

OPTICS_ID = "twin2e"


class EcoSceneOptions(ConradModel):
    """SYNTHETIC_ONLY conversion settings."""

    optics_spacing_m: float = Field(default=1.0, gt=0)
    optics_margin_m: float = Field(default=4.0, ge=0, description="beyond the Twin2S world bounds")
    update_period_s: float = Field(default=1.0, gt=0, description="optics grid refresh period (sim time)")
    error_points: int = Field(default=400, gt=10)
    error_rays: int = Field(default=100, gt=5)
    error_ray_length_m: float = Field(default=4.0, gt=0, description="about the structural sensor max range")
    fouling_samples_per_entity: int = Field(default=80, gt=5)
    water_rgb: tuple[float, float, float] = (0.05, 0.25, 0.30)
    fouling_rgb: tuple[float, float, float] = (0.25, 0.45, 0.12)
    error_seed: int = 0x2E5C


class OpticsGrid(ConradModel):
    origin_m: tuple[float, float, float]
    spacing_m: tuple[float, float, float]
    shape: tuple[int, int, int]
    attenuation_per_m: tuple[float, ...] = Field(description="index (i*ny + j)*nz + k")

    def values(self) -> np.ndarray:
        return np.asarray(self.attenuation_per_m, dtype=np.float64).reshape(self.shape)


class OpticsError(ConradModel):
    """Measured disagreement between Twin2E optics and the grid Unity renders with (same interpolation as C#)."""

    t_s: float
    points: int
    attenuation_max_abs_per_m: float
    attenuation_rms_per_m: float
    rays: int
    ray_length_m: float
    transmission_max_abs: float
    transmission_rms: float


class FoulingError(ConradModel):
    entities: int
    fouled_entities: int
    cover_max_abs: float = Field(
        description="max |Twin2E cover at a surface sample - the structure's constant|"
    )
    cover_mean_abs: float


class EcoConversionReport(ConradModel):
    grid_shape: tuple[int, int, int]
    spacing_m: float
    optics: OpticsError
    fouling: FoulingError
    notes: tuple[str, ...] = ()


# ------------------------------------------------------------------------------------------------ truth sampling
def attenuation_truth(t2e: Twin2E, pts: np.ndarray) -> np.ndarray:
    """Twin2E beam attenuation (1/m) at points: vectorised ``Twin2EFields.sample('turbidity')`` (local grid wins
    where it covers the point) mapped through Twin2E's own optical constants."""
    f = t2e.fields
    p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    lo, hi = np.asarray(f.local_grid.origin_m), np.asarray(f.local_grid.extent_max_m)
    local = np.all((p >= lo) & (p <= hi), axis=1)
    turb = np.empty(len(p))
    if local.any():
        turb[local] = f.local.grid.sample(f.local.turbidity, p[local])
    if (~local).any():
        turb[~local] = f.regional.grid.sample(f.regional.turbidity, p[~local])
    o = t2e.cfg.observation
    return np.asarray(o.beam_attenuation_clear_per_m + o.beam_attenuation_per_m_per_ntu * turb)


def trilinear(grid: OpticsGrid, pts: np.ndarray) -> np.ndarray:
    """Exactly the C# ``OpticsField.Attenuation``: trilinear on cell centres, clamped at the faces."""
    v = grid.values()
    p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    u = (p - np.asarray(grid.origin_m)) / np.asarray(grid.spacing_m)
    shape = np.asarray(grid.shape)
    u = np.clip(u, 0.0, shape - 1)
    i0 = np.floor(u).astype(int)
    i1 = np.minimum(i0 + 1, shape - 1)
    w = u - i0
    out = np.zeros(len(p))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = i1[:, 0] if dx else i0[:, 0]
                iy = i1[:, 1] if dy else i0[:, 1]
                iz = i1[:, 2] if dz else i0[:, 2]
                wt = (
                    (w[:, 0] if dx else 1 - w[:, 0])
                    * (w[:, 1] if dy else 1 - w[:, 1])
                    * (w[:, 2] if dz else 1 - w[:, 2])
                )
                out += wt * v[ix, iy, iz]
    return out


def optical_depth(
    field: Any, origin: np.ndarray, direction: np.ndarray, length: float, step_m: float
) -> float:
    """Midpoint-rule optical depth of ``field(pts) -> c`` along one ray (same rule as the C# camera)."""
    n = max(4, math.ceil(length / step_m))
    ds = length / n
    pts = origin[None, :] + ((np.arange(n) + 0.5) * ds)[:, None] * direction[None, :]
    return float(np.sum(field(pts)) * ds)


# ------------------------------------------------------------------------------------------------ conversion
def build_optics(t2e: Twin2E, world: SpatialWorld, o: EcoSceneOptions) -> OpticsGrid:
    lo = np.asarray(world.bounds_min, dtype=np.float64) - o.optics_margin_m
    hi = np.asarray(world.bounds_max, dtype=np.float64) + o.optics_margin_m
    s = o.optics_spacing_m
    shape = tuple(math.ceil((hi[a] - lo[a]) / s) + 1 for a in range(3))
    axes = [lo[a] + s * np.arange(shape[a]) for a in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    c = attenuation_truth(t2e, pts)
    return OpticsGrid(
        origin_m=(float(lo[0]), float(lo[1]), float(lo[2])),
        spacing_m=(s, s, s),
        shape=(shape[0], shape[1], shape[2]),
        attenuation_per_m=tuple(float(round(x, 6)) for x in c),
    )


def optics_primitive(grid: OpticsGrid, o: EcoSceneOptions) -> dict[str, Any]:
    return {
        "kind": "optics_grid",
        "id": OPTICS_ID,
        "origin_m": list(grid.origin_m),
        "spacing_m": list(grid.spacing_m),
        "shape": list(grid.shape),
        "beam_attenuation_per_m": list(grid.attenuation_per_m),
        "water_rgb": list(o.water_rgb),
    }


def optics_error(t2e: Twin2E, grid: OpticsGrid, world: SpatialWorld, o: EcoSceneOptions) -> OpticsError:
    rng = np.random.default_rng([o.error_seed, int(t2e.t_s * 1000)])
    lo, hi = np.asarray(world.bounds_min, dtype=np.float64), np.asarray(world.bounds_max, dtype=np.float64)
    pts = lo + (hi - lo) * rng.random((o.error_points, 3))
    d = trilinear(grid, pts) - attenuation_truth(t2e, pts)
    step = 0.5 * o.optics_spacing_m
    terr = []
    for _ in range(o.error_rays):
        a = lo + (hi - lo) * rng.random(3)
        u = rng.standard_normal(3)
        u /= float(np.linalg.norm(u))
        tg = math.exp(-optical_depth(lambda q: trilinear(grid, q), a, u, o.error_ray_length_m, step))
        tt = math.exp(-optical_depth(lambda q: attenuation_truth(t2e, q), a, u, o.error_ray_length_m, step))
        terr.append(tg - tt)
    te = np.asarray(terr)
    return OpticsError(
        t_s=float(t2e.t_s),
        points=len(pts),
        attenuation_max_abs_per_m=float(np.max(np.abs(d))),
        attenuation_rms_per_m=float(np.sqrt(np.mean(d**2))),
        rays=len(te),
        ray_length_m=o.error_ray_length_m,
        transmission_max_abs=float(np.max(np.abs(te))),
        transmission_rms=float(np.sqrt(np.mean(te**2))),
    )


def _cover(t2e: Twin2E, pts: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            float(
                t2e.observability_modifiers(
                    FramedPoint(frame_id=WORLD, xyz_m=(float(x), float(y), float(z)))
                )["biofouling_cover"]
            )
            for x, y, z in pts
        ]
    )


def fouling_covers(
    t2e: Twin2E, world: SpatialWorld, entity_indices: list[int], o: EcoSceneOptions
) -> tuple[dict[int, float], FoulingError]:
    """Per structure (Twin2S entity index): mean Twin2E cover over its surface samples, and the measured error."""
    rng = np.random.default_rng(o.error_seed)
    covers: dict[int, float] = {}
    errs: list[np.ndarray] = []
    for i in entity_indices:
        pts, _ = sample_surface(world, i, o.fouling_samples_per_entity, rng)
        if len(pts) == 0:
            continue
        c = _cover(t2e, pts)
        covers[i] = float(np.mean(c))
        errs.append(np.abs(c - covers[i]))
    e = np.concatenate(errs) if errs else np.zeros(1)
    return covers, FoulingError(
        entities=len(covers),
        fouled_entities=sum(1 for v in covers.values() if v > 0.0),
        cover_max_abs=float(np.max(e)),
        cover_mean_abs=float(np.mean(e)),
    )


def with_fouling(scene_json: dict[str, Any], covers: dict[int, float], o: EcoSceneOptions) -> dict[str, Any]:
    """Add ``fouling_cover`` to the box/capsule primitives of fouled structures (ids ``p<entity index>[-k]``)."""
    out = dict(scene_json)
    prims = []
    for p in scene_json["primitives"]:
        q = dict(p)
        pid = str(p.get("id", ""))
        if p.get("kind") in ("box", "capsule") and pid.startswith("p"):
            idx = int(pid[1:].split("-")[0])
            cover = covers.get(idx, 0.0)
            if cover > 0.0:
                q["fouling_cover"] = round(min(max(cover, 0.0), 1.0), 6)
                q["fouling_rgb"] = list(o.fouling_rgb)
        prims.append(q)
    out["primitives"] = prims
    return out


def eco_scene(
    t2e: Twin2E, world: SpatialWorld, scene_json: dict[str, Any], o: EcoSceneOptions
) -> tuple[dict[str, Any], OpticsGrid, EcoConversionReport]:
    """Scene JSON with fouling colours and the optics grid appended, plus the measured conversion report."""
    grid = build_optics(t2e, world, o)
    structures = [
        i
        for i, e in enumerate(world.entities)
        if e.active and not isinstance(e.primitive, HeightfieldTerrain)
    ]
    covers, ferr = fouling_covers(t2e, world, structures, o)
    scene = with_fouling(scene_json, covers, o)
    scene["primitives"] = [*scene["primitives"], optics_primitive(grid, o)]
    report = EcoConversionReport(
        grid_shape=grid.shape,
        spacing_m=o.optics_spacing_m,
        optics=optics_error(t2e, grid, world, o),
        fouling=ferr,
        notes=(
            "optics: Twin2E turbidity -> beam attenuation with Twin2E ObservationConfig constants; camera only",
            "fouling: one cover per structure (visual colour); colliders unchanged",
            "range imager and sonar are not attenuated (kernel parity)",
        ),
    )
    return scene, grid, report


def optics_update(t2e: Twin2E, world: SpatialWorld, o: EcoSceneOptions) -> tuple[dict[str, Any], OpticsGrid]:
    """Incremental CONFIGURE_SCENE body (replace=false) carrying only the refreshed optics grid."""
    grid = build_optics(t2e, world, o)
    return {"frame": WORLD, "replace": False, "primitives": [optics_primitive(grid, o)]}, grid
