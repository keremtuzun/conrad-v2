"""Builds MCBR's belief-side predictive model for one information need of the mission. DEPLOYMENT PLANE.

Inputs are belief-plane only (``conrad.active.surface_predictive`` explains the model):

* Model2T component belief of the need's target: surface coverage cells of the SURVEYED design geometry, the
  covered set, the worst-indication locus, the posterior of each quantity and the population prior;
* Model2T's declared sensor datasheet (``SensorCharacteristics``) for per-reading noise, partial-view scatter,
  persistent bias and probability of detection;
* Model2S: ray-cast clear-line probability through the belief map (sensor range / FOV / occluders) and which
  water-side surface probes the geometric payload already observed;
* motion cost stays in the shared ``navigation_cost`` of the request (belief map + runtime travel model).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from conrad.active.candidates import SensorOption, look_at
from conrad.active.production import load_frozen
from conrad.active.surface_predictive import QuantityChannel, SurfaceCellPredictive, SurfacePredictiveConfig
from conrad.domains.spatial.model import Model2S
from conrad.domains.technical import Model2T
from conrad.domains.technical.claims import wall_m
from conrad.domains.technical.config import SensorCharacteristics
from conrad.domains.technical.coverage import SurfaceGeometry
from conrad.domains.technical.measurement import scatter_sigmas, view_mixture
from conrad.domains.technical.registry import CORROSION_DEPTH, CRACK_LENGTH
from conrad.domains.technical.state import ComponentBelief
from conrad.schemas.belief import BeliefMessage
from conrad.schemas.frames import WORLD, Pose, SpatialSupport
from conrad.schemas.world import SensorSpec

QUANTITIES = (CORROSION_DEPTH, CRACK_LENGTH)
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(16)
_GH_W = _GH_W / _GH_W.sum()


def capsule_cells(g: SurfaceGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres and outward normals, indexed exactly as ``SurfaceGeometry.cell_of`` indexes cells."""
    a, b = np.asarray(g.p0, dtype=np.float64), np.asarray(g.p1, dtype=np.float64)
    axis = b - a
    length = float(np.linalg.norm(axis))
    d = axis / max(length, 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = ref - (ref @ d) * d
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    pts, nrm = [], []
    for i in range(g.n_along):
        for s in range(g.sectors):
            ang = (s + 0.5) * 2.0 * math.pi / g.sectors
            n = math.cos(ang) * u + math.sin(ang) * v
            pts.append(a + (i + 0.5) / g.n_along * length * d + g.radius_m * n)
            nrm.append(n)
    return np.asarray(pts), np.asarray(nrm)


def face_cells(g: SurfaceGeometry) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(g.p0, dtype=np.float64)
    ext = np.asarray(g.half_extent_m) if g.shape == "BOX" else np.full(3, g.radius_m)
    pts, nrm = [], []
    for k in range(3):
        for sign in (1.0, -1.0):
            n = np.zeros(3)
            n[k] = sign
            pts.append(c + n * ext[k])
            nrm.append(n)
    return np.asarray(pts), np.asarray(nrm)


def detection_probability(mean: float, var: float, aleatoric: float, sc: SensorCharacteristics) -> float:
    """Declared POD averaged over the (Gaussian, truncated at 0) belief of the crack length."""
    ls = mean + math.sqrt(max(var, 0.0)) * _GH_X
    a50 = sc.crack_pod_a50_m * (1.0 + sc.pod_ua_gain * aleatoric)
    ok = ls > 0.0
    z = (np.log(np.maximum(ls, 1e-9)) - math.log(a50)) / sc.crack_pod_log_width
    pod = np.where(ok, 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0))), 0.0)
    return float(np.sum(_GH_W * pod))


def look_rel_sd(
    q: str, level: float, aleatoric: float, reliability: float, sc: SensorCharacteristics
) -> tuple[float, float]:
    """(independent relative sd of one reading incl. partial-view scatter, persistent relative bias sd)."""
    ind, sys_ = scatter_sigmas(q, level, aleatoric, sc)
    f, w = view_mixture(q, reliability, sc)
    ef = float(np.sum(w * f))
    rel_f = math.sqrt(max(float(np.sum(w * f * f)) - ef * ef, 0.0)) / max(ef, 1e-9)
    return math.hypot(ind, rel_f), sys_


def worst_band_probability(q: str, mean: float, var: float, b: ComponentBelief) -> float:
    """P(quantity in the component's worst condition band) under the Gaussian belief (Model2T condition rule:
    severity = level / scale, worst band from ``ConditionConfig.bands[2]``)."""
    cfg = b.condition_cfg
    scale = wall_m(b, cfg) if q == CORROSION_DEPTH else cfg.crack_critical_m
    level = cfg.bands[2] * scale
    sd = math.sqrt(max(var, 1e-24))
    return 0.5 * math.erfc((level - mean) / (sd * math.sqrt(2.0)))


class MissionPredictive:
    """Callable hook of ``Deliberation.plan``: (beliefs of the need) -> SurfaceCellPredictive | None."""

    def __init__(
        self,
        m2t: Model2T,
        m2s: Model2S | None,
        boresight_sensor: SensorSpec,
        unknown_block_probability: float,
        cfg: SurfacePredictiveConfig | None = None,
        track: Callable[[], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        self.m2t, self.m2s, self.sensor = m2t, m2s, boresight_sensor
        self.p_block = unknown_block_probability
        self.cfg = cfg or SurfacePredictiveConfig()
        self.track = track  # the runtime's own ESTIMATED track rows [t_s, x, y, z] (belief side)

    def _target(self, beliefs: Sequence[BeliefMessage]) -> ComponentBelief | None:
        comps = self.m2t.beliefs
        for m in beliefs:
            b = comps.get(m.world_entity_id) if m.world_entity_id is not None else None
            if b is not None and b.geometry is not None:
                return b
        return None

    def _cells(self, g: SurfaceGeometry) -> tuple[np.ndarray, np.ndarray]:
        return capsule_cells(g) if g.shape == "CAPSULE" else face_cells(g)

    def _cell_prior(
        self, b: ComponentBelief, pts: np.ndarray, nrm: np.ndarray, probes: np.ndarray
    ) -> np.ndarray:
        n = len(probes)
        prior = np.ones(n)
        covered = [c for c in b.covered if 0 <= c < n]
        if covered and self.cfg.footprint_discount > 0.0 and b.geometry is not None:
            # Declared reading footprint: a reading whose measured point fell in cell k looked at the cells around
            # k (surface normal within the half angle, axially close). Those readings did not put the component in
            # its worst band (the need is still open), so an undetected worst case is less likely there.
            axis = np.asarray(b.geometry.p1) - np.asarray(b.geometry.p0)
            d = axis / max(float(np.linalg.norm(axis)), 1e-12)
            cos_lim = math.cos(math.radians(self.cfg.footprint_half_angle_deg))
            t = pts @ d
            viewed = np.zeros(n, dtype=bool)
            for k in covered:
                viewed |= (nrm @ nrm[k] >= cos_lim - 1e-9) & (np.abs(t - t[k]) <= self.cfg.footprint_axial_m)
            prior[viewed] *= 1.0 - self.cfg.footprint_discount
        if self.cfg.track_discount > 0.0 and self.track is not None:
            viewed_w = self._track_views(pts, nrm, probes)
            prior *= 1.0 - self.cfg.track_discount * viewed_w
        prior[covered] = 0.0
        if self.m2s is not None and self.cfg.seen_surface_discount > 0.0:
            seen = np.asarray([s.value == "OBSERVED" for s in self.m2s.occupancy_status(probes)], dtype=bool)
            prior[seen] *= 1.0 - self.cfg.seen_surface_discount
        total = float(prior.sum())
        if total <= 0.0:
            prior = np.ones(n)
            prior[[c for c in b.covered if 0 <= c < n]] = 0.0
            total = float(prior.sum())
        return prior / total if total > 0.0 else np.zeros(n)

    def _track_views(self, pts: np.ndarray, nrm: np.ndarray, probes: np.ndarray) -> np.ndarray:
        """Per cell: max over the robot's own ESTIMATED past positions (runtime track, belief side) of the predicted
        observation weight, i.e. how well the payload has probably already looked at that cell. Positions are
        thinned to ``track_max_positions`` within sensor range of the component."""
        assert self.track is not None
        track = np.asarray([row[1:4] for row in self.track()], dtype=np.float64).reshape(-1, 3)
        centre = pts.mean(axis=0)
        reach = float(self.sensor.parameters.get("max_range_m", 4.0)) + float(np.ptp(pts, axis=0).max()) / 2.0
        near = track[np.linalg.norm(track - centre, axis=1) <= reach]
        if len(near) == 0:
            return np.zeros(len(pts))
        k = self.cfg.track_max_positions
        near = near[np.linspace(0, len(near) - 1, min(k, len(near))).round().astype(int)]
        fn = CellWeightFn(self, pts, nrm, probes)
        opt = SensorOption(
            sensor_id=self.sensor.sensor_id,
            modality=self.sensor.modality,
            min_range_m=float(self.sensor.parameters.get("min_range_m", 0.3)),
            max_range_m=float(self.sensor.parameters.get("max_range_m", 4.0)),
        )
        out = np.zeros(len(pts))
        c = tuple(float(x) for x in centre)
        for p in near:
            pose = look_at((float(p[0]), float(p[1]), float(p[2])), (c[0], c[1], c[2]), WORLD)
            out = np.maximum(out, fn(pose, opt))
        return out

    def _weights(self, pts: np.ndarray, nrm: np.ndarray, probes: np.ndarray) -> CellWeightFn:
        return CellWeightFn(self, pts, nrm, probes)

    def _channels(self, b: ComponentBelief) -> tuple[QuantityChannel, ...]:
        sc = self.m2t.config.sensor
        rel = self.cfg.nominal_reliability
        ua = min(1.0, b.ua)
        out = []
        for q in QUANTITIES:
            if q not in b.valid:
                continue
            pr = b.prior[q]
            level = max(pr.level + math.sqrt(max(pr.level_var, 0.0)), 1e-6)  # a defect is a worst case
            ind, sys_ = look_rel_sd(q, level, ua, rel, sc)
            look = level * math.hypot(ind, sys_)
            det = detection_probability(pr.level, pr.level_var, ua, sc) if q == CRACK_LENGTH else 1.0
            kw: dict[str, float | int | None] = {}
            est = b.estimates[q]
            if (
                self.cfg.read_channel
                and est.known
                and q in b.locus
                and b.geometry is not None
                and worst_band_probability(q, est.level, est.level_var, b)
                >= self.cfg.read_min_band_probability
            ):
                lvl = max(abs(est.level), 1e-6)
                r_ind, r_sys = look_rel_sd(q, lvl, ua, rel, sc)
                sensors = max(1, sum(1 for _, qq in b.bias_counts if qq == q))
                kw = {
                    "read_mean": est.level,
                    "read_var": max(est.level_var, 1e-18),
                    "look_std_read": lvl * r_ind,
                    "read_var_floor": (r_sys * lvl) ** 2 / sensors,
                    "locus_cell": b.geometry.cell_of(b.locus[q][0]),
                }
            out.append(
                QuantityChannel(
                    quantity=q,
                    unread_mean=pr.level,
                    unread_var=max(pr.level_var, 1e-18),
                    look_std_unread=look,
                    detection_probability=det,
                    **kw,  # type: ignore[arg-type]
                )
            )
        return tuple(out)

    def __call__(self, beliefs: Sequence[BeliefMessage]) -> SurfaceCellPredictive | None:
        if not self.cfg.enabled:
            return None
        b = self._target(beliefs)
        if b is None or b.geometry is None:
            return None
        pts, nrm = self._cells(b.geometry)
        probes = pts + self.cfg.probe_offset_m * nrm
        channels = self._channels(b)
        if not channels:
            return None
        return SurfaceCellPredictive(
            cell_prior=self._cell_prior(b, pts, nrm, probes),
            channels=channels,
            cell_weights=self._weights(pts, nrm, probes),
            epistemic=min(1.0, b.ue),
        )


def production_predictive_config() -> SurfacePredictiveConfig | None:
    """The frozen production planner's mission predictive model (``mission_predictive`` section), if any."""
    raw = load_frozen().get("mission_predictive")
    return None if raw is None else SurfacePredictiveConfig(**raw["config"])


def mission_predictive_provider(
    m2t: Model2T,
    m2s: Model2S | None,
    boresight_sensor: SensorSpec,
    unknown_block_probability: float,
    track: Callable[[], Sequence[Sequence[float]]] | None = None,
) -> MissionPredictive | None:
    cfg = production_predictive_config()
    if cfg is None or not cfg.enabled:
        return None
    return MissionPredictive(m2t, m2s, boresight_sensor, unknown_block_probability, cfg, track)


class CellWeightFn:
    """Per-cell observation weight from a candidate SENSOR-boresight pose (Model2S clear-ray x incidence)."""

    def __init__(
        self, owner: MissionPredictive, pts: np.ndarray, nrm: np.ndarray, probes: np.ndarray
    ) -> None:
        self.o, self.pts, self.nrm, self.probes = owner, pts, nrm, probes

    def __call__(self, pose: Pose, sensor: SensorOption) -> np.ndarray:
        origin = np.asarray(pose.position_m, dtype=np.float64)
        ray = origin[None, :] - self.pts
        dist = np.linalg.norm(ray, axis=1)
        cos = np.einsum("ij,ij->i", self.nrm, ray) / np.maximum(dist, 1e-9)
        w = np.zeros(len(self.pts))
        ok = (cos > 0.0) & (dist <= sensor.max_range_m + 1e-9)
        m2s = self.o.m2s
        for i in np.nonzero(ok)[0]:
            if m2s is None:
                clear = 1.0
            else:
                cell = SpatialSupport(frame_id=WORLD, center_m=tuple(float(x) for x in self.probes[i]))
                clear = float(m2s.predicted_visibility(pose, cell, self.o.sensor, self.o.p_block).mean)
            w[i] = clear * float(cos[i]) ** self.o.cfg.incidence_power
        return w
