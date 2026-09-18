"""Phi: environmental field state and its MEIFE dynamics (ch12-13, ch33). TRUTH PLANE.

Fields (each justified by a consumer): temperature (ecological thermal response), turbidity
(light attenuation + sensor visibility), current (robot disturbance, resuspension, larval supply),
light (photosynthetic/fouling growth). Current is a prescribed stochastic profile (not advected);
temperature and turbidity are advected-diffused; light is diagnostic (Beer-Lambert).
This is NOT an ocean model: no density, no momentum equation, flat seabed at the grid bottom.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from conrad.twins.twin2e.config import BoundaryCondition, Twin2EConfig
from conrad.twins.twin2e.grid import FieldGrid
from conrad.twins.twin2e.solver import StepReport, advect_diffuse

FIELD_UNITS: dict[str, str] = {
    "temperature": "degC",
    "turbidity": "NTU",
    "current": "m s-1",
    "light": "W m-2",
}
SCALAR_FIELDS = ("temperature", "turbidity", "light")


@dataclass
class FieldState:
    grid: FieldGrid
    temperature: np.ndarray
    turbidity: np.ndarray
    current: np.ndarray
    light: np.ndarray

    def get(self, name: str) -> np.ndarray:
        if name not in FIELD_UNITS:
            raise KeyError(f"unknown field {name!r}")
        arr: np.ndarray = getattr(self, name)
        return arr

    def copy(self) -> FieldState:
        return FieldState(
            self.grid, self.temperature.copy(), self.turbidity.copy(), self.current.copy(), self.light.copy()
        )


@dataclass
class FieldParams:
    surface_temperature_c: float
    lapse_c_per_m: float
    turbidity_bg_ntu: float
    current_base_m_s: np.ndarray
    surface_light_w_m2: float
    kd_water_per_m: float
    k_turbidity: float
    kh_m2_s: float
    kv_m2_s: float
    resuspension_critical_m_s: float
    resuspension_rate: float
    settling_time_s: float
    temperature_relaxation_s: float
    ou_sigma_m_s: float
    ou_time_s: float
    filtration_m3_s_per_m2: float
    start_time_of_day_h: float
    surface_z_m: float
    anomaly_relaxation_s: float = 6 * 3600.0


@dataclass
class Forcing:
    """Active disturbance forcing on Phi. Each item keeps its own end time (seconds)."""

    temperature_anomalies: list[dict[str, Any]] = field(default_factory=list)
    light_reductions: list[dict[str, Any]] = field(default_factory=list)


def region_mask(grid: FieldGrid, center: Any, radius_m: float | None) -> np.ndarray:
    if center is None or radius_m is None:
        return np.ones(grid.shape, dtype=bool)
    gx, gy, gz = grid.centres()
    c = [float(v) for v in center]
    return bool_mask((gx - c[0]) ** 2 + (gy - c[1]) ** 2 + (gz - c[2]) ** 2 <= float(radius_m) ** 2)


def bool_mask(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=bool)


class FieldEngine:
    """Regional + local Phi with downward nudging (regional -> local) and cautious upward coupling."""

    def __init__(self, cfg: Twin2EConfig, params: FieldParams) -> None:
        self.cfg = cfg
        self.p = params
        self.regional_grid = FieldGrid.from_config("regional", cfg.regional_grid)
        self.local_grid = FieldGrid.from_config("local", cfg.local_grid)
        if self.regional_grid.frame_id != self.local_grid.frame_id:
            raise ValueError("regional and local grids must share a frame")
        if not self.regional_grid.contains_grid(self.local_grid):
            raise ValueError("local grid must lie inside the regional grid")
        self.ou = np.zeros(2)
        self.forcing = Forcing()
        self.regional = self._background(self.regional_grid, 0.0)
        self.local = self._background(self.local_grid, 0.0)
        self._local_bg = self.local.copy()
        self._regional_bg = self.regional.copy()
        self._owner = self.regional_grid.owning_cells(self.local_grid)
        self.last_reports: dict[str, StepReport] = {}

    # ---------------------------------------------------------------- construction
    def _t_bg(self, grid: FieldGrid) -> np.ndarray:
        _, _, gz = grid.centres()
        return self.p.surface_temperature_c + self.p.lapse_c_per_m * (gz - self.p.surface_z_m)

    def _current(self, grid: FieldGrid) -> np.ndarray:
        _, _, gz = grid.centres()
        bottom = grid.origin_m[2]
        height = max(self.p.surface_z_m - bottom, 1e-6)
        profile = np.clip(np.clip((gz - bottom) / height, 0.0, None) ** (1.0 / 7.0), 0.1, 1.0)
        vec = np.array(self.p.current_base_m_s, dtype=np.float64)
        vec[:2] = vec[:2] + self.ou
        return np.stack([vec[a] * profile for a in range(3)], axis=0)

    def _background(self, grid: FieldGrid, t_s: float) -> FieldState:
        turb = np.full(grid.shape, self.p.turbidity_bg_ntu, dtype=np.float64)
        st = FieldState(grid, self._t_bg(grid), turb, self._current(grid), np.zeros(grid.shape))
        st.light = self.light_field(st, t_s)
        return st

    # ---------------------------------------------------------------- diagnostics
    def surface_irradiance(self, t_s: float) -> float:
        tod = (self.p.start_time_of_day_h + t_s / 3600.0) % 24.0
        diel = max(0.0, math.sin(math.pi * (tod - 6.0) / 12.0))
        factor = 1.0
        for item in self.forcing.light_reductions:
            if t_s < item["end_s"]:
                factor *= float(item["factor"])
        return self.p.surface_light_w_m2 * diel * factor

    def light_field(self, st: FieldState, t_s: float) -> np.ndarray:
        """Beer-Lambert: I = I0 exp(-(kd * depth + k_t * integral of turbidity above))."""
        g = st.grid
        dz = g.spacing_m[2]
        _, _, gz = g.centres()
        depth = np.clip(self.p.surface_z_m - gz, 0.0, None)
        top_gap = max(self.p.surface_z_m - g.extent_max_m[2], 0.0)
        above = np.cumsum(st.turbidity[:, :, ::-1], axis=2)[:, :, ::-1] - 0.5 * st.turbidity
        optical = self.p.kd_water_per_m * depth + self.p.k_turbidity * (
            above * dz + top_gap * st.turbidity[:, :, -1:]
        )
        return np.asarray(self.surface_irradiance(t_s) * np.exp(-optical), dtype=np.float64)

    def state_for(self, pts: np.ndarray) -> FieldState:
        return self.local if all(self.local_grid.contains(p) for p in np.atleast_2d(pts)) else self.regional

    def sample(self, name: str, pts: np.ndarray) -> np.ndarray:
        """Values at points (N,3); current returns (N,3). Local grid wins where it covers the point."""
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        out: list[Any] = []
        for p in pts:
            st = self.state_for(p)
            if name == "current":
                out.append([float(st.grid.sample(st.current[a], p[None])[0]) for a in range(3)])
            else:
                out.append(float(st.grid.sample(st.get(name), p[None])[0]))
        return np.asarray(out, dtype=np.float64)

    # ---------------------------------------------------------------- dynamics
    def step(
        self,
        dt_s: float,
        t_end_s: float,
        rng: dict[str, np.random.Generator],
        filtration: list[tuple[np.ndarray, float]],
        stochastic: bool,
        multi_scale: bool,
        entity_to_field: bool,
    ) -> None:
        if stochastic and self.p.ou_sigma_m_s > 0:
            a = math.exp(-dt_s / self.p.ou_time_s)
            self.ou = self.ou * a + self.p.ou_sigma_m_s * math.sqrt(1 - a * a) * rng[
                "current"
            ].standard_normal(2)
        t_noise = rng["temperature"].standard_normal() if stochastic else 0.0
        c_noise = rng["turbidity"].standard_normal() if stochastic else 0.0
        grids = [("local", self.local, self._local_bg)]
        if multi_scale:
            grids.insert(0, ("regional", self.regional, self._regional_bg))
        for name, st, bg in grids:
            st.current = self._current(st.grid)
            ghost_t = ghost_c = None
            if name == "local":
                ghost_t = (
                    self.local_grid.resample_from(self.regional_grid, self.regional.temperature)
                    if multi_scale
                    else self._temperature_target(st, bg, t_end_s)[0]
                )
                ghost_c = (
                    self.local_grid.resample_from(self.regional_grid, self.regional.turbidity)
                    if multi_scale
                    else bg.turbidity
                )
            else:
                # boundary climatology includes active (e.g. regional-scale) anomalies
                ghost_t, ghost_c = self._temperature_target(st, bg, t_end_s)[0], bg.turbidity
            self._transport(name, st, dt_s, ghost_t, ghost_c)
            self._temperature_sources(st, bg, dt_s, t_end_s, t_noise)
            self._turbidity_sources(st, dt_s, c_noise)
            if name == "local" and multi_scale:
                w = 1.0 - math.exp(-dt_s / self.cfg.solver.nudging_time_s)
                st.temperature += w * (ghost_t - st.temperature)
                st.turbidity += w * (ghost_c - st.turbidity)
        if entity_to_field:
            self._filtration(filtration, dt_s)
        if multi_scale:
            self._upward(dt_s)
        for st in (self.regional, self.local):
            np.clip(st.turbidity, 0.0, None, out=st.turbidity)
            st.light = self.light_field(st, t_end_s)
        self.forcing.temperature_anomalies = [
            a for a in self.forcing.temperature_anomalies if a["end_s"] > t_end_s
        ]

    def _transport(self, name: str, st: FieldState, dt_s: float, gt: np.ndarray, gc: np.ndarray) -> None:
        k = (self.p.kh_m2_s, self.p.kh_m2_s, self.p.kv_m2_s)
        bc: BoundaryCondition = self.cfg.solver.boundary
        s = self.cfg.solver
        st.temperature, rep = advect_diffuse(
            st.temperature, st.current, k, st.grid, dt_s, bc, s.cfl_max, s.max_substeps, gt
        )
        st.turbidity, _ = advect_diffuse(
            st.turbidity, st.current, k, st.grid, dt_s, bc, s.cfl_max, s.max_substeps, gc
        )
        self.last_reports[name] = rep

    def _temperature_target(
        self, st: FieldState, bg: FieldState, t_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        target = bg.temperature.copy()
        tau = np.full(st.grid.shape, self.p.temperature_relaxation_s)
        for an in self.forcing.temperature_anomalies:
            if an["start_s"] <= t_s < an["end_s"]:
                m = region_mask(st.grid, an.get("center_m"), an.get("radius_m"))
                target[m] += float(an["delta_c"])
                tau[m] = np.minimum(tau[m], self.p.anomaly_relaxation_s)
        return target, tau

    def _temperature_sources(
        self, st: FieldState, bg: FieldState, dt_s: float, t_s: float, noise: float
    ) -> None:
        target, tau = self._temperature_target(st, bg, t_s)
        st.temperature = target + (st.temperature - target) * np.exp(-dt_s / tau)
        st.temperature += self.cfg.solver.field_noise_fraction * math.sqrt(dt_s / 3600.0) * noise * 10.0

    def _turbidity_sources(self, st: FieldState, dt_s: float, noise: float) -> None:
        speed = np.linalg.norm(st.current[:, :, :, 0], axis=0)
        excess = np.clip(speed - self.p.resuspension_critical_m_s, 0.0, None)
        st.turbidity[:, :, 0] += self.p.resuspension_rate * excess**2 * dt_s
        bg = self.p.turbidity_bg_ntu
        st.turbidity = bg + (st.turbidity - bg) * math.exp(-dt_s / self.p.settling_time_s)
        sig = self.cfg.solver.field_noise_fraction * math.sqrt(dt_s / 3600.0) * 10.0
        st.turbidity *= math.exp(sig * noise - 0.5 * sig * sig)

    def _filtration(self, sources: list[tuple[np.ndarray, float]], dt_s: float) -> None:
        for pos, covered_area_m2 in sources:
            st = self.state_for(pos)
            idx = st.grid.cell_index(pos)
            rate = self.p.filtration_m3_s_per_m2 * covered_area_m2 / st.grid.cell_volume_m3
            st.turbidity[idx] *= math.exp(-rate * dt_s)

    def _upward(self, dt_s: float) -> None:
        """Local -> regional: regional cells under the local domain relax to the local block mean."""
        w = 1.0 - math.exp(-dt_s / self.cfg.solver.upward_nudging_time_s)
        n = int(np.prod(self.regional_grid.shape))
        counts = np.bincount(self._owner, minlength=n)
        hit = counts > 0
        for name in ("temperature", "turbidity"):
            reg = self.regional.get(name).reshape(-1)
            sums = np.bincount(self._owner, weights=self.local.get(name).reshape(-1), minlength=n)
            reg[hit] += w * (sums[hit] / counts[hit] - reg[hit])
