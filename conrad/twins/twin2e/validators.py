"""Sanity validators for Twin2E: positivity, bounds, CFL, conservation, determinism.

These are software sanity checks, not scientific validation against real data (that is OPEN).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from conrad.schemas.base import digest_of
from conrad.twins.twin2e.config import BoundaryCondition
from conrad.twins.twin2e.grid import FieldGrid
from conrad.twins.twin2e.solver import advect_diffuse, total_mass

if TYPE_CHECKING:
    from conrad.twins.twin2e.twin import Twin2E


def field_problems(twin: Twin2E) -> list[str]:
    out = []
    for st in (twin.fields.regional, twin.fields.local):
        for name in ("temperature", "turbidity", "current", "light"):
            if not np.all(np.isfinite(st.get(name))):
                out.append(f"{st.grid.name}/{name} has non-finite values")
        for name in ("turbidity", "light"):
            if float(np.min(st.get(name))) < 0:
                out.append(f"{st.grid.name}/{name} is negative")
    return out


def entity_problems(twin: Twin2E) -> list[str]:
    out = []
    for eid, e in twin.entities.items():
        for name, v in (("cover", e.cover), ("condition", e.condition)):
            if not (0.0 <= v <= 1.0) or not np.isfinite(v):
                out.append(f"{eid} {name}={v} outside [0,1]")
        if e.thermal_stress < 0:
            out.append(f"{eid} negative thermal stress")
    return out


def cfl_problems(twin: Twin2E) -> list[str]:
    lim = twin.cfg.solver.cfl_max
    return [
        f"{k} CFL {r.cfl_number:.3g} > {lim}"
        for k, r in twin.fields.last_reports.items()
        if r.cfl_number > lim + 1e-12
    ]


def validate_twin(twin: Twin2E) -> list[str]:
    return field_problems(twin) + entity_problems(twin) + cfl_problems(twin)


def closed_box_mass_drift(
    grid: FieldGrid,
    c0: np.ndarray,
    velocity: np.ndarray,
    diffusivity: tuple[float, float, float],
    dt_s: float,
    steps: int,
) -> float:
    """Relative change of total tracer mass under CLOSED boundaries (should be ~1e-12)."""
    m0 = total_mass(c0, grid)
    c = c0
    for _ in range(steps):
        c, _ = advect_diffuse(c, velocity, diffusivity, grid, dt_s, BoundaryCondition.CLOSED)
    return abs(total_mass(c, grid) - m0) / max(abs(m0), 1e-300)


def state_digest(twin: Twin2E) -> str:
    """Digest of Z and Phi (rounded to 12 significant digits) for determinism checks."""
    z = {str(k): v.as_truth() for k, v in sorted(twin.entities.items(), key=lambda kv: str(kv[0]))}
    phi = {
        f"{st.grid.name}/{n}": digest_of(np.round(st.get(n), 12).tolist())
        for st in (twin.fields.regional, twin.fields.local)
        for n in ("temperature", "turbidity", "current", "light")
    }
    return digest_of({"t": twin.t_s, "Z": z, "Phi": phi})
