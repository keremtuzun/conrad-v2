"""Finite-volume advection-diffusion for scalar fields (ch33 Twin2E freeze). TRUTH PLANE.

Flux form, first-order upwind advection plus central diffusion, explicit Euler sub-steps.
The sub-step satisfies the positivity (CFL-type) bound

    dt * sum_a (2 max|u_a| / dx_a + 2 K_a / dx_a^2) <= cfl_max <= 1,

which guarantees a non-negative update for non-negative inputs and, with CLOSED boundaries,
exact conservation of sum(c) * cell_volume (face fluxes cancel pairwise).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from conrad.twins.twin2e.config import BoundaryCondition
from conrad.twins.twin2e.grid import FieldGrid


class CFLViolationError(RuntimeError):
    """The requested step cannot be integrated stably within the configured sub-step budget."""


@dataclass(frozen=True)
class StepReport:
    substeps: int
    substep_dt_s: float
    cfl_number: float


def cfl_number(
    velocity: np.ndarray, diffusivity: tuple[float, float, float], grid: FieldGrid, dt_s: float
) -> float:
    """Positivity number of one explicit step of ``dt_s``; must be <= 1 for a stable update."""
    if velocity.shape != (3, *grid.shape):
        raise ValueError(f"velocity must have shape (3, {grid.shape}), got {velocity.shape}")
    total = 0.0
    for a in range(3):
        dx = grid.spacing_m[a]
        umax = float(np.max(np.abs(velocity[a]))) if velocity[a].size else 0.0
        total += 2.0 * umax / dx + 2.0 * diffusivity[a] / dx**2
    return dt_s * total


def plan_substeps(
    velocity: np.ndarray,
    diffusivity: tuple[float, float, float],
    grid: FieldGrid,
    dt_s: float,
    cfl_max: float,
    max_substeps: int,
) -> StepReport:
    if dt_s <= 0 or not math.isfinite(dt_s):
        raise ValueError(f"dt_s must be positive and finite, got {dt_s}")
    if any(k < 0 for k in diffusivity):
        raise ValueError("diffusivity must be non-negative")
    full = cfl_number(velocity, diffusivity, grid, dt_s)
    n = max(1, math.ceil(full / cfl_max))
    if n > max_substeps:
        raise CFLViolationError(
            f"grid {grid.name}: CFL {full:.3g} for dt={dt_s}s needs {n} sub-steps > max {max_substeps}"
        )
    return StepReport(substeps=n, substep_dt_s=dt_s / n, cfl_number=full / n)


def _face_flux(
    c: np.ndarray,
    u: np.ndarray,
    k: float,
    dx: float,
    bc: BoundaryCondition,
    ghost: np.ndarray | None,
) -> np.ndarray:
    """Flux through the n+1 faces along axis 0 (arrays already moved so the axis is first)."""
    n = c.shape[0]
    flux = np.zeros((n + 1, *c.shape[1:]), dtype=np.float64)
    uf = 0.5 * (u[:-1] + u[1:])
    flux[1:n] = np.maximum(uf, 0.0) * c[:-1] + np.minimum(uf, 0.0) * c[1:] - k * (c[1:] - c[:-1]) / dx
    if bc is BoundaryCondition.CLOSED:
        return flux
    if bc is BoundaryCondition.PERIODIC:
        ub = 0.5 * (u[-1] + u[0])
        fb = np.maximum(ub, 0.0) * c[-1] + np.minimum(ub, 0.0) * c[0] - k * (c[0] - c[-1]) / dx
        flux[0] = fb
        flux[n] = fb
        return flux
    # OPEN: upwind advection, inflow carries the ghost (exterior) value; no diffusive boundary flux.
    g_lo = c[0] if ghost is None else ghost[0]
    g_hi = c[-1] if ghost is None else ghost[-1]
    flux[0] = np.maximum(u[0], 0.0) * g_lo + np.minimum(u[0], 0.0) * c[0]
    flux[n] = np.maximum(u[-1], 0.0) * c[-1] + np.minimum(u[-1], 0.0) * g_hi
    return flux


def advect_diffuse(
    c: np.ndarray,
    velocity: np.ndarray,
    diffusivity: tuple[float, float, float],
    grid: FieldGrid,
    dt_s: float,
    bc: BoundaryCondition,
    cfl_max: float = 0.8,
    max_substeps: int = 4000,
    ghost: np.ndarray | None = None,
) -> tuple[np.ndarray, StepReport]:
    """Integrate dc/dt = -div(u c) + div(K grad c) over ``dt_s``. Returns a new array."""
    if c.shape != grid.shape:
        raise ValueError(f"field shape {c.shape} does not match grid {grid.shape}")
    if ghost is not None and ghost.shape != grid.shape:
        raise ValueError("ghost field must have the grid shape")
    report = plan_substeps(velocity, diffusivity, grid, dt_s, cfl_max, max_substeps)
    out = np.array(c, dtype=np.float64, copy=True)
    h = report.substep_dt_s
    for _ in range(report.substeps):
        tend = np.zeros_like(out)
        for a in range(3):
            ca = np.moveaxis(out, a, 0)
            ua = np.moveaxis(velocity[a], a, 0)
            ga = None if ghost is None else np.moveaxis(ghost, a, 0)
            f = _face_flux(ca, ua, diffusivity[a], grid.spacing_m[a], bc, ga)
            tend += np.moveaxis(-(f[1:] - f[:-1]) / grid.spacing_m[a], 0, a)
        out = out + h * tend
    return out, report


def total_mass(c: np.ndarray, grid: FieldGrid) -> float:
    return float(np.sum(c) * grid.cell_volume_m3)
