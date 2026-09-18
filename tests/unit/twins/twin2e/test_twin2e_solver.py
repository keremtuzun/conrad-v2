from __future__ import annotations

import numpy as np
import pytest

from conrad.twins.twin2e.config import BoundaryCondition, GridConfig
from conrad.twins.twin2e.grid import FieldGrid
from conrad.twins.twin2e.solver import CFLViolationError, advect_diffuse, cfl_number, total_mass
from conrad.twins.twin2e.validators import closed_box_mass_drift

GRID = FieldGrid.from_config("t", GridConfig(origin_m=(0, 0, 0), spacing_m=(2.0, 3.0, 1.0), shape=(10, 8, 6)))


def _velocity(rng, scale=0.3, uniform=False):
    if uniform:
        return np.stack([np.full(GRID.shape, v) for v in (0.2, -0.1, 0.0)])
    return scale * rng.standard_normal((3, *GRID.shape))


def test_closed_box_conserves_mass_with_nonuniform_flow():
    rng = np.random.default_rng(0)
    c0 = rng.uniform(0.0, 5.0, GRID.shape)
    drift = closed_box_mass_drift(GRID, c0, _velocity(rng), (0.5, 0.2, 0.01), 20.0, 25)
    assert drift < 1e-12


def test_periodic_conserves_mass():
    rng = np.random.default_rng(1)
    c0 = rng.uniform(0.0, 5.0, GRID.shape)
    c, _ = advect_diffuse(c0, _velocity(rng), (0.5, 0.5, 0.1), GRID, 60.0, BoundaryCondition.PERIODIC)
    assert abs(total_mass(c, GRID) - total_mass(c0, GRID)) / total_mass(c0, GRID) < 1e-12


@pytest.mark.parametrize("bc", list(BoundaryCondition))
def test_positivity_preserved(bc):
    rng = np.random.default_rng(2)
    c0 = np.zeros(GRID.shape)
    c0[4, 4, 3] = 100.0  # sharp spike is the hardest case for positivity
    c, rep = advect_diffuse(c0, _velocity(rng, 0.5), (1.0, 1.0, 0.1), GRID, 300.0, bc)
    assert c.min() >= 0.0
    assert rep.cfl_number <= 0.8 + 1e-12
    assert rep.substeps > 1


def test_open_boundary_uniform_state_is_steady():
    c0 = np.full(GRID.shape, 3.0)
    c, _ = advect_diffuse(
        c0,
        _velocity(None, uniform=True),
        (0.3, 0.3, 0.0),
        GRID,
        100.0,
        BoundaryCondition.OPEN,
        ghost=np.full(GRID.shape, 3.0),
    )
    assert np.allclose(c, 3.0, atol=1e-12)


def test_open_boundary_inflow_carries_ghost_value():
    c0 = np.zeros(GRID.shape)
    c, _ = advect_diffuse(
        c0,
        _velocity(None, uniform=True),
        (0.0, 0.0, 0.0),
        GRID,
        200.0,
        BoundaryCondition.OPEN,
        ghost=np.full(GRID.shape, 1.0),
    )
    assert c[0].mean() > 0.1  # inflow face at low x (u > 0)
    assert c[-1].max() < c[0].min() + 1e-9


def test_diffusion_smooths_and_reduces_variance():
    rng = np.random.default_rng(3)
    c0 = rng.uniform(0, 1, GRID.shape)
    c, _ = advect_diffuse(
        c0, np.zeros((3, *GRID.shape)), (0.5, 0.5, 0.1), GRID, 50.0, BoundaryCondition.CLOSED
    )
    assert c.var() < c0.var()


def test_cfl_enforced_by_substeps_and_violation_raises():
    rng = np.random.default_rng(4)
    v = _velocity(rng, 1.0)
    assert cfl_number(v, (1.0, 1.0, 1.0), GRID, 1000.0) > 1.0
    with pytest.raises(CFLViolationError):
        advect_diffuse(
            np.ones(GRID.shape), v, (1.0, 1.0, 1.0), GRID, 1000.0, BoundaryCondition.CLOSED, max_substeps=3
        )


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        advect_diffuse(
            np.ones(GRID.shape), np.zeros((3, *GRID.shape)), (1, 1, 1), GRID, -1.0, BoundaryCondition.OPEN
        )
    with pytest.raises(ValueError):
        advect_diffuse(
            np.ones((2, 2, 2)), np.zeros((3, *GRID.shape)), (1, 1, 1), GRID, 1.0, BoundaryCondition.OPEN
        )
    with pytest.raises(ValueError):
        advect_diffuse(
            np.ones(GRID.shape), np.zeros((3, *GRID.shape)), (-1, 1, 1), GRID, 1.0, BoundaryCondition.OPEN
        )
