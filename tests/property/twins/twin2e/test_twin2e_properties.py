from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.twins.twin2e import BoundaryCondition, Twin2E, build_small_scenario, load_config, validate_twin
from conrad.twins.twin2e.config import GridConfig
from conrad.twins.twin2e.grid import FieldGrid
from conrad.twins.twin2e.solver import advect_diffuse, total_mass

REPO = Path(__file__).resolve().parents[4]
CFG = load_config(REPO / "configs" / "sim" / "twin2e_test_small.yaml")
GRID = FieldGrid.from_config("p", GridConfig(origin_m=(0, 0, 0), spacing_m=(4.0, 4.0, 2.0), shape=(6, 5, 4)))
_STORE = ObjectStore(tempfile.mkdtemp(prefix="twin2e-prop-"))


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    u_scale=st.floats(0.0, 1.0),
    k=st.floats(0.0, 2.0),
    dt=st.floats(1.0, 500.0),
    bc=st.sampled_from(list(BoundaryCondition)),
)
def test_solver_positive_and_closed_conservative(seed, u_scale, k, dt, bc):
    rng = np.random.default_rng(seed)
    c0 = rng.exponential(1.0, GRID.shape)
    v = u_scale * rng.standard_normal((3, *GRID.shape))
    c, rep = advect_diffuse(c0, v, (k, k, k / 10), GRID, dt, bc, ghost=np.zeros(GRID.shape))
    assert np.all(np.isfinite(c))
    assert c.min() >= -1e-12
    assert rep.cfl_number <= 0.8 + 1e-12
    if bc is not BoundaryCondition.OPEN:
        assert abs(total_mass(c, GRID) - total_mass(c0, GRID)) <= 1e-10 * total_mass(c0, GRID)


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 5_000), dts=st.lists(st.floats(1.0, 4000.0), min_size=1, max_size=4))
def test_twin_state_bounded(seed, dts):
    ids = IdFactory(seed)
    tw = Twin2E(IdFactory(seed + 1), _STORE, CFG)
    tw.initialize(build_small_scenario(ids, seed, CFG))
    for dt in dts:
        tw.step(dt)
        assert validate_twin(tw) == []
    for t in tw.get_truth(tw.now()):
        if "cover" in t.state:
            assert 0.0 <= t.state["cover"] <= 1.0
            assert 0.0 <= t.state["condition"] <= 1.0
