from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from conrad.domains.ecological import EntityBeliefStore, FieldBeliefGrid, Model2EConfig
from conrad.domains.ecological.config import EntityConfig
from conrad.schemas.ids import IdFactory

CFG = Model2EConfig()
NS = 10**9

reading = st.tuples(
    st.floats(-40, 40), st.floats(-40, 40), st.floats(-20, 0), st.floats(-5.0, 40.0), st.floats(0.0, 7200.0)
)


@settings(max_examples=40, deadline=None)
@given(readings=st.lists(reading, min_size=1, max_size=12))
def test_field_variance_stays_between_floor_and_prior(readings):
    fg = FieldBeliefGrid(CFG)
    fb = fg.fields["temperature"]
    # level floor + local floor; prior_var = level prior + largest admissible (EB) local + trend variance
    floor = CFG.variance_floor_fraction * (fb.spec.prior_sd**2 + fb.spec.local_sd**2)
    for x, y, z, v, t in readings:
        fg.update_point("temperature", 0, np.array([x, y, z]), v, 0.05**2, round(t * NS))
        assert np.all(fb.var >= floor - 1e-12) and np.all(fb.var <= fb.prior_var * (1 + 1e-9))
        assert np.all(np.isfinite(fb.mean))


@settings(max_examples=40, deadline=None)
@given(v=st.floats(0.0, 40.0), dt1=st.floats(1.0, 1e5), extra=st.floats(1.0, 1e5))
def test_prediction_variance_is_monotone_in_delta_t(v, dt1, extra):
    fg = FieldBeliefGrid(CFG)
    fg.update_point("temperature", 0, np.array([0.0, 0.0, -10.0]), v, 0.01, 0)
    base = fg.fields["temperature"].var.copy()
    _, v1 = fg.moments_at_time("temperature", round(dt1 * NS))
    _, v2 = fg.moments_at_time("temperature", round((dt1 + extra) * NS))
    assert np.all(v1 >= base - 1e-12) and np.all(v2 >= v1 - 1e-12)


@settings(max_examples=40, deadline=None)
@given(
    obs=st.lists(
        st.tuples(st.floats(-0.5, 1.5), st.floats(1e-4, 0.3), st.floats(0.0, 86400.0 * 5)), max_size=10
    )
)
def test_cover_belief_stays_bounded(obs):
    store = EntityBeliefStore(EntityConfig())
    b = store.add(IdFactory(1).new(), "BENTHIC_PATCH", np.zeros(3), "WORLD", None)
    for y, r, t in obs:
        store.update_cover(b, y, r, round(t * NS), inflation=2.0)
        assert 0.0 <= b.cover_mean <= 1.0 and 0.0 < b.cover_var <= 0.25
