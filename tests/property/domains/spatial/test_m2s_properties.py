"""Property tests: cell indexing round trips and knowledge-status invariants of the UAHSM cell store."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from conrad.domains.spatial.config import SpatialConfig
from conrad.domains.spatial.grid import INFERRED, OBSERVED, PREDICTED, UNKNOWN, SparseLevel
from conrad.domains.spatial.keys import (
    children_index,
    decode,
    encode,
    index_to_center,
    parent_index,
    point_to_index,
)
from conrad.domains.spatial.relational import relational_fill
from conrad.domains.spatial.temporal import predict_level
from conrad.schemas.ids import IdFactory

idx = st.integers(-100_000, 100_000)
res = st.sampled_from([0.25, 0.125, 0.0625, 0.5])


@given(st.lists(st.tuples(idx, idx, idx), min_size=1, max_size=40))
def test_encode_decode_round_trip(cells):
    a = np.array(cells, dtype=np.int64)
    assert np.array_equal(decode(encode(a)), a)
    assert len(np.unique(encode(a))) == len({tuple(c) for c in cells})


@given(st.tuples(idx, idx, idx), res)
def test_center_maps_back_to_its_cell_and_children_to_parent(cell, r):
    a = np.array([cell], dtype=np.int64)
    assert np.array_equal(point_to_index(index_to_center(a, r), r), a)
    kids = children_index(a)
    assert len(kids) == 8 and np.array_equal(parent_index(kids), np.repeat(a, 8, axis=0))
    # every child centre lies inside the parent cell
    assert np.array_equal(point_to_index(index_to_center(kids, r / 2), r), np.repeat(a, 8, axis=0))


@given(
    st.floats(-500.0, 500.0, allow_nan=False),
    st.floats(-500.0, 500.0, allow_nan=False),
    st.floats(-500.0, 500.0, allow_nan=False),
    res,
)
def test_point_lies_inside_its_cell(x, y, z, r):
    p = np.array([[x, y, z]])
    i = point_to_index(p, r)
    lo = i * r
    assert np.all(lo <= p + 1e-9) and np.all(p < lo + r + 1e-9)


updates = st.lists(
    st.tuples(
        st.integers(0, 11),  # cell
        st.floats(0.0, 1.0),  # hit mass
        st.floats(0.0, 1.0),  # free mass
        st.floats(0.0, 1.0),  # pose sigma (m)
    ),
    min_size=1,
    max_size=60,
)


@settings(max_examples=60, deadline=None)
@given(updates, st.floats(0.0, 30.0))
def test_status_invariants_hold_under_any_update_sequence(seq, dt):
    cfg = SpatialConfig()
    lvl = SparseLevel(0.25, cfg)
    ids = IdFactory(1)
    cells = encode(np.array([[i % 4, i // 4, 0] for i in range(12)], dtype=np.int64))
    for t, (c, h, m, s) in enumerate(seq):
        rows = lvl.rows(cells[[c]], create=True)
        ue = np.array([s / (s + 0.25)])
        lvl.apply_direct(rows, np.array([h]), np.array([m]), np.array([0.1]), ue, np.array([s]), ids.new(), t)
    relational_fill(lvl, cfg, lambda codes: lvl.rows(codes, create=True))
    predict_level(lvl, dt, cfg)
    rows = lvl.all_rows()
    status, p, u = lvl.status(rows), lvl.probability(rows), lvl.uncertainty(rows)
    w = lvl.f["w"][rows]
    assert np.all((p >= 0) & (p <= 1)) and np.all((u >= 0) & (u <= 1))
    assert set(status.tolist()) <= {UNKNOWN, OBSERVED, INFERRED, PREDICTED}
    unknown = status == UNKNOWN
    assert np.all(w[unknown] < cfg.occupancy.observed_min_weight)
    assert np.all(np.maximum(p, 1 - p)[unknown] < 0.9)  # UNKNOWN is never a confident claim
    assert np.all(w[status == OBSERVED] >= cfg.occupancy.observed_min_weight)
    assert np.all(u[unknown & (w == 0), 3] == 1.0)
    inferred = status == INFERRED
    assert np.all(~lvl.b["predicted"][rows][inferred])
    assert np.all(p[inferred] < 0.9)
    assert np.all((status == PREDICTED) == lvl.b["predicted"][rows])
