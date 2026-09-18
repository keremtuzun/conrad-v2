"""Relational gap fill, hidden side, contradiction -> U_C, temporal prediction, refinement."""

from __future__ import annotations

import math

import numpy as np
from m2s_unit_helpers import depth_image, depth_spec, make_obs, new_model, pose

from conrad.domains.spatial.grid import SRC_RELATIONAL
from conrad.domains.spatial.keys import points_to_codes
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp

NO_REFINE = {"refinement": {"enabled": False}}


def _observe(model, ids, store, spec, robot, wall_x, t, **kw):
    obs = make_obs(ids, store, spec, robot, depth_image(spec, robot, wall_x, **kw), t)
    evs = model.ingest_observations([obs])
    return model.update_beliefs(stamp(t, "SIM")), evs


def test_small_gap_is_inferred_with_relational_provenance(store):
    ids = IdFactory(11)
    spec = depth_spec(ids)
    m = new_model(ids, store, [spec], **NO_REFINE)
    consumed = set()
    for k in range(3):
        _, evs = _observe(m, ids, store, spec, pose(), 3.1, 1.0 + k, hole_y=(0.0, 0.25))
        consumed |= {e.evidence_id for e in evs}
    gap = np.array([[3.1, 0.125, z] for z in (-0.2, 0.1, 0.3)])
    assert set(m.occupancy_status(gap)) == {KnowledgeStatus.INFERRED}
    p = m.occupancy_probability(gap)
    assert (p > 0.5).all() and (p < 0.9).all()  # sub-confident
    st = m.occupancy_state(gap)
    wall = m.occupancy_state(np.array([[3.1, -0.3, 0.1]]))
    assert (st.uncertainty[:, 3] >= 0.7).all() and st.uncertainty[0, 3] > wall.uncertainty[0, 3]
    base = m.map.base
    rows = base.rows(points_to_codes(gap, base.res))
    assert (base.src[rows] == SRC_RELATIONAL).all()
    for r in rows.tolist():
        assert base.provenance[r] and set(base.provenance[r]) <= consumed
    assert not m.is_free(gap).any()


def test_large_gap_and_hidden_side_stay_unknown(store):
    ids = IdFactory(12)
    spec = depth_spec(ids)
    m = new_model(ids, store, [spec], **NO_REFINE)
    for k in range(3):
        _observe(m, ids, store, spec, pose(), 3.1, 1.0 + k, hole_y=(-0.5, 0.5))
    hole = np.array([[3.1, 0.125, 0.1], [3.1, -0.125, 0.1]])
    assert set(m.occupancy_status(hole)) == {KnowledgeStatus.UNKNOWN}
    # a slab occupies x in [3.1, 4.85]; its back face layer is hidden from the front views
    behind = SpatialSupport(frame_id="WORLD", center_m=(4.875, 0.0, 0.0), half_extent_m=(0.1, 1.0, 1.0))
    assert m.unknown_fraction(behind) == 1.0 and m.coverage(behind) == 0.0
    # observing the back face from the far side is what resolves it, not inference
    _observe(m, ids, store, spec, pose(x=7.0, yaw=math.pi), 4.85, 5.0)
    assert m.unknown_fraction(behind) < 1.0 and m.coverage(behind) > 0.0


def test_contradicting_reobservation_raises_uc_and_marks_dynamic(store):
    ids = IdFactory(13)
    spec = depth_spec(ids)
    m = new_model(ids, store, [spec], **NO_REFINE)
    for k in range(3):
        _observe(m, ids, store, spec, pose(), 3.1, 1.0 + k)
    wall = np.array([[3.1, 0.1, 0.1], [3.1, -0.3, -0.2]])
    before = m.occupancy_state(wall)
    assert (before.probability > 0.9).all() and (before.uncertainty[:, 2] == 0).all()
    for k in range(2):
        msgs, _ = _observe(m, ids, store, spec, pose(), 5.1, 10.0 + k)  # the wall moved away
    after = m.occupancy_state(wall)
    assert (after.uncertainty[:, 2] > 0.3).all()
    rows = m.map.base.rows(points_to_codes(wall, 0.25))
    assert m.map.base.b["dynamic"][rows].all()
    assert max(msg.uncertainty.contradiction for msg in msgs) > 0.0


def test_predict_decays_dynamic_cells_and_only_ages_static_ones(store):
    ids = IdFactory(14)
    spec = depth_spec(ids)
    m = new_model(ids, store, [spec], **NO_REFINE)
    for k in range(3):
        _observe(m, ids, store, spec, pose(), 3.1, 1.0 + k)
    for k in range(2):
        _observe(m, ids, store, spec, pose(), 5.1, 10.0 + k)
    moved = np.array([[3.1, 0.1, 0.1]])
    static = np.array([[1.1, 0.1, 0.1]])
    s0, d0 = m.occupancy_state(static), m.occupancy_state(moved)
    msgs = m.predict(20.0, stamp(31.0, "SIM"))
    s1, d1 = m.occupancy_state(static), m.occupancy_state(moved)
    assert m.occupancy_status(moved) == [KnowledgeStatus.PREDICTED]
    assert d1.uncertainty[0, 3] > d0.uncertainty[0, 3]
    assert abs(d1.probability[0] - 0.5) < abs(d0.probability[0] - 0.5)
    assert np.allclose(s1.probability, s0.probability) and (s1.status == s0.status).all()
    assert msgs and all(msg.evidence_support == () for msg in msgs)
    assert all(c.status is KnowledgeStatus.PREDICTED for msg in msgs for c in msg.state_summary)
    ages = m.export_grid(stamp(31.0, "SIM").time_ns).level(0).age_s
    assert np.nanmin(ages) >= 20.0 - 1e-9
    # a fresh direct observation turns a predicted cell back into OBSERVED
    _observe(m, ids, store, spec, pose(), 5.1, 32.0)
    assert m.occupancy_status(moved) == [KnowledgeStatus.OBSERVED]


def test_refinement_is_deterministic_and_local(store):
    ids1, ids2 = IdFactory(15), IdFactory(15)
    exports = []
    for ids in (ids1, ids2):
        spec = depth_spec(ids)
        m = new_model(ids, store, [spec])
        for k in range(2):
            _observe(m, ids, store, spec, pose(), 3.1, 1.0 + k)
        exports.append(m.export_grid())
    a, b = exports
    assert len(a.level(1).indices) > 0 and len(a.level(2).indices) > 0
    assert np.array_equal(a.level(1).indices, b.level(1).indices)
    assert np.linalg.norm(a.level(1).centers_m, axis=1).max() <= m.cfg.refinement.local_radius_m + 0.5
    # the finest level resolves the surface position better than the base cell size
    fine = a.level(2)
    occ = fine.centers_m[fine.probability > 0.8][:, 0]
    assert occ.size and np.all(np.abs(occ - 3.1) < 0.15)
