"""Direct updates: UNKNOWN != occupied, ray free space physics, pose-sigma spread, sonar footprint."""

from __future__ import annotations

import numpy as np
from m2s_unit_helpers import depth_image, depth_spec, line, make_obs, new_model, pose, sonar_image, sonar_spec

from conrad.domains.spatial.queries import UnknownPolicy
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality
from conrad.schemas.timebase import stamp

NO_REFINE = {"refinement": {"enabled": False}}


def _observe(model, ids, store, spec, robot, wall_x=3.1, t=1.0, **kw):
    obs = make_obs(ids, store, spec, robot, depth_image(spec, robot, wall_x, **kw), t)
    model.ingest_observations([obs])
    return model.update_beliefs(stamp(t, "SIM"))


def test_unknown_is_never_occupied_and_is_free_follows_policy(store):
    ids = IdFactory(1)
    spec = depth_spec(ids)
    m = new_model(ids, store, [spec], **NO_REFINE)
    pts = line(-2.0, 8.0)
    assert set(m.occupancy_status(pts)) == {KnowledgeStatus.UNKNOWN}
    assert np.allclose(m.occupancy_probability(pts), 0.5)
    assert not m.is_free(pts).any()  # conservative default
    assert m.is_free(pts, UnknownPolicy.PERMISSIVE).all()
    _observe(m, ids, store, spec, pose())
    behind = line(3.5, 6.0)
    st = m.occupancy_status(behind)
    assert set(st) == {KnowledgeStatus.UNKNOWN}
    assert (m.occupancy_probability(behind) < m.cfg.occupancy.p_occupied).all()
    assert not m.is_free(behind).any()
    assert m.is_free(behind, UnknownPolicy.PERMISSIVE).all()
    # the map state itself is unchanged by the policy choice
    assert set(m.occupancy_status(behind)) == {KnowledgeStatus.UNKNOWN}


def test_free_space_stops_before_the_hit_and_never_passes_it(store):
    ids = IdFactory(2)
    spec = depth_spec(ids)
    m = new_model(ids, store, [spec], **NO_REFINE)
    for k in range(3):
        _observe(m, ids, store, spec, pose(), t=1.0 + k)
    front = line(0.5, 2.5)
    assert set(m.occupancy_status(front)) == {KnowledgeStatus.OBSERVED}
    assert m.is_free(front).all()
    wall = np.array([[3.1, 0.1, 0.1], [3.1, -0.4, 0.3]])
    assert (m.occupancy_probability(wall) > 0.9).all()
    assert not m.is_free(wall).any()
    exp = m.export_grid().level(0)
    assert exp.centers_m[:, 0].max() < 3.25 + 0.25 * (m.cfg.pose.max_splat_radius_cells + 1)
    assert not ((exp.centers_m[:, 0] > 3.6) & (exp.status != "UNKNOWN")).any()


def test_dropout_and_out_of_range_carve_nothing(store):
    ids = IdFactory(3)
    spec = depth_spec(ids, max_range=5.0)
    m = new_model(ids, store, [spec], **NO_REFINE)
    img = np.full((spec.parameters["height_px"], spec.parameters["width_px"]), np.nan, dtype=np.float32)
    m.ingest_observations([make_obs(ids, store, spec, pose(), img, 1.0)])
    m.update_beliefs(stamp(1.0, "SIM"))
    assert len(m.map.base) == 0  # an all-dropout image says nothing about free space
    _observe(m, ids, store, spec, pose(), wall_x=7.0, t=2.0)  # wall beyond max range -> NaN returns
    assert len(m.map.base) == 0
    # a point cloud return beyond max range is ignored too
    pc = np.array([[6.0, 0.0, 0.0]], dtype=np.float32)
    m.ingest_observations([make_obs(ids, store, spec, pose(), pc, 3.0, Modality.POINT_CLOUD)])
    m.update_beliefs(stamp(3.0, "SIM"))
    assert len(m.map.base) == 0


def test_pose_sigma_widens_footprint_and_lowers_confidence(store):
    ids = IdFactory(4)
    spec = depth_spec(ids)
    stats = {}
    for sigma in (0.0, 0.3):
        m = new_model(ids, store, [spec], **NO_REFINE)
        for k in range(5):
            _observe(m, ids, store, spec, pose(sigma=sigma), t=1.0 + k)
        base = m.map.base
        rows = base.all_rows()
        hit = rows[base.f["w_hit"][rows] > 0]
        stats[sigma] = (len(hit), base.probability(hit).max(), base.uncertainty(hit)[:, 1].mean())
    assert stats[0.3][0] > 1.5 * stats[0.0][0]  # wider footprint
    assert stats[0.3][1] < 0.9 <= stats[0.0][1]  # never confident at a precise cell under a large sigma
    assert stats[0.3][2] > stats[0.0][2]  # pose-induced spread shows up in U_E


def test_missing_covariance_is_not_treated_as_zero(store):
    ids = IdFactory(5)
    spec = depth_spec(ids)
    m = new_model(ids, store, [spec], **NO_REFINE)
    for k in range(5):
        _observe(m, ids, store, spec, pose(sigma=None), t=1.0 + k)
    rows = m.map.base.all_rows()
    assert m.map.base.probability(rows).max() < 0.9
    assert m.map.base.uncertainty(rows)[:, 1].max() > 0.5


def test_sonar_elevation_ambiguity_gives_no_thin_surface(store):
    ids = IdFactory(6)
    son, dep = sonar_spec(ids), depth_spec(ids)
    ms = new_model(ids, store, [son], **NO_REFINE)
    for k in range(3):
        ms.ingest_observations([make_obs(ids, store, son, pose(), sonar_image(son, 3.1), 1.0 + k)])
        ms.update_beliefs(stamp(1.0 + k, "SIM"))
    md = new_model(ids, store, [dep], **NO_REFINE)
    for k in range(3):
        _observe(md, ids, store, dep, pose(), t=1.0 + k)
    es, ed = ms.export_grid().level(0), md.export_grid().level(0)
    s_hit = es.probability > 0.5
    assert s_hit.any()
    assert np.ptp(es.indices[s_hit][:, 2]) >= 2  # the return spreads over the elevation arc
    assert es.probability.max() < ed.probability.max()
    assert es.uncertainty[s_hit, 0].mean() > ed.uncertainty[ed.probability > 0.5, 0].mean()
