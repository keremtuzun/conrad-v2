"""Model2T iteration 4 (docs/audits/MODEL2T_REPAIR.md): coverage credit per view, per-region (locus) sizing,
intact-band calibration. No Twin2T involvement."""

from __future__ import annotations

from dataclasses import replace

from m2t_helpers import claim, context, evidence, model, registry_ids

from conrad.domains.technical import CORROSION_DEPTH, CRACK_LENGTH, Model2T, Model2TConfig
from conrad.domains.technical.config import CoverageConfig
from conrad.domains.technical.coverage import SurfaceGeometry
from conrad.domains.technical.state import WHOLE_COMPONENT, reveal_probability
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.frames import WORLD, SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp

CFG = Model2TConfig()
NEAR = (2.0, -0.25, 0.0)  # -y side of a pipe running along x
FAR = (2.0, 0.25, 0.0)


def _geometry(r):
    return [
        {
            "registry_id": r["seg_a"],
            "shape": "CAPSULE",
            "p0_m": (0.0, 0.0, 0.0),
            "p1_m": (4.0, 0.0, 0.0),
            "radius_m": 0.25,
        }
    ]


def _model(cfg=None, occlusion=None):
    r = registry_ids()
    m = Model2T(IdFactory(seed=3, namespace="model2t"), cfg)
    ctx = {**context(r), "design_geometry": _geometry(r)}
    if occlusion is not None:
        ctx["surface_occlusion"] = occlusion
    m.initialize(ctx)
    return m, r, IdFactory(seed=3, namespace="evidence")


def _located(ev, target, t, point, **meas):
    e = evidence(ev, target, t, group=f"sensor:S|obs:{t}", **meas)
    sup = SpatialSupport(frame_id=WORLD, center_m=point, position_sigma_m=0.05)
    return e.model_copy(update={"spatial_support": sup})


def _feed(m, r, ev, point, n, *, t0=10.0, **meas):
    for i in range(n):
        t = t0 + i
        m.ingest([_located(ev, r["seg_a"], t, point, **meas)])
        m.update_beliefs(stamp(t, "sim"))
    return m.beliefs[r["seg_a"]]


# ---------------------------------------------------------------- coverage credit per view
def test_one_view_credits_its_footprint_and_never_the_far_side():
    m, r, ev = _model()
    b = _feed(m, r, ev, NEAR, 1, wall=4e-4, crack=0.0)
    assert b.geometry is not None
    anchor = b.geometry.cell_of(NEAR)
    assert anchor in b.covered
    assert len(b.covered) > 1, "a view credits the cells it looked at, not only the measured one"
    assert b.geometry.cell_of(FAR) not in b.covered
    _, normals, _ = b.geometry.cell_frames
    limit = CFG.coverage.footprint_half_angle_deg
    for cell in b.covered:
        cos = float(normals[cell] @ normals[anchor])
        assert cos >= __import__("math").cos(__import__("math").radians(limit)) - 1e-9


def test_measured_cell_ablation_reproduces_iteration_3_credit():
    cfg = replace(CFG, coverage=replace(CFG.coverage, view_credit="MEASURED_CELL"))
    m, r, ev = _model(cfg)
    b = _feed(m, r, ev, NEAR, 1, wall=4e-4, crack=0.0)
    assert b.covered == {b.geometry.cell_of(NEAR)}


def test_model2s_occlusion_removes_credit_for_cells_that_could_not_be_seen():
    seen: list[int] = []

    def blocked(points):
        seen.append(len(points))
        return [True] * len(points)  # the belief map says every probe sits in occupied space

    m, r, ev = _model(occlusion=blocked)
    b = _feed(m, r, ev, NEAR, 1, wall=4e-4, crack=0.0)
    assert seen and seen[0] > 1
    assert b.covered == {b.geometry.cell_of(NEAR)}, "only the cell actually measured survives"


def test_a_never_observed_component_stays_unknown_with_full_observational_uncertainty():
    m, r, ev = _model()
    _feed(m, r, ev, NEAR, 2, wall=4e-4, crack=0.0)
    other = next(x for x in m.export_beliefs() if x.world_entity_id == r["seg_b"])
    assert claim(other, "condition").status is KnowledgeStatus.UNKNOWN
    assert other.uncertainty.observational >= 0.999


def test_coverage_alone_does_not_make_the_worst_case_seen():
    """U_O uses coverage x the declared probability that a look reveals a band-size defect, not coverage."""
    m, r, ev = _model()
    b = _feed(m, r, ev, NEAR, 1, wall=4e-4, crack=0.0)
    cov = b.coverage_fraction()
    assert cov is not None and cov > 0.1
    assert b.reveal < 1.0 and b.seen_fraction() == cov * b.reveal
    assert b.uncertainty().observational > 1.0 - cov
    assert reveal_probability(frozenset({CORROSION_DEPTH}), CFG, 0.0) == 1.0


# ---------------------------------------------------------------- per-region (locus) sizing
def test_a_far_side_finding_is_not_diluted_by_near_side_readings():
    """The whole point of per-region sizing for I4: one informative view moves the component estimate."""
    m, r, ev = _model()
    _feed(m, r, ev, NEAR, 12, wall=4e-4, crack=0.0)
    before = m.beliefs[r["seg_a"]].estimates[CRACK_LENGTH].level
    b = _feed(m, r, ev, FAR, 1, t0=40.0, wall=4e-4, crack=0.06)
    assert b.estimates[CRACK_LENGTH].level > 10.0 * max(before, 1e-6)
    assert b.estimates[CRACK_LENGTH].level > 0.02
    assert len(b.regions) == 2, "the near-side and far-side looks are separate regions"


def test_repeated_looks_at_one_region_revise_a_false_call_down():
    """A miss in the region the call came from is informative about that region (iteration 3 called it
    'elsewhere' only by distance, so a false call at a locus could never be revised)."""
    m, r, ev = _model()
    b = _feed(m, r, ev, NEAR, 3, wall=4e-4, crack=0.02)
    after_call = b.estimates[CRACK_LENGTH].level
    assert after_call > 0.005
    b = _feed(m, r, ev, NEAR, 20, t0=40.0, wall=4e-4, crack=0.0)
    assert b.estimates[CRACK_LENGTH].level < after_call


def test_regions_are_only_used_where_there_is_design_geometry():
    m, r, ev = model()
    m.ingest([evidence(ev, r["seg_a"], 10.0, wall=4e-4)])
    m.update_beliefs(stamp(10.0, "sim"))
    b = m.beliefs[r["seg_a"]]
    assert set(b.regions) == {WHOLE_COMPONENT}
    assert b.regions[WHOLE_COMPONENT].estimates is b.estimates  # the iteration-3 path, updated in place


# ---------------------------------------------------------------- intact band
def test_an_indication_at_the_noise_floor_does_not_condemn_a_pristine_surface():
    """A structured-light payload reports crack-like indications from welds, scratches and growth, so a long
    look at a pristine surface throws one now and then. With the declared false-indication tail, and with the
    misses in the SAME region now informative, the region is not left condemned by it."""
    m, r, ev = _model()
    b = _feed(m, r, ev, NEAR, 1, wall=1e-4, crack=0.0035)
    b = _feed(m, r, ev, NEAR, 10, t0=20.0, wall=1e-4, crack=0.0)
    sev = b.estimates[CRACK_LENGTH].level / CFG.condition.crack_critical_m
    assert sev < CFG.condition.bands[0], f"pristine surface reported severity {sev}"


def test_a_real_crack_read_repeatedly_is_still_separated_from_the_noise_floor():
    m, r, ev = _model()
    b = _feed(m, r, ev, NEAR, 6, wall=1e-4, crack=0.012)
    assert b.estimates[CRACK_LENGTH].level > CFG.condition.bands[0] * CFG.condition.crack_critical_m
    assert b.estimates[CRACK_LENGTH].status is KnowledgeStatus.OBSERVED


def test_the_false_indication_tail_is_configurable_and_off_reproduces_iteration_3():
    off = replace(CFG, sensor=replace(CFG.sensor, crack_false_call_weight=0.0))
    m, r, ev = _model(off)
    b = _feed(m, r, ev, NEAR, 2, wall=1e-4, crack=0.012)
    with_tail, _, ev2 = _model()
    b2 = _feed(with_tail, r, ev2, NEAR, 2, wall=1e-4, crack=0.012)
    assert b.estimates[CRACK_LENGTH].level > b2.estimates[CRACK_LENGTH].level


# ---------------------------------------------------------------- geometry helpers
def test_cells_in_view_is_symmetric_and_bounded_by_the_declared_footprint():
    geo = SurfaceGeometry(
        shape="CAPSULE", p0=(0.0, 0.0, 0.0), p1=(4.0, 0.0, 0.0), radius_m=0.25, n_along=8, sectors=8
    )
    cfg = CoverageConfig()
    anchor, mask = geo.cells_in_view(NEAR, cfg.footprint_half_angle_deg, cfg.footprint_axial_m)
    assert mask[anchor]
    assert 1 < int(mask.sum()) < geo.n_cells
    probes = geo.probe_points(mask, cfg.probe_offset_m)
    assert len(probes) == int(mask.sum())
