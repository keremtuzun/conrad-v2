"""Model2T iteration 3 (docs/audits/MODEL2T_REPAIR.md): regime-mixture crack model, surface coverage, local
2T partitions. No Twin2T involvement."""

from __future__ import annotations

from dataclasses import replace

import pytest
from m2t_helpers import DAY, claim, context, evidence, model, registry_ids

from conrad.domains.technical import CRACK_LENGTH, Model2T, Model2TConfig, PropagationMode
from conrad.domains.technical.crack_filter import moments, population_prior, propagate, runaway_probability
from conrad.domains.technical.persist import SURFACE_SUFFIX
from conrad.evaluation.partitions import PartitionAccessError
from conrad.evaluation.structural_experiments.common import checked_seeds
from conrad.schemas.belief import KnowledgeStatus, UpdateKind
from conrad.schemas.frames import WORLD, SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp

CFG = Model2TConfig()


def _feed(m, r, ev, readings, *, t0=10.0, dt=1.0, target="seg_a"):
    for i, z in enumerate(readings):
        t = t0 + i * dt
        m.ingest([evidence(ev, r[target], t, crack=z, group=f"sensor:S|obs:{t}")])
        m.predict(dt if i else 0.0, stamp(t, "sim"))
        m.update_beliefs(stamp(t, "sim"))
    return m.beliefs[r[target]]


# ---------------------------------------------------------------- crack temporal model
def test_crack_predictive_uncertainty_grows_with_physical_elapsed_time():
    g30, g365 = population_prior(CFG.crack_growth, CFG.prior), population_prior(CFG.crack_growth, CFG.prior)
    propagate(g30, 30 * DAY, CFG.crack_growth)
    propagate(g365, 365 * DAY, CFG.crack_growth)
    v30, v365 = moments(g30, CFG.crack_growth).level_var, moments(g365, CFG.crack_growth).level_var
    assert v365 > 2.0 * v30
    # physical delta_t, not a step count: 12 steps of ~30 days ~ one step of 365 days
    stepped = population_prior(CFG.crack_growth, CFG.prior)
    for _ in range(12):
        propagate(stepped, 365 * DAY / 12, CFG.crack_growth)
    assert moments(stepped, CFG.crack_growth).level_var == pytest.approx(v365, rel=0.25)


def test_short_ticks_accumulate_until_applied():
    g = population_prior(CFG.crack_growth, CFG.prior)
    before = g.p.copy()
    assert not propagate(g, 0.1, CFG.crack_growth)  # a 0.1 s mission tick is only accumulated
    assert (g.p == before).all() and g.pending_s == pytest.approx(0.1)
    assert propagate(g, 0.0, CFG.crack_growth, force=True) and g.pending_s == 0.0


def test_growing_readings_raise_the_run_away_regime_and_static_ones_do_not():
    m, r, ev = model(mode=PropagationMode.NONE)
    grow = _feed(m, r, ev, [0.02, 0.035, 0.06, 0.10, 0.17], dt=60 * DAY)
    m2, r2, ev2 = model(mode=PropagationMode.NONE)
    static = _feed(m2, r2, ev2, [0.02, 0.021, 0.019, 0.02, 0.022], dt=60 * DAY)
    assert grow.crack_grid is not None and static.crack_grid is not None
    assert runaway_probability(grow.crack_grid) > 0.5 > runaway_probability(static.crack_grid)
    # between inspections the run-away crack's prediction is wider and larger than the static one's
    m.predict(180 * DAY, stamp(10.0 + 5 * 60 * DAY, "sim"))
    m2.predict(180 * DAY, stamp(10.0 + 5 * 60 * DAY, "sim"))
    a, b = grow.estimates[CRACK_LENGTH], static.estimates[CRACK_LENGTH]
    assert a.level > 0.17 and a.sd > 5 * b.sd


def test_constant_rate_ablation_keeps_the_iteration_2_gaussian_model():
    cfg = replace(CFG, crack_growth=replace(CFG.crack_growth, model="CONSTANT_RATE"))
    m, r, ev = model(config=cfg, mode=PropagationMode.NONE)
    b = _feed(m, r, ev, [0.02, 0.03])
    assert b.crack_grid is None and b.estimates[CRACK_LENGTH].status is KnowledgeStatus.OBSERVED


# ---------------------------------------------------------------- surface coverage
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


def _located(ev, target, t, point, **meas):
    e = evidence(ev, target, t, group=f"sensor:S|obs:{t}", **meas)
    sup = SpatialSupport(frame_id=WORLD, center_m=point, position_sigma_m=0.05)
    return e.model_copy(update={"spatial_support": sup})


def _coverage_model(repo=None):
    r = registry_ids()
    m = Model2T(IdFactory(seed=3, namespace="model2t"), repository=repo, run_id=IdFactory(seed=4).new())
    m.initialize({**context(r), "design_geometry": _geometry(r)})
    return m, r, IdFactory(seed=3, namespace="evidence")


def _component(m, rid):
    return next(x for x in m.export_beliefs() if x.world_entity_id == rid)


def test_near_side_view_observes_only_the_read_surface(repo):
    m, r, ev = _coverage_model(repo)
    near = (2.0, -0.25, 0.0)  # -y side of a pipe along x
    m.ingest([_located(ev, r["seg_a"], 10.0, near, wall=4e-4, crack=0.0)])
    out = m.update_beliefs(stamp(10.0, "sim"))
    comp = _component(m, r["seg_a"])
    assert claim(comp, "condition").status is KnowledgeStatus.UNKNOWN
    assert claim(comp, "corrosion_depth_m").value is None  # the component worst case is not observed
    cov = claim(comp, "surface_coverage").value
    # Iteration 4 credits the reading's DECLARED footprint, not one cell; the far side is never credited.
    belief = m.beliefs[r["seg_a"]]
    assert belief.geometry is not None
    assert belief.geometry.cell_of((2.0, 0.25, 0.0)) not in belief.covered
    assert 0.0 < cov < belief.coverage_complete
    assert comp.uncertainty.observational >= 1.0 - cov - 1e-12
    surface = [x for x in out if x.world_entity_id is None]
    assert len(surface) == 1 and claim(surface[0], "corrosion_depth_m").status is KnowledgeStatus.OBSERVED
    revs = repo.all_revisions()
    direct = [x for x in revs if x.update_kind is UpdateKind.DIRECT]
    assert len(direct) == 1 and direct[0].cell.registry_entity_id is None
    assert direct[0].cell.entity_type.endswith(SURFACE_SUFFIX)
    assert any(
        x.update_kind is UpdateKind.RELATIONAL and x.cell.registry_entity_id == r["seg_a"] for x in revs
    )


def test_a_worst_band_finding_determines_the_component_condition():
    m, r, ev = _coverage_model()
    far = (2.0, 0.25, 0.0)
    for i in range(3):
        m.ingest([_located(ev, r["seg_a"], 10.0 + i, far, crack=0.08)])
        out = m.update_beliefs(stamp(10.0 + i, "sim"))
    comp = _component(m, r["seg_a"])
    assert claim(comp, "condition").status is KnowledgeStatus.OBSERVED
    assert claim(comp, "condition").value == "FAILED"  # a worst case can only be worse than what was seen
    assert any(x.world_entity_id == r["seg_a"] and x.evidence_support for x in out)


def test_components_without_geometry_keep_whole_component_views():
    m, r, ev = model(mode=PropagationMode.NONE)
    m.ingest([evidence(ev, r["seg_a"], 10.0, wall=4e-4)])
    m.update_beliefs(stamp(10.0, "sim"))
    comp = _component(m, r["seg_a"])
    assert claim(comp, "condition").status is KnowledgeStatus.OBSERVED
    assert all(c.name != "surface_coverage" for c in comp.state_summary)


def test_geometry_for_an_unregistered_component_is_refused():
    r = registry_ids()
    m = Model2T(IdFactory(seed=3, namespace="model2t"))
    bad = [{**_geometry(r)[0], "registry_id": IdFactory(seed=99).new()}]
    with pytest.raises(ValueError, match="non-registry"):
        m.initialize({**context(r), "design_geometry": bad})


# ---------------------------------------------------------------- local 2T partitions
def _cfg(part, purpose):
    return {
        "partition": part,
        "purpose": purpose,
        "partitions": {
            "development": {"range": [5100000, 5100040]},
            "final_3": {"range": [6500000, 6500060]},
            "spent_final": {"range": [5300000, 5300060]},
        },
    }


def test_local_final_split_is_guarded():
    assert checked_seeds(_cfg("final_3", "final_evaluation"), [6500000, 6500059]) == [6500000, 6500059]
    with pytest.raises(PartitionAccessError):
        checked_seeds(_cfg("final_3", "design"), [6500000])  # design may not read a final split
    with pytest.raises(PartitionAccessError):
        checked_seeds(_cfg("final_3", "final_evaluation"), [5300000])  # spent R2 FINAL seed
    with pytest.raises(PartitionAccessError):
        checked_seeds(_cfg("spent_final", "final_evaluation"), [5300000])
    reused = _cfg("final_3", "final_evaluation")
    reused["partitions"]["final_3"] = {"range": [5300000, 5300010]}
    with pytest.raises(PartitionAccessError):
        checked_seeds(reused, [5300000])  # a final split may not reuse spent (or pinned) seeds
