"""OCPWE controls, counterfactual indistinguishability, discriminating views, evaluation helpers."""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pytest
from twin2s_sim_helpers import make_twin

from conrad.schemas.ids import IdFactory
from conrad.twins.twin2s.counterfactual import (
    KINDS,
    discriminating_views,
    make_counterfactual_pair,
    verify_indistinguishable,
)
from conrad.twins.twin2s.coverage import counterfactual_consensus, three_masks
from conrad.twins.twin2s.evaluation import (
    occupancy_error_split,
    occupancy_truth_at,
    unsupported_confidence,
    visible_fraction,
)
from conrad.twins.twin2s.families import generate_scenario
from conrad.twins.twin2s.ocpwe import InformationConditions, ScheduledView, look_at_pose, plan_schedule
from conrad.twins.twin2s.world import SpatialWorld

FAST = {"n_surface_samples": 80, "n_azimuths": 6}


@pytest.fixture(scope="module")
def straight(tmp_path_factory):
    s = generate_scenario(7, IdFactory(7), "straight_pipeline")
    tw = make_twin(s, tmp_path_factory.mktemp("cf"))
    first = UUID(s.spatial_state["groups"]["segments"][0])
    cond = InformationConditions((first,), n_views=2, target_coverage=0.5, **FAST)
    return s, tw, plan_schedule(tw, s.robots[0].sensors, cond, np.random.default_rng(0))


@pytest.mark.parametrize("kind", KINDS)
def test_counterfactual_pair_is_indistinguishable_but_different(straight, kind):
    s, _, sched = straight
    pair = make_counterfactual_pair(s, sched, IdFactory(500), np.random.default_rng(1), kind=kind)
    assert pair.report.max_difference <= 1e-3 and pair.report.n_observations >= 6
    wa = SpatialWorld.from_spatial_state(pair.world_a.spatial_state)
    wb = SpatialWorld.from_spatial_state(pair.world_b.spatial_state)
    lo, hi = np.array(pair.changed_region_min_m), np.array(pair.changed_region_max_m)
    q = np.random.default_rng(2).uniform(lo, hi, (4000, 3))
    differs = wa.occupied(q) != wb.occupied(q)
    assert differs.any(), "W_A must differ from W_B"
    # consensus over the compatible worlds leaves the differing cells UNKNOWN, never INFERABLE
    masks = three_masks(np.zeros(len(q), bool), counterfactual_consensus([wa, wb], q))
    assert not masks["inferable"][differs].any() and masks["unknown"][differs].all()


def test_discriminating_view_separates_the_worlds(straight):
    s, _tw, sched = straight
    pair = make_counterfactual_pair(s, sched, IdFactory(501), np.random.default_rng(1), kind="bend_tail")
    lo, hi = np.array(pair.changed_region_min_m), np.array(pair.changed_region_max_m)
    focus = 0.5 * (lo + hi)
    look = ScheduledView(
        10.0, (p := look_at_pose(focus + np.array([0.0, -3.5, 2.5]), focus)), p, sched.views[0].sensors, {}
    )
    found = discriminating_views(pair, [sched.views[0], look])
    assert [v for v, _ in found] == [look]
    assert verify_indistinguishable(s, s, sched).max_difference == 0.0


def test_ocpwe_coverage_control_is_monotone_and_redundancy_is_not_diversity(straight):
    s, tw, _ = straight
    segs = tuple(UUID(u) for u in s.spatial_state["groups"]["segments"])
    got = {}
    for target in (0.15, 0.6):
        cond = InformationConditions(segs, target_coverage=target, n_views=6, **FAST)
        got[target] = plan_schedule(tw, s.robots[0].sensors, cond, np.random.default_rng(0)).achieved
    assert got[0.15]["cumulative_coverage"] < got[0.6]["cumulative_coverage"]
    red = plan_schedule(
        tw,
        s.robots[0].sensors,
        InformationConditions(segs, style="redundant", n_views=5, **FAST),
        np.random.default_rng(0),
    )
    assert len(red.views) == 5 and red.achieved["redundancy_ratio"] == pytest.approx(5.0)
    assert red.achieved["mean_effective_views"] <= red.achieved["cumulative_coverage"] + 1e-12
    noisy = plan_schedule(
        tw,
        s.robots[0].sensors,
        InformationConditions(segs[:1], n_views=1, pose_sigma_m=0.2, modalities=("SONAR",), **FAST),
        np.random.default_rng(0),
    )
    v = noisy.views[0]
    assert v.true_pose != v.estimated_pose and v.estimated_pose.position_sigma_m() == pytest.approx(0.2)
    assert [x.modality for x in v.sensors] == ["SONAR"]


def test_evaluation_helpers(straight):
    s, tw, sched = straight
    seg = UUID(s.spatial_state["groups"]["segments"][0])
    depth = next(x for x in s.robots[0].sensors if x.modality == "DEPTH_RANGE")
    poses = [v.true_pose for v in sched.views]
    assert 0.0 < visible_fraction(tw, seg, depth, poses, n_samples=80) <= 1.0
    assert visible_fraction(tw, seg, depth, [], n_samples=80) == 0.0
    pts = np.array([[0.0, 0.0, -1.2], [0.0, 0.0, 4.5]])
    assert list(occupancy_truth_at(tw, pts)) == [True, False]
    hidden = np.array([True, True, True, False])
    m = unsupported_confidence(
        [0.99, 0.5, 0.02, 0.99], ["UNKNOWN", "UNKNOWN", "OBSERVED", "OBSERVED"], hidden
    )
    assert m["n_cells"] == 3 and m["uc_rate"] == pytest.approx(2 / 3)
    assert m["uc_rate_not_unknown"] == pytest.approx(1 / 3) and m["claimed_observed_rate"] == pytest.approx(
        1 / 3
    )
    assert np.isnan(unsupported_confidence([0.9], ["UNKNOWN"], [False])["uc_rate"])
    e = occupancy_error_split([1.0, 0.0, 0.5], [True, True, False], [True, False, False])
    assert e == {"E_observed": 0.0, "E_hidden": pytest.approx(0.625)}
