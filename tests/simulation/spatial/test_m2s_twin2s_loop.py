"""Twin2S -> Observations -> Model2S closed loop with small sensors (truth only in the test, never in Model2S)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from conrad.domains.spatial.model import Model2S
from conrad.evaluation.spatial_experiments import common as c
from conrad.evaluation.spatial_experiments import e002_counterfactual
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp

SMALL: dict[str, dict[str, Any]] = {
    "scenario": {
        "family": "pipeline_with_supports",
        "sensor_overrides": {
            "DEPTH_RANGE": {"width_px": 32, "height_px": 24},
            "SONAR": {"n_beams": 32, "n_range_bins": 48},
        },
    },
    "conditions": {
        "style": "coverage",
        "target_coverage": 0.6,
        "n_views": 3,
        "modalities": ["DEPTH_RANGE", "SONAR"],
        "pose_sigma_m": 0.02,
        "pose_sigma_rad": 0.005,
        "n_azimuths": 6,
    },
}


@pytest.fixture(scope="module")
def loop():
    world = c.build_world(2026201, SMALL)
    twin = world.twin()
    targets = world.group("segments")
    cond = c.conditions(targets, SMALL["conditions"])
    sched, obs = c.observe(world, twin, cond, 2026201)
    pts = c.eval_points(twin, targets, 0.5, 0.25)
    return world, twin, sched, obs, pts


def test_model_consumes_only_observations_and_publishes_beliefs(loop):
    world, _, sched, obs, _ = loop
    m = Model2S(IdFactory(5), world.store)
    m.initialize({"sensors": world.sensors})
    evs = m.ingest_observations(obs)
    assert all(e.source_observation_id in {o.observation_id for o in obs} for e in evs)
    msgs = m.update_beliefs(stamp(sched.views[-1].time_s, "SIM"))
    assert msgs and m.diagnostics["integrated"] >= len(sched.views)
    assert m.diagnostics["duplicate_capture"] == len(sched.views)  # point clouds of the depth captures
    assert all(x.world_entity_id is None for x in msgs)  # no registry -> no simulator identity leaks in
    assert {x.knowledge_status for x in msgs} <= {
        KnowledgeStatus.MIXED,
        KnowledgeStatus.OBSERVED,
        KnowledgeStatus.UNKNOWN,
    }


def test_uahsm_is_right_where_it_is_confident_and_silent_where_hidden(loop):
    world, twin, sched, obs, pts = loop
    truth = c.truth_occupancy(twin, pts)
    seen = c.observed_mask(twin, sched, pts, 0.25)
    t_end = sched.views[-1].time_s
    ua = c.map_metrics(c.run_model("uahsm", world, obs, c.model_config({}), t_end), pts, truth, seen)
    ml = c.map_metrics(c.run_model("ml_fill", world, obs, c.model_config({}), t_end), pts, truth, seen)
    assert ua["hidden_uc_rate"] < 0.1 < ml["hidden_uc_rate"]
    assert ua["hidden_claimed_observed_rate"] <= ml["hidden_claimed_observed_rate"] + 0.2
    assert ua["confident_error_rate"] < ml["confident_error_rate"]


def test_free_space_query_agrees_with_truth_where_claimed_free(loop):
    world, twin, sched, obs, pts = loop
    m = c.run_model("uahsm", world, obs, c.model_config({}), sched.views[-1].time_s)
    free = m.is_free(pts)
    assert free.any()
    wrong = c.truth_occupancy(twin, pts[free]).mean()
    assert wrong < 0.05


def test_counterfactual_region_is_not_confidently_resolved():
    cfg = {
        **SMALL,
        "front_segments": 1,
        "kinds": ["bend_tail", "remove_tail", "hidden_object"],
        "conditions": {**SMALL["conditions"], "target_coverage": 0.8},
        "discriminating_conditions": SMALL["conditions"],
        "models": ["uahsm", "ml_fill"],
        "max_attempts": 4,
    }
    r = e002_counterfactual.run_seed(cfg, 2026203)  # a seed whose bend_tail pair verifies quickly
    assert r["pair_found"] == 1.0 and r["max_observation_difference"] <= 1e-3
    assert r["uahsm"]["differing"]["n_differing_cells"] > 0
    assert r["uahsm"]["differing"]["confident_claim_rate"] <= 0.05
    assert r["ml_fill"]["differing"]["confident_claim_rate"] > r["uahsm"]["differing"]["confident_claim_rate"]
    assert np.isfinite(r["uahsm"]["differing"]["mean_UO"])
