"""Longer Twin2E runs: disturbance -> impact -> recovery, and determinism over a trajectory."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.twins.twin2e import Twin2E, build_small_scenario, load_config, validate_twin
from conrad.twins.twin2e.priors import KIND_BENTHIC
from conrad.twins.twin2e.validators import state_digest

REPO = Path(__file__).resolve().parents[3]
CFG = load_config(REPO / "configs" / "sim" / "twin2e_test_small.yaml")


def _twin(seed, events=()):
    ids = IdFactory(seed)
    scen = build_small_scenario(ids, seed, CFG, events=events)
    tw = Twin2E(IdFactory(seed + 1), ObjectStore(tempfile.mkdtemp(prefix="twin2e-sim-")), CFG)
    tw.initialize(scen)
    return tw


def _benthic(tw):
    return next(e for e in tw.entities.values() if e.kind == KIND_BENTHIC)


def test_heatwave_stress_then_recovery():
    # 3-day +10 C anomaly, then normal conditions. time_scale=50 in the test config.
    tw = _twin(31, events=((0.0, "TEMPERATURE_ANOMALY", None, {"delta_c": 10.0, "duration_s": 6 * 3600.0}),))
    e = _benthic(tw)
    e.params["thermal_optimum_c"] = float(tw.local_env(e)["temperature"])  # no pre-existing stress
    cond = [e.condition]
    for _ in range(40):
        tw.step(1800.0)
        cond.append(e.condition)
        assert validate_twin(tw) == []
    trough = int(np.argmin(cond))
    assert cond[trough] < cond[0] - 0.02, "anomaly should stress the patch"
    assert cond[-1] > cond[trough] + 0.01, "condition should recover once the anomaly ends"
    assert tw.event_log[0]["status"] == "APPLIED"


def test_trajectory_determinism_and_seed_sensitivity():
    def run(seed):
        tw = _twin(seed)
        digests = []
        for dt in (60.0, 900.0, 1800.0, 30.0):
            tw.step(dt)
            digests.append(state_digest(tw))
        return digests

    assert run(41) == run(41)
    assert run(41)[-1] != run(42)[-1]
