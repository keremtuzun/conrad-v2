"""T0 REALISTIC sensor model (docs/audits/STRUCTURAL_LINEAGE_AUDIT.md): crack sizing is crude, partial views
under-size, detection is probabilistic, and repeated looks from one sensor cannot average the error away."""

from __future__ import annotations

import numpy as np
import pytest

from conrad.twins.twin2t.config import ObservationConfig
from conrad.twins.twin2t.observation import ObservationModel, SensingQuality
from tests.unit.twins.twin2t.t2t_unit_helpers import new_ids, runtime

CLEAR = SensingQuality.from_degradation({})
TURBID = SensingQuality.from_degradation({"turbidity": 1.0, "biofouling_cover": 0.5})


def readings(model, rt, n, *, q=CLEAR, vis=1.0, sensor=None, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([model.t0(rt, q, rng, visibility=vis, sensor_id=sensor).values for _ in range(n)])


def test_config_rejects_unknown_measurement_model():
    with pytest.raises(ValueError):
        ObservationConfig(measurement_model="PERFECT")
    with pytest.raises(ValueError):
        ObservationConfig(crack_pod_a50_m=0.0)


def test_crack_sizing_error_is_multiplicative_not_a_tiny_additive_term():
    (eid,) = new_ids(1)
    rt = runtime(eid, crack=0.08)
    vals = readings(ObservationModel(ObservationConfig(), 3), rt, 400, sensor=new_ids(1, 9)[0])[:, 2]
    det = vals[vals > 0.01]
    assert np.std(det) > 0.1 * np.mean(det)  # >= 10 % scatter at 80 mm (the old model gave ~1.2 %)
    legacy = readings(ObservationModel(ObservationConfig(measurement_model="IDEALISED_ADDITIVE"), 3), rt, 400)
    assert abs(np.mean(legacy[:, 2]) - 0.08) < 5e-4 and np.std(legacy[:, 2]) < 2e-3


def test_repeated_looks_do_not_average_to_truth_because_the_sizing_bias_persists():
    (eid,) = new_ids(1)
    rt = runtime(eid, crack=0.08)
    model = ObservationModel(ObservationConfig(), 11)
    per_sensor = [
        np.median(readings(model, rt, 200, sensor=s, seed=i)[:, 2]) for i, s in enumerate(new_ids(20, 8))
    ]
    # each sensor's median converges to its own biased value, and those values disagree with each other
    assert np.std(per_sensor) > 0.1 * np.mean(per_sensor)
    # the bias is drawn once per (component, sensor) pair and then fixed
    s0 = new_ids(1, 8)[0]
    assert model.persistent_bias(eid, s0) == model.persistent_bias(eid, s0)


def test_partial_view_undersizes_and_turbid_water_misses_small_cracks():
    (eid,) = new_ids(1)
    model = ObservationModel(ObservationConfig(), 5)
    s = new_ids(1, 3)[0]
    rt = runtime(eid, crack=0.08)
    full = np.median(readings(model, rt, 300, vis=1.0, sensor=s)[:, 2])
    part = np.median(readings(model, rt, 300, vis=0.35, sensor=s)[:, 2])
    assert part < 0.75 * full
    assert model.crack_pod(0.003, 1.0) < 0.1 < 0.95 < model.crack_pod(0.08, 1.0)
    assert model.crack_pod(0.015, model.noise_gain(TURBID)) < model.crack_pod(0.015, 1.0)
    small = runtime(eid, crack=0.004)
    hits = readings(model, small, 400, sensor=s)[:, 2] > 0.003
    assert hits.mean() < 0.3  # a 4 mm crack is usually not called by a visual payload


def test_wall_loss_has_relative_error_that_grows_with_depth():
    (a, b) = new_ids(2)
    model = ObservationModel(ObservationConfig(), 2)
    thin = readings(model, runtime(a, corrosion=0.001), 400)[:, 0]
    deep = readings(model, runtime(b, corrosion=0.008), 400)[:, 0]
    assert np.std(deep) > 2.0 * np.std(thin)
