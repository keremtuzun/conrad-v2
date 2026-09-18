from __future__ import annotations

import math

import numpy as np
import pytest
from t2t_unit_helpers import YEAR, new_ids, params, runtime

from conrad.twins.twin2t.mechanisms import (
    CorrosionModel,
    coating_breakdown,
    coating_step,
    corrosion_curve,
    corrosion_step,
    environment_modifier,
    fatigue_step,
    paris_integrate,
    stress_intensity_range,
)
from conrad.twins.twin2t.state import Environment

RNG = np.random.default_rng(0)


def _corr(rt, dt, model=CorrosionModel.POWER_LAW, exposed=1.0, stochastic=False, env=None):
    return corrosion_step(
        rt.state,
        rt.params,
        env or rt.environment,
        exposed,
        dt,
        RNG,
        model,
        material_factor=1.0,
        stochastic=stochastic,
        environment_conditioned=True,
    )


def test_power_law_matches_closed_form():
    rt = runtime(new_ids(1)[0])
    res = _corr(rt, 4 * YEAR)
    assert res.state.corrosion_depth_m == pytest.approx(1.0e-4 * 4**0.6, rel=1e-12)
    # incremental stepping reproduces the curve (exposure time carried in state)
    s = rt.state
    for _ in range(4):
        rt.state = _corr(rt, YEAR).state
    assert rt.state.corrosion_depth_m == pytest.approx(1.0e-4 * 4**0.6, rel=1e-9)
    assert s.corrosion_depth_m == 0.0


def test_bilinear_form():
    p = params()
    assert corrosion_curve(CorrosionModel.BILINEAR, 1.0, p) == pytest.approx(1.0e-4)
    assert corrosion_curve(CorrosionModel.BILINEAR, 5.0, p) == pytest.approx(2.0e-4 + 3 * 5.0e-5)


def test_environment_modifiers():
    p = params()
    base = environment_modifier(Environment(), p)
    assert base == pytest.approx(1.0)
    assert environment_modifier(Environment(temperature_c=20.0), p) == pytest.approx(1.5)
    assert environment_modifier(Environment(cp_active=True), p) == pytest.approx(0.05)
    assert environment_modifier(Environment(buried=True), p) == pytest.approx(0.5)
    assert environment_modifier(Environment(dissolved_oxygen_mg_l=3.5), p) == pytest.approx(0.5)
    assert environment_modifier(Environment(cp_active=True), p, conditioned=False) == 1.0


def test_no_exposure_no_corrosion_and_bounded_by_wall():
    rt = runtime(new_ids(1)[0])
    assert _corr(rt, YEAR, exposed=0.0).state.corrosion_depth_m == 0.0
    rt.state = rt.state.with_(corrosion_depth_m=0.01199)
    out = _corr(rt, 50 * YEAR, stochastic=True)
    assert out.state.corrosion_depth_m <= rt.params.wall_thickness_m


def test_coating_breakdown_dnv_linear_and_clipped():
    p = params()
    assert coating_breakdown(0.0, p) == pytest.approx(0.05)
    assert coating_breakdown(10.0, p) == pytest.approx(0.25)
    assert coating_breakdown(1000.0, p) == 1.0
    rt = runtime(new_ids(1)[0], coated=True, coating=0.05)
    s = coating_step(rt.state, rt.params, 10 * YEAR)
    assert s.coating_breakdown_fraction == pytest.approx(0.25)


def test_paris_closed_form_matches_numeric_integration():
    a0, n, c, m, y, ds = 5e-3, 2.0e5, 1e-11, 3.0, 1.12, 60.0
    closed = paris_integrate(a0, n, c, m, y, ds)
    a = a0
    for _ in range(20000):
        a += c * (y * ds * math.sqrt(math.pi * a)) ** m * (n / 20000)
    assert closed == pytest.approx(a, rel=1e-3)
    assert closed > a0


def test_fatigue_threshold_and_bounds():
    eid = new_ids(1)[0]
    rt = runtime(eid, crack=1e-3, stress=20e6)  # dK = 1.12*20*sqrt(pi*1e-3) ~ 1.26 < 3
    assert stress_intensity_range(1e-3, 20e6, 1.12) < 3.0
    fr = fatigue_step(rt.state, rt.params, 20e6, 0.1, YEAR, RNG, stochastic=True, max_crack_length_m=1.0)
    assert not fr.grew and fr.state.crack_length_m == 1e-3
    rt2 = runtime(eid, crack=5e-3, stress=80e6)
    fr2 = fatigue_step(rt2.state, rt2.params, 80e6, 0.1, YEAR, RNG, stochastic=False, max_crack_length_m=1.0)
    assert fr2.grew and fr2.delta_k >= fr2.threshold
    assert fr2.state.crack_depth_m <= rt2.params.wall_thickness_m
    none = runtime(eid, crack=0.0, stress=200e6)
    assert not fatigue_step(
        none.state, none.params, 200e6, 1.0, YEAR, RNG, stochastic=True, max_crack_length_m=1.0
    ).grew
