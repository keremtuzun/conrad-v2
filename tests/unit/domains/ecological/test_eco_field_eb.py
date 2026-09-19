"""Hierarchical empirical-Bayes field model (2E-E001 repair, docs/audits/MODEL2E_REPAIR.md)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from conrad.domains.ecological import FieldBeliefGrid, Model2EConfig, baseline_config

NS = 10**9
SENSOR_VAR = 0.2**2  # the stated turbidity sensor noise


def _stations(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack([rng.uniform(-38, 38, n), rng.uniform(-38, 38, n), rng.uniform(-19, -1, n)])


def _feed(fg, name, pos, truth, rng, noise_sd, steps=24, dt_s=900.0, var=SENSOR_VAR):
    for i in range(steps):
        for p in pos:
            fg.update_point(name, 0, p, float(truth(p) + rng.normal(0, noise_sd)), var, round(i * dt_s * NS))


def _hidden_rmse(fg, name, pos, truth):
    mean, _ = fg.moments_at_time(name, fg.fields[name].time_ns)
    hidden = np.ones(fg.grid.n_cells, dtype=bool)
    for p in pos:
        hidden[fg.grid.cell_of(p)] = False
    c = fg.grid.centers[hidden]
    return float(np.sqrt(np.mean((mean[0, hidden] - np.array([truth(x) for x in c])) ** 2)))


def test_error_does_not_grow_with_more_sensors_on_a_near_uniform_field():
    """The 2E-E001 defect: extra sensors used to make the turbidity field worse."""
    rng = np.random.default_rng(3)
    pos = _stations(8, rng)

    def truth(x):
        return 2.0 + 0.05 * np.sin(x[0] / 30.0)

    errs = []
    for k in (1, 2, 4, 8):
        fg = FieldBeliefGrid(Model2EConfig())
        _feed(fg, "turbidity", pos[:k], truth, np.random.default_rng(k), 0.15)
        errs.append(_hidden_rmse(fg, "turbidity", pos[:k], truth))
    assert errs[-1] <= errs[0] + 1e-9 and errs[3] < 0.1, errs
    assert all(b <= a + 0.02 for a, b in pairwise(errs)), errs


def test_tau2_shrinks_when_stations_agree_and_grows_when_they_differ():
    rng = np.random.default_rng(4)
    pos = _stations(6, rng)
    agree = FieldBeliefGrid(Model2EConfig())
    _feed(agree, "turbidity", pos, lambda x: 2.0, rng, 0.15, steps=6)
    spread = FieldBeliefGrid(Model2EConfig())
    offsets = {tuple(p): v for p, v in zip(pos, rng.normal(0, 1.5, len(pos)), strict=True)}
    _feed(spread, "turbidity", pos, lambda x: 5.0 + offsets[tuple(x)], rng, 0.15, steps=6)
    t_agree = agree.fields["turbidity"]
    t_spread = spread.fields["turbidity"]
    _ = t_agree.mean, t_spread.mean  # triggers the empirical-Bayes refresh
    assert t_spread.tau2[0] > 100 * t_agree.tau2[0]
    # the domain level is estimated from all stations
    assert t_agree.level[0] == pytest.approx(2.0, abs=0.1)


def test_noise_scale_and_drift_are_learned_from_the_readings():
    rng = np.random.default_rng(5)
    fg = FieldBeliefGrid(Model2EConfig())
    p = np.array([0.0, 0.0, -10.0])
    for i in range(200):  # constant truth, noise sd 0.1 while 0.2 is stated
        fg.update_point("turbidity", 0, p, 2.0 + rng.normal(0, 0.1), SENSOR_VAR, i * 900 * NS)
    fb = fg.fields["turbidity"]
    _ = fb.mean
    assert 0.1 < fb.noise_scale[0] < 0.5  # true ratio 0.25, shrunk toward 1 by the prior
    q_static = float(fb.q[0])
    walk = FieldBeliefGrid(Model2EConfig())
    x = 2.0
    for i in range(200):  # random walk with 0.2 NTU per step
        x += rng.normal(0, 0.2)
        walk.update_point("turbidity", 0, p, x + rng.normal(0, 0.1), SENSOR_VAR, i * 900 * NS)
    wb = walk.fields["turbidity"]
    _ = wb.mean
    assert wb.q[0] > 10 * max(q_static, 1e-12)
    assert wb.q[0] == pytest.approx(0.04 / 900, rel=0.5)


def test_depth_trend_extrapolates_stratification():
    rng = np.random.default_rng(6)
    pos = np.array([[-20.0, -20.0, -17.0], [20.0, 20.0, -3.0], [20.0, -20.0, -10.0]])

    def truth(x):
        return 15.0 + 0.1 * x[2]  # 0.1 degC per metre, warmer at the surface

    fg = FieldBeliefGrid(Model2EConfig())
    _feed(fg, "temperature", pos, truth, rng, 0.02, steps=8, var=0.05**2)
    assert _hidden_rmse(fg, "temperature", pos, truth) < 0.15
    flat = Model2EConfig().fields["temperature"].model_copy(update={"depth_trend_sd": 0.0})
    fg0 = FieldBeliefGrid(Model2EConfig(fields={**Model2EConfig().fields, "temperature": flat}))
    _feed(fg0, "temperature", pos, truth, np.random.default_rng(6), 0.02, steps=8, var=0.05**2)
    assert _hidden_rmse(fg, "temperature", pos, truth) < _hidden_rmse(fg0, "temperature", pos, truth)


def test_static_baseline_is_one_uniform_level_without_variance_growth():
    fg = FieldBeliefGrid(baseline_config("static_field"))
    fg.update_point("turbidity", 0, np.array([-30.0, -30.0, -10.0]), 1.0, SENSOR_VAR, 0)
    fg.update_point("turbidity", 0, np.array([30.0, 30.0, -10.0]), 3.0, SENSOR_VAR, NS)
    fb = fg.fields["turbidity"]
    mean, var = fg.moments_at_time("turbidity", NS)
    assert fb.tau2[0] == 0.0 and np.ptp(mean) < 1e-9 and mean[0, 0] == pytest.approx(2.0, abs=0.01)
    _, var_later = fg.moments_at_time("turbidity", 10**6 * NS)
    assert np.allclose(var, var_later)


def test_production_baseline_uses_the_production_switches():
    assert baseline_config("production").switches == Model2EConfig().switches


def _step_world(cfg, jump_at=12, jump=15.0, steps=36, seed=7):
    """Four stations, a uniform level of 2 NTU; station 0 reads every step, the others every 3rd step.
    At ``jump_at`` the whole field jumps by ``jump`` (the 2E-E003 spike, abruptly)."""
    rng = np.random.default_rng(seed)
    pos = _stations(4, rng)
    fg = FieldBeliefGrid(cfg)
    out = []
    for i in range(steps):
        level = 2.0 + (jump if i >= jump_at else 0.0)
        for s, p in enumerate(pos):
            if s == 0 or i % 3 == 0:
                fg.update_point("turbidity", 0, p, level + rng.normal(0, 0.1), SENSOR_VAR, i * 900 * NS)
        mean, var = fg.moments_at_time("turbidity", i * 900 * NS)
        out.append((level, mean[0], var[0]))
    return fg, out


def test_abrupt_change_is_detected_and_not_over_confident():
    """2E-E003-R2 regression: a 15 NTU step used to give mean z^2 ~ 35 (DEV-derived drift priors assume small
    deviations). With change-point intervention the jump is absorbed, stale stations carry its variance, and
    the variogram (drift q, noise scale) is not polluted by the jump."""
    no_cd = Model2EConfig(field_model={"change_detection": False})
    fg0, out0 = _step_world(no_cd)
    fg1, out1 = _step_world(Model2EConfig())

    def z2(out, lo, hi):
        return float(np.mean([np.mean((m - lv) ** 2 / v) for lv, m, v in out[lo:hi]]))

    assert z2(out0, 12, 20) > 20.0  # the defect is reproduced without the mechanism
    assert z2(out1, 12, 20) < 3.0
    # after the jump the belief tracks the new level
    lv, m, _ = out1[-1]
    assert np.max(np.abs(m - lv)) < 0.5
    fb1, fb0 = fg1.fields["turbidity"], fg0.fields["turbidity"]
    _ = fb1.mean, fb0.mean
    assert sum(len(s.changes[0]) for s in fb1.stations) >= 1
    assert fb1.q[0] < fb0.q[0] and fb1.noise_scale[0] < fb0.noise_scale[0]


def test_change_detection_is_quiet_on_a_stationary_field():
    """No change points (beyond the 3-sigma false-alarm rate) and no drift inflation without a change."""
    fg, out = _step_world(Model2EConfig(), jump=0.0, steps=48)
    fb = fg.fields["turbidity"]
    _ = fb.mean
    n_reads = sum(len(s.buf[0]) for s in fb.stations)
    n_changes = sum(len(s.changes[0]) for s in fb.stations)
    assert n_changes <= max(1, round(0.01 * n_reads))
    _, ref = _step_world(Model2EConfig(field_model={"change_detection": False}), jump=0.0, steps=48)
    rmse = [float(np.sqrt(np.mean([np.mean((m - lv) ** 2) for lv, m, _ in o[12:]]))) for o in (out, ref)]
    assert rmse[0] <= rmse[1] + 0.02


def test_static_baseline_never_detects_changes():
    fg, _ = _step_world(baseline_config("static_field"))
    assert all(not s.changes[0] and s.qa[0] == 0.0 for s in fg.fields["turbidity"].stations)
