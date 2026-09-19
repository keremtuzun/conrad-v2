"""Gate I2 FORMAL path: estimator -> planner -> trajectory -> controller -> allocator -> safety -> CommandGateway
-> UnityRobotHardware, NAV-001..NAV-006 geometries as Unity colliders, on ESTIMATED state.

Success checks and tolerances are exactly those of ``configs/sim/nav_benchmarks.yaml``. Unity truth is read here
for metrics only; the navigation stack is constructed with the hardware adapter alone.
"""

from __future__ import annotations

import functools

import pytest
from unity_gate_support import GATE_RUNS, NAV_SEEDS, measured, reset_measured

from conrad.sim.mission.unity_run import NAV_UNITY_IDS, replay_unity_nav, run_unity_nav

GATE = "I2"
BIAS_Y_M = 0.5  # counterfactual USBL bias for the estimated-state check
BIAS_TOL_M = 0.15
_I2_FAIL = (
    "GATE I2 = FAIL (docs/audits/UNITY_INTEGRATION_I1_I3.md): NAV-005 station RMS 0.1534 m > 0.15 m on the "
    "held-out seed (python kernel 0.1574 m on the same seed)"
)


@functools.cache
def nav(benchmark: str, bias_y: float = 0.0) -> dict:
    tag = "" if bias_y == 0.0 else f"-fixbias{bias_y:+.2f}"
    return run_unity_nav(
        benchmark, NAV_SEEDS[benchmark], GATE_RUNS / GATE / f"{benchmark}{tag}", (0.0, bias_y, 0.0)
    )


@pytest.fixture(scope="module", autouse=True)
def _meta():
    reset_measured(
        GATE, {"scenario": "I2-UNITY-NAV", "seeds": NAV_SEEDS, "bundles": "artifacts/unity/gate_runs/I2"}
    )


def _keep(r: dict) -> dict:
    keys = (
        "success",
        "checks",
        "final_position_error_m",
        "path_rms_error_m",
        "estimation_rms_error_m",
        "collisions",
        "min_clearance_m",
        "time_to_goal_s",
        "standoff_rms_error_m",
        "station_rms_error_m",
        "pipeline_cross_track_rms_m",
        "gateway_accepted",
        "gateway_rejected",
        "position_fixes_accepted",
    )
    return {k: r[k] for k in keys if k in r}


@pytest.mark.parametrize(
    ("criterion", "benchmarks"),
    [
        ("reach waypoint", ("NAV-001", "NAV-002")),
        ("avoid obstacle", ("NAV-003",)),
        ("follow pipeline", ("NAV-004",)),
        pytest.param("station keep", ("NAV-005",), marks=pytest.mark.xfail(strict=True, reason=_I2_FAIL)),
    ],
)
def test_primitive(criterion, benchmarks):
    results = {b: nav(b) for b in benchmarks}
    measured(GATE, criterion, **{b: _keep(r) for b, r in results.items()})
    assert all(r["success"] for r in results.values()), {b: r["checks"] for b, r in results.items()}


@pytest.mark.xfail(strict=True, reason=_I2_FAIL)
def test_nav_001_to_006():
    results = {b: nav(b) for b in NAV_UNITY_IDS}
    measured(
        GATE,
        "NAV-001..NAV-006",
        **{b: _keep(r) for b, r in results.items()},
        passed=sum(r["success"] for r in results.values()),
    )
    assert all(r["success"] for r in results.values()), {b: r["checks"] for b, r in results.items()}


def test_uses_estimated_state():
    """The stack sees only RHI readings + the USBL-like fix. A biased fix moves the TRUE end point by the bias:
    control acts on the estimate, not on truth."""
    clean, biased = nav("NAV-001"), nav("NAV-001", BIAS_Y_M)
    shift = biased["final_true_position_m"][1] - clean["final_true_position_m"][1]
    est_err = {b: nav(b)["estimation_rms_error_m"] for b in NAV_UNITY_IDS}
    measured(
        GATE,
        "uses estimated state",
        estimation_rms_error_m=est_err,
        fix_bias_y_m=BIAS_Y_M,
        true_final_y_shift_m=shift,
        tolerance_m=BIAS_TOL_M,
        stack_inputs=clean["estimator_inputs"],
    )
    assert all(0.0 < e < 0.5 for e in est_err.values())
    assert abs(shift + BIAS_Y_M) <= BIAS_TOL_M


@pytest.mark.parametrize("benchmark", ["NAV-003", "NAV-006"])
def test_bundle_replays_deterministically(benchmark):
    nav(benchmark)
    rep = replay_unity_nav(GATE_RUNS / GATE / benchmark, GATE_RUNS / "replay" / GATE)
    measured(GATE, f"replay {benchmark}", **rep)
    assert rep["equal"], rep
