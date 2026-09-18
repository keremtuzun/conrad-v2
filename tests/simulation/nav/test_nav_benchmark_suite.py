import json

import pytest

from conrad.evaluation.nav_benchmarks import BENCHMARK_IDS, load_scenario, run_benchmark


def test_suite_is_locked_to_eight_ids():
    assert tuple(f"NAV-00{i}" for i in range(1, 9)) == BENCHMARK_IDS
    for b in BENCHMARK_IDS:
        assert load_scenario(b)["description"]
    with pytest.raises(KeyError):
        load_scenario("NAV-999")


def test_nav001_succeeds_and_is_deterministic(tmp_path):
    a = run_benchmark("NAV-001", 11, tmp_path)
    b = run_benchmark("NAV-001", 11)
    assert a["success"], a
    assert a["simulation_validity_level"] == "L1"
    assert a["gateway_rejected"] == 0 and a["collisions"] == 0
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert (tmp_path / "NAV-001_seed11.json").exists()


def test_nav008_thruster_failure_degrades_gracefully():
    r = run_benchmark("NAV-008", 3)
    assert r["success"], r
    assert r["faults_injected"] == ["THRUSTER_FAILURE"]
    assert any("DEGRADED_MANEUVERABILITY" in e["reasons"] for e in r["safety_events"])
