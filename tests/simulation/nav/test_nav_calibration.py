import json

from conrad.evaluation import dispatch
from conrad.evaluation.nav_benchmarks import calibration, run_benchmark

SCN = {
    "control_period_s": 0.02,
    "start": [0.0, 0.0, -5.0],
    "goal_target": [60.0, 0.0, -5.0],
    "outage_start_s": 6.0,
    "post_outage_s": 1.0,
    "fixes": {"period_s": 1.0, "sigma_m": 0.1},
    "current_speed_max_mps": 0.25,
    "gust_speed_max_mps": 0.15,
    "gust_duration_s": 10.0,
}


def test_nav007_loss_is_declared_before_the_error_outgrows_the_envelope():
    r = run_benchmark("NAV-007", 1)
    assert r["success"], r
    held = [e for e in r["safety_events"] if "LOCALIZATION_LOST" in e["reasons"]]
    assert held and held[0]["state"] == "HOLD"
    # legacy EST-B0 (seed 1): LOST only at t=39.1 s, max error 5.82 m while reporting sigma < 1.5 m
    assert held[0]["t_s"] < 30.0
    assert r["estimation_max_error_m"] < 4.0


def test_calibration_case_metrics_and_registration(tmp_path):
    assert dispatch.EXPERIMENTS["NAV-CAL-E001"][0] == "conrad.evaluation.nav_benchmarks.calibration"
    cfg = {
        "dev_seeds": [1],
        "seeds": [2],
        "variants": ["after_fix"],
        "outage_durations_s": [5],
        "imu_noise_multipliers": [5.0],
        "scenario": SCN,
        "metrics": {"sample_period_s": 0.5},
        "workers": 1,
    }
    doc = calibration.run(cfg, cfg["seeds"], tmp_path)
    agg = doc["summary"]["final/after_fix/outage5s/imu5x"]
    assert agg["n_seeds"] == 1 and agg["missed_danger_max_s"] == 0.0
    assert 0.0 <= agg["within_1sigma"] <= agg["within_2sigma"] <= agg["within_3sigma"] <= 1.0
    assert doc["seed_partitions"] == {"dev": [1], "final_held_out": [2]}
    assert json.loads((tmp_path / "summary.json").read_text())["experiment_id"] == "NAV-CAL-E001"


def test_seed_partitions_must_not_overlap(tmp_path):
    cfg = {
        "dev_seeds": [3],
        "outage_durations_s": [],
        "imu_noise_multipliers": [],
        "scenario": SCN,
        "metrics": {"sample_period_s": 0.5},
    }
    try:
        calibration.run(cfg, [3], tmp_path)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping partitions accepted")
