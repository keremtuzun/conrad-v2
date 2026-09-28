from conrad.foundation.universal_v11.benchmark import run_v11_pillar_benchmark


def test_v11_pillar_benchmark_fast_probe_passes_and_does_not_launch_training() -> None:
    report = run_v11_pillar_benchmark(
        {
            "fast_probe": True,
            "batch_size": 96,
            "rank_floor": 8.0,
            "rank_target": 8.0,
            "max_hours": 24.0,
            "max_budget_tl": 9000.0,
        }
    )
    assert report["status"] == "VALIDATED-RUN"
    assert report["formal_training_launched"] is False
    assert report["cloud_budget"]["gpu_count"] == 8
    assert report["decision"] == "GO"
    assert len(report["probes"]) >= 8


def test_v11_pillar_benchmark_fails_closed_on_too_high_rank_floor() -> None:
    report = run_v11_pillar_benchmark(
        {
            "fast_probe": True,
            "batch_size": 16,
            "rank_floor": 1_000.0,
            "rank_target": 1_000.0,
        }
    )
    assert report["decision"] == "NO-GO"
