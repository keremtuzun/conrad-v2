from conrad.foundation.universal_v11.benchmark import run_v11_pillar_benchmark, run_v11_training_microbenchmark


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


def test_v11_pillar_benchmark_records_one_l4_fallback_budget() -> None:
    report = run_v11_pillar_benchmark(
        {
            "fast_probe": True,
            "batch_size": 16,
            "rank_floor": 8.0,
            "rank_target": 8.0,
            "gpu_count": 1,
            "max_hours": 72.0,
            "max_budget_tl": 8000.0,
        }
    )
    assert report["formal_training_launched"] is False
    assert report["cloud_budget"]["gpu_count"] == 1
    assert report["cloud_budget"]["max_hours"] == 72.0
    assert report["cloud_budget"]["max_budget_tl"] == 8000.0


def test_v11_training_microbenchmark_projects_one_l4_fallback_without_promoting() -> None:
    report = run_v11_training_microbenchmark(
        {
            "optimizer_steps": 2,
            "warmup_discard_steps": 0,
            "batch_size": 1,
            "family_depth": 1,
            "rank_floor": 1.0,
            "target_steps": 4,
            "max_hours": 72.0,
            "max_budget_tl": 8000.0,
            "hourly_cost_tl": 50.0,
            "output_dir": "artifacts/gates/V1.1/test_microbenchmark",
        }
    )
    assert report["status"] == "VALIDATED-RUN"
    assert report["formal_training_launched"] is False
    assert report["successful_steps"] == 2
    assert report["routeable_family_count"] == 15
    assert report["checkpoint_reload_matches"] is True
    assert report["projected_hours"] <= 72.0
