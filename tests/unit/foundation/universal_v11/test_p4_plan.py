from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
PLAN = ROOT / "configs" / "train" / "osfm" / "v11_p4_formal_training_plan.yaml"
BENCHMARK = ROOT / "configs" / "train" / "osfm" / "v11_p4_8l4_parallel_benchmark.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_v11_p4_plan_has_exact_step_total_and_budget_gates() -> None:
    plan = _load(PLAN)
    total = sum(int(stage["optimizer_steps"]) for stage in plan["step_allocation"])

    assert total == plan["training_budget"]["total_optimizer_steps"] == 800000
    assert plan["compute_target"]["gpu_count"] == 8
    assert plan["compute_target"]["accelerator"] == "NVIDIA_L4"
    assert plan["compute_target"]["max_wall_clock_hours"] == 24.0
    assert plan["compute_target"]["max_total_cost_try"] == 8500.0
    assert plan["source_constraints"]["min_effective_rank"] == 75.0
    assert plan["source_constraints"]["registered_modality_count"] == 829
    assert plan["formal_training_allowed"] is False
    assert plan["full_run_allowed"] is False


def test_v11_p4_plan_requires_parallel_benchmark_before_training() -> None:
    plan = _load(PLAN)
    blockers = set(plan["do_not_start_if"])
    required = {
        "8_l4_benchmark_missing",
        "measured_projection_over_24h",
        "measured_projection_over_8500_try",
        "rank_below_75",
        "registry_coverage_incomplete",
        "p4_v1_line_dirty_or_modified_by_v11_run",
    }

    assert required <= blockers
    assert any("V1.1-P4-8L4-PARALLEL-BENCHMARK" in item for item in plan["may_start_after"])


def test_v11_p4_parallel_benchmark_is_bounded_and_fail_closed() -> None:
    benchmark = _load(BENCHMARK)
    fail_closed = set(benchmark["fail_closed_on"])

    assert benchmark["formal_training_allowed"] is False
    assert benchmark["full_run_allowed"] is False
    assert benchmark["pilot"]["optimizer_steps"] == 1000
    assert benchmark["required_compute"]["gpu_count"] == 8
    assert benchmark["budget_guard"]["max_wall_clock_hours"] == 24.0
    assert benchmark["budget_guard"]["max_total_cost_try"] == 8500.0
    assert benchmark["measured_go_rules"]["min_effective_rank"] == 75.0
    assert benchmark["measured_go_rules"]["projection_must_fit"]["steps"] == "conservative"
    assert {"rank_below_75", "projected_runtime_over_24h", "projected_cost_over_8500_try"} <= fail_closed
