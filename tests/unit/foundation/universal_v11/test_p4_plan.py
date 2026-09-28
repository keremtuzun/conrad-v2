from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
PLAN = ROOT / "configs" / "train" / "osfm" / "v11_p4_formal_training_plan.yaml"
BENCHMARK = ROOT / "configs" / "train" / "osfm" / "v11_p4_8l4_parallel_benchmark.yaml"
TEN_P = ROOT / "configs" / "train" / "osfm" / "v11_p4_10p_829_semantic_candidate.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_v11_p4_plan_has_exact_step_total_and_budget_gates() -> None:
    plan = _load(PLAN)
    total = sum(int(stage["optimizer_steps"]) for stage in plan["step_allocation"])

    assert total == plan["training_budget"]["total_optimizer_steps"] == 3_600_000
    assert plan["summary"]["budget_capped_candidate_steps"] == 800000
    assert plan["summary"]["old_20p_candidate_replaced"] is True
    assert plan["summary"]["optimal_10p_829_semantic_candidate_steps"] == 1_200_000
    assert plan["summary"]["full_v11_steps"] == 3_600_000
    assert plan["phase_budget_policy"]["do_not_compress_full_plan_to_fit_one_window"] is True
    assert plan["compute_target"]["gpu_count_minimum"] == 8
    assert plan["compute_target"]["accelerator"] == "NVIDIA_L4"
    assert plan["phase_budget_policy"]["launch_window_hours"] == 24.0
    assert plan["phase_budget_policy"]["user_budget_cap_try_per_launch_window"] == 8500.0
    assert plan["source_constraints"]["min_effective_rank"] == 75.0
    assert plan["source_constraints"]["registered_modality_count"] == 829
    assert len(plan["families"]) == 15
    assert plan["formal_training_allowed"] is False
    assert plan["full_run_allowed"] is False


def test_v11_p4_plan_requires_parallel_benchmark_before_training() -> None:
    plan = _load(PLAN)
    blockers = set(plan["do_not_start_if"])
    required = {
        "8_l4_benchmark_missing",
        "measured_phase_projection_missing",
        "current_phase_projection_over_approved_budget",
        "current_phase_projection_over_approved_time_window",
        "rank_below_75",
        "registry_coverage_incomplete",
        "p4_v1_line_dirty_or_modified_by_v11_run",
        "data_manifest_missing_for_claimed_active_modality",
    }

    assert required <= blockers
    assert any("V1.1-P4-8L4-PARALLEL-BENCHMARK" in item for item in plan["may_start_after"])


def test_v11_p4_full_plan_does_not_claim_all_registry_items_are_semantically_trained() -> None:
    plan = _load(PLAN)
    stages = {stage["stage"]: stage for stage in plan["step_allocation"]}

    assert (
        plan["source_constraints"]["registered_modality_coverage_semantics"]
        == "registry_interface_coverage_is_not_semantic_pretraining"
    )
    assert (
        stages["V11-P4-C-active-and-high-priority-modality-curriculum"]["coverage_policy"]
        == "active_or_data_backed_modalities_only"
    )
    assert (
        stages["V11-P4-C-active-and-high-priority-modality-curriculum"]["gates"][
            "no_claim_that_all_829_modalities_are_semantically_pretrained_without_data"
        ]
        is True
    )


def test_v11_10p_829_semantic_candidate_replaces_20p_and_is_in_range() -> None:
    plan = _load(TEN_P)
    total = sum(int(stage["optimizer_steps"]) for stage in plan["step_allocation"])

    assert plan["plan_id"] == "OSFM-UNIVERSAL-V1.1-10P-829-SEMANTIC"
    assert plan["replaces"] == "configs/train/osfm/v11_p4_20p_candidate.yaml"
    assert total == plan["training_budget"]["total_optimizer_steps"] == 1_200_000
    assert plan["source_constraints"]["candidate_steps_min"] == 700000
    assert plan["source_constraints"]["candidate_steps_selected"] == 1_200_000
    assert plan["source_constraints"]["candidate_steps_max"] == 1_600_000
    assert plan["parent_full_semantic_plan"]["min_optimizer_steps"] == 7_000_000
    assert plan["parent_full_semantic_plan"]["max_optimizer_steps"] == 16_000_000
    assert plan["source_constraints"]["compression_policy"] == "weighted_semantic_curriculum_not_flat_registry_fraction"
    assert plan["promotion"]["may_be_called_full_v11"] is False
    assert plan["promotion"]["may_be_called_full_829_semantic_v11"] is False
    assert plan["promotion"]["may_be_used_as_v11_candidate_checkpoint"] is True
    assert len(plan["family_priority"]["high"]) == 7
    assert len(plan["family_priority"]["medium"]) == 4
    assert len(plan["family_priority"]["light"]) == 4


def test_v11_10p_829_semantic_candidate_protects_evidence_and_semantic_claims() -> None:
    plan = _load(TEN_P)
    blockers = set(plan["do_not_start_if"])
    stages = {stage["stage"]: stage for stage in plan["step_allocation"]}

    assert plan["source_constraints"]["learned_representations_may_not_replace_exact_evidence"] is True
    assert plan["source_constraints"]["registry_interface_coverage_not_semantic_pretraining"] is True
    assert plan["source_constraints"]["semantic_training_claim_requires_data_manifest"] is True
    assert (
        stages["V11-10P-C-active-modality-semantic-curriculum"]["coverage_policy"]
        == "data_backed_modalities_only_for_semantic_claims"
    )
    assert (
        stages["V11-10P-C-active-modality-semantic-curriculum"]["gates"][
            "no_claim_that_all_829_modalities_are_semantically_pretrained_without_data"
        ]
        is True
    )
    assert {
        "8_l4_benchmark_missing",
        "measured_projection_over_24h",
        "measured_projection_over_8500_try",
        "rank_below_75",
        "p4_v1_line_dirty_or_modified_by_v11_run",
        "semantic_data_manifest_missing_for_claimed_modality",
    } <= blockers


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
