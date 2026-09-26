"""The Spatial V1.1 final verifier derives the gate from rows and fails closed."""

from copy import deepcopy

from scripts.verify_i5_spatial_v1_1_final import (
    EMPTY_SHA256,
    _canonical_sha256,
    checkpoint_from_manifest,
    verify,
)


def _fixture() -> tuple[dict, dict, dict]:
    scenarios = ["I5-NOMINAL", "I5-CRITICAL-FINDING"]
    arms = ["egdc_structured", "rule_fsm", "naive_act_on_claims"]
    config = {
        "experiment_id": "TEST-FINAL",
        "partition_domain": "i5_mission_v10",
        "seeds": [1],
        "scenarios": scenarios,
        "arms": arms,
        "final_rule": {
            "expected_rows": 6,
            "primary_nominal_warrants_min": 1,
            "each_scenario_warrants_min": 1,
            "each_scenario_correct_given_warrant_rate_min": 0.9,
            "false_intact_on_covered_resolvable_defect_max": 0,
            "decision_violations_max": 0,
            "nominal_over_escalations_max": 0,
            "traceability_min": 1.0,
            "uir_max": 0.0,
        },
    }
    rows = []
    for scenario in scenarios:
        for arm in arms:
            rows.append(
                {
                    "seed": 1,
                    "scenario": scenario,
                    "arm": arm,
                    "driving": {
                        "warrant_reached": True,
                        "correct": arm != "naive_act_on_claims",
                        "violations_total": 0 if arm != "naive_act_on_claims" else 1,
                        "over_escalations": 0,
                        "n_decisions": 1,
                        "traceable_decisions": 1,
                        "uir": {"relied_world_claims": 1, "relied_unsupported_claims": 0},
                    },
                    "outcome": {
                        "task_success": arm != "naive_act_on_claims",
                        "safety_events": 0 if arm != "naive_act_on_claims" else 1,
                    },
                    "spatial_v1": {"false_intact_on_covered_resolvable_defect": False},
                }
            )
    source = {"git_commit": "a" * 40, "source_diff_sha256": EMPTY_SHA256}
    declaration = {
        "experiment_id": config["experiment_id"],
        "partition": "final_test",
        "partition_domain": config["partition_domain"],
        "seeds": config["seeds"],
        "scenarios": scenarios,
        "arms": arms,
        "config": config,
        "source": source,
    }
    declaration_sha256 = _canonical_sha256(declaration)
    checkpoint = {
        "declaration_sha256": declaration_sha256,
        "declaration": declaration,
        "completed_rows": deepcopy(rows),
    }
    result = {
        "experiment_id": config["experiment_id"],
        "evidence_class": "SURROGATE (python, not Unity)",
        "data_status": "SYNTHETIC_ONLY",
        "partition": "final_test",
        "partition_domain": config["partition_domain"],
        "seeds": config["seeds"],
        "scenarios": {scenario: {} for scenario in scenarios},
        "arms": arms,
        "config": config,
        "checkpoint": {"declaration_sha256": declaration_sha256},
        "per_run": rows,
    }
    return config, result, checkpoint


def test_final_verifier_accepts_complete_raw_passing_evidence() -> None:
    config, result, checkpoint = _fixture()
    report = verify(config, result, checkpoint, expected_source_commit="a" * 40)
    assert report["status"] == "PASS"
    assert all(report["checks"].values())


def test_final_verifier_accepts_compact_checkpoint_manifest() -> None:
    config, result, checkpoint = _fixture()
    manifest = {
        "checkpoint_sha256": "c" * 64,
        "declaration_sha256": checkpoint["declaration_sha256"],
        "completed_rows": len(result["per_run"]),
        "completed_rows_canonical_sha256": _canonical_sha256(result["per_run"]),
        "source": checkpoint["declaration"]["source"],
    }
    reconstructed = checkpoint_from_manifest(result, manifest)
    report = verify(
        config,
        result,
        reconstructed,
        expected_source_commit="a" * 40,
        checkpoint_manifest=manifest,
    )
    assert report["status"] == "PASS"
    assert report["checks"]["checkpoint_manifest_rows_exact"]


def test_final_verifier_refuses_aggregate_claim_when_raw_row_fails() -> None:
    config, result, checkpoint = _fixture()
    broken = deepcopy(result)
    broken["per_run"][0]["driving"]["correct"] = False
    report = verify(config, broken, checkpoint, expected_source_commit="a" * 40)
    assert report["status"] == "FAIL"
    assert not report["checks"]["checkpoint_rows_exact"]
    assert not report["checks"]["each_scenario_correct_given_warrant"]


def test_final_verifier_refuses_duplicate_row_and_dirty_source() -> None:
    config, result, checkpoint = _fixture()
    broken = deepcopy(result)
    broken["per_run"][-1] = deepcopy(broken["per_run"][0])
    dirty_checkpoint = deepcopy(checkpoint)
    dirty_checkpoint["declaration"]["source"]["source_diff_sha256"] = "b" * 64
    dirty_checkpoint["declaration_sha256"] = _canonical_sha256(dirty_checkpoint["declaration"])
    broken["checkpoint"]["declaration_sha256"] = dirty_checkpoint["declaration_sha256"]
    report = verify(config, broken, dirty_checkpoint, expected_source_commit="a" * 40)
    assert report["status"] == "FAIL"
    assert not report["checks"]["complete_unique_grid"]
    assert not report["checks"]["source_clean"]
