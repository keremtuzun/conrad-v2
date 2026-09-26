"""Selection verification is fail closed on missing or duplicated rows."""

from copy import deepcopy

from scripts.verify_i5_spatial_v1_1_selection import verify


def _fixture() -> tuple[dict, dict]:
    scenarios = ["I5-NOMINAL", "I5-CRITICAL-FINDING"]
    arms = ["egdc_structured", "rule_fsm"]
    config = {
        "partition_domain": "i5_mission_v10",
        "seeds": [1],
        "scenarios": scenarios,
        "arms": arms,
        "selection_rule": {
            "primary_nominal_warrants_min": 1,
            "non_nominal_warrants_per_scenario_min": 1,
            "non_nominal_correct_given_warrant_rate_min": 0.9,
            "false_intact_on_covered_resolvable_defect_max": 0,
            "decision_violations_max": 0,
            "nominal_over_escalations_max": 0,
            "traceability_min": 1.0,
            "uir_max": 0.0,
        },
    }
    rows = [
        {
            "seed": 1,
            "scenario": scenario,
            "arm": arm,
            "spatial_v1": {"false_intact_on_covered_resolvable_defect": False},
        }
        for scenario in scenarios
        for arm in arms
    ]
    result = {
        "config": config,
        "partition": "validation",
        "partition_domain": "i5_mission_v10",
        "per_run": rows,
        "verdicts": {
            "per_scenario_warrant_reached": dict.fromkeys(scenarios, 1),
            "per_scenario_correct_given_warrant_rate": dict.fromkeys(scenarios, 1.0),
            "violations_total": 0,
            "nominal_over_escalations": 0,
            "traceable_fraction": 1.0,
            "uir": 0.0,
            "competitive_outcomes": True,
            "competitive": {},
            "spatial_v1_runtime_selected": True,
        },
    }
    return config, result


def test_selection_verifier_accepts_complete_passing_grid() -> None:
    config, result = _fixture()
    report = verify(config, result)
    assert report["status"] == "PASS"
    assert all(report["checks"].values())


def test_selection_verifier_refuses_duplicate_and_missing_row() -> None:
    config, result = _fixture()
    broken = deepcopy(result)
    broken["per_run"][-1] = deepcopy(broken["per_run"][0])
    report = verify(config, broken)
    assert report["status"] == "FAIL"
    assert not report["checks"]["complete_unique_grid"]
