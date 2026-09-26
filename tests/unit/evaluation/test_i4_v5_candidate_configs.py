from __future__ import annotations

from pathlib import Path

import yaml

from conrad.evaluation.dispatch import EXPERIMENTS


ROOT = Path(__file__).resolve().parents[3]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "eval" / name).read_text(encoding="utf-8"))


def test_i4_v5_experiments_are_registered_and_not_final_test() -> None:
    assert EXPERIMENTS["ACTIVE-MCBR-E008-DEV"][1] == "configs/eval/i4_v5_development.yaml"
    assert EXPERIMENTS["ACTIVE-MCBR-E008-VAL"][1] == "configs/eval/i4_v5_validation.yaml"
    dev = _load("i4_v5_development.yaml")
    val = _load("i4_v5_validation.yaml")
    assert dev["partition"] == "development"
    assert dev["purpose"] == "design"
    assert val["partition"] == "validation"
    assert val["purpose"] == "selection"
    assert dev["partition"] != "final_test"
    assert val["partition"] != "final_test"


def test_i4_v5_candidate_targets_failed_resource_metric_without_changing_production() -> None:
    dev = _load("i4_v5_development.yaml")
    v5 = dev["arms"]["V5_efficiency_candidate"]
    assert dev["primary_metric"] == "info_per_kj"
    assert dev["baseline"] == "A-B0_random"
    assert v5["planner"] == "V4"
    assert v5["v4"]["ranker"] == {"kind": "bayes_eig", "cost_mode": "ratio", "cost_weight": 1.25}
    assert v5["view_execution"]["route_aware_navigation_cost"] is True
    assert v5["view_execution"]["budget_from_mission_duration"] is True


def test_i4_v6_development_screen_is_registered_and_keeps_held_out_splits_closed() -> None:
    assert EXPERIMENTS["ACTIVE-MCBR-E009-DEV"][1] == "configs/eval/i4_v6_development.yaml"
    dev = _load("i4_v6_development.yaml")
    assert dev["partition"] == "development"
    assert dev["purpose"] == "design"
    assert dev["partition"] != "validation"
    assert dev["partition"] != "final_test"
    assert dev["primary_metric"] == "info_per_kj"
    assert dev["baseline"] == "A-B0_random"
    assert dev["reference_arms"] == ["V4_protocol_only"]
    assert "V6_route_cost_only" in dev["arms"]
    assert "V6_deadline_only" in dev["arms"]
    assert "V6_stop_low_value" in dev["arms"]


def test_i4_v7_stop_threshold_screen_is_registered_and_development_only() -> None:
    assert EXPERIMENTS["ACTIVE-MCBR-E010-DEV"][1] == "configs/eval/i4_v7_stop_threshold_development.yaml"
    dev = _load("i4_v7_stop_threshold_development.yaml")
    assert dev["partition"] == "development"
    assert dev["purpose"] == "design"
    assert dev["primary_metric"] == "info_per_kj"
    assert dev["baseline"] == "A-B0_random"
    assert dev["reference_arms"] == ["V4_protocol_only"]
    assert dev["partition"] != "validation"
    assert dev["partition"] != "final_test"
    assert all(name.startswith(("A-B0_", "V4_", "V7_")) for name in dev["arms"])


def test_i4_v8_ranker_family_screen_is_registered_and_development_only() -> None:
    assert EXPERIMENTS["ACTIVE-MCBR-E011-DEV"][1] == "configs/eval/i4_v8_ranker_family_development.yaml"
    dev = _load("i4_v8_ranker_family_development.yaml")
    assert dev["partition"] == "development"
    assert dev["purpose"] == "design"
    assert dev["primary_metric"] == "info_per_kj"
    assert dev["baseline"] == "A-B0_random"
    assert dev["reference_arms"] == ["V4_protocol_only"]
    assert dev["partition"] != "validation"
    assert dev["partition"] != "final_test"
    assert dev["partition"] != "ood_test"
    assert dev["arms"]["V8_mission_ratio"]["v4"]["ranker"]["kind"] == "mission_conditioned"
    assert dev["arms"]["V8_hypothesis_ratio"]["v4"]["ranker"]["kind"] == "hypothesis_discrimination"


def test_i4_v8_validation_screen_is_registered_and_keeps_final_splits_closed() -> None:
    assert EXPERIMENTS["ACTIVE-MCBR-E011-VAL"][1] == "configs/eval/i4_v8_ranker_family_validation.yaml"
    val = _load("i4_v8_ranker_family_validation.yaml")
    assert val["partition"] == "validation"
    assert val["purpose"] == "selection"
    assert val["primary_metric"] == "info_per_kj"
    assert val["baseline"] == "A-B0_random"
    assert val["reference_arms"] == ["V4_protocol_only"]
    assert val["partition"] != "final_test"
    assert val["partition"] != "ood_test"
    assert list(val["arms"]) == ["A-B0_random", "V4_protocol_only", "V8_bayes_ratio_floor_001"]
    assert val["arms"]["V8_bayes_ratio_floor_001"]["v4"]["ranker"] == {
        "kind": "bayes_eig",
        "cost_mode": "ratio",
        "cost_weight": 1.0,
        "cost_floor": 0.01,
    }
