"""The v10 final declaration is fixed, powered, and contract-preserving."""

from pathlib import Path

import yaml

from conrad.evaluation import partitions as P
from conrad.evaluation.dispatch import EXPERIMENTS

ROOT = Path(__file__).resolve().parents[3]
VALIDATION = ROOT / "configs/eval/m1_action_spatial_v1_1_validation.yaml"
FINAL = ROOT / "configs/eval/m1_action_spatial_v1_1_final.yaml"


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_final_uses_only_the_fixed_28_world_prefix() -> None:
    cfg = _config(FINAL)
    final_pool = P.split(P.I5_V10_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION).world_seeds
    assert cfg["partition"] == "final_test"
    assert cfg["seeds"] == list(final_pool[:28])
    assert not set(cfg["seeds"]) & set(final_pool[28:])
    assert cfg["final_rule"]["expected_rows"] == 28 * 7 * 3


def test_final_preserves_the_validation_mission_contract() -> None:
    validation, final = _config(VALIDATION), _config(FINAL)
    for cfg in (validation, final):
        cfg.pop("experiment_id")
        cfg.pop("hypothesis")
        cfg.pop("partition")
        cfg.pop("seeds")
        cfg.pop("limitations")
    validation.pop("selection_rule")
    final.pop("final_rule")
    assert final == validation


def test_final_rule_matches_the_exact_power_artifact_and_is_registered() -> None:
    cfg = _config(FINAL)
    rule = cfg["final_rule"]
    power = rule["power_design"]
    assert rule["primary_nominal_warrants_min"] == 19
    assert rule["each_scenario_correct_given_warrant_rate_min"] == 0.9
    assert rule["false_intact_on_covered_resolvable_defect_max"] == 0
    assert rule["decision_violations_max"] == 0
    assert rule["nominal_over_escalations_max"] == 0
    assert rule["traceability_min"] == 1.0
    assert rule["uir_max"] == 0.0
    assert power["artifact_sha256"] == "cecf8a6593194654100089410f51816cf1cf50c0f7c7abc86292fa474ef2195c"
    assert power["false_positive_probability_at_null"] <= power["alpha"]
    assert power["power_at_validation_lower_bound"] >= power["target_power"]
    assert EXPERIMENTS[cfg["experiment_id"]][1] == str(FINAL.relative_to(ROOT))
