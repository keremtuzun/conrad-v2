"""The v10 validation declaration is prospective and contract-preserving."""

from pathlib import Path

import yaml

from conrad.evaluation import partitions as P
from conrad.evaluation.dispatch import EXPERIMENTS

ROOT = Path(__file__).resolve().parents[3]
DEVELOPMENT = ROOT / "configs/eval/m1_action_spatial_v1_1_development_r6.yaml"
VALIDATION = ROOT / "configs/eval/m1_action_spatial_v1_1_validation.yaml"


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_validation_uses_all_and_only_v10_validation_seeds() -> None:
    cfg = _config(VALIDATION)
    validation = P.split(P.I5_V10_DOMAIN, P.Partition.VALIDATION, P.Purpose.SELECTION).world_seeds
    final = P.split(P.I5_V10_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION).world_seeds
    assert cfg["partition"] == "validation"
    assert cfg["partition_domain"] == P.I5_V10_DOMAIN
    assert cfg["seeds"] == list(validation)
    assert not set(cfg["seeds"]) & set(final)


def test_validation_preserves_the_r6_mission_contract() -> None:
    development, validation = _config(DEVELOPMENT), _config(VALIDATION)
    for cfg in (development, validation):
        cfg.pop("experiment_id")
        cfg.pop("hypothesis")
        cfg.pop("partition")
        cfg.pop("seeds")
        cfg.pop("limitations")
    development.pop("advancement_rule")
    validation.pop("selection_rule")
    assert validation == development


def test_validation_rule_is_fail_closed_and_registered() -> None:
    cfg = _config(VALIDATION)
    rule = cfg["selection_rule"]
    assert rule["primary_nominal_warrants_min"] == 6
    assert rule["non_nominal_warrants_per_scenario_min"] == 1
    assert rule["non_nominal_correct_given_warrant_rate_min"] == 0.9
    assert rule["false_intact_on_covered_resolvable_defect_max"] == 0
    assert rule["decision_violations_max"] == 0
    assert rule["nominal_over_escalations_max"] == 0
    assert rule["traceability_min"] == 1.0
    assert rule["uir_max"] == 0.0
    assert EXPERIMENTS[cfg["experiment_id"]][1] == str(VALIDATION.relative_to(ROOT))
