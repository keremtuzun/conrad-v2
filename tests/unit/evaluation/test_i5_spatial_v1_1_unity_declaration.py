"""The fresh Spatial V1.1 Unity formal cycle is sealed and contract-preserving."""

from pathlib import Path

import yaml

from conrad.evaluation import partitions as P

ROOT = Path(__file__).resolve().parents[3]
FORMAL = ROOT / "configs/eval/i5_spatial_v1_1_unity_formal.yaml"
SURROGATE = ROOT / "configs/eval/m1_action_spatial_v1_1_final.yaml"


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_formal_uses_only_the_declared_fresh_two_world_prefix() -> None:
    cfg = _config(FORMAL)
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        pool = P.split(P.I5_UNITY_V3_DOMAIN, P.Partition.FINAL_TEST, P.Purpose.FINAL_EVALUATION).world_seeds
    assert cfg["final_worlds"] == list(pool[:2])
    assert cfg["unused_final_worlds"] == list(pool[2:])
    assert cfg["formal_rule"]["expected_rows"] == 2 * 7 * 3


def test_formal_preserves_the_frozen_spatial_mission_contract() -> None:
    formal, surrogate = _config(FORMAL), _config(SURROGATE)
    assert formal["spatial_v1"] == surrogate["spatial_v1"]
    assert formal["scenarios"] == surrogate["scenarios"]
    assert formal["arms"] == surrogate["arms"]
    assert formal["latency_budget_s"] == surrogate["latency_budget_s"]
    assert formal["success_floor"] == surrogate["success_floor"]
    assert formal["uir_max"] == surrogate["uir_max"]
    assert formal["contact_clearance_m"] == surrogate["contact_clearance_m"]


def test_formal_is_pinned_to_the_passed_surrogate_and_native_player() -> None:
    cfg = _config(FORMAL)
    assert cfg["surrogate_final"]["status"] == "PASS"
    assert cfg["surrogate_final"]["result_sha256"] == (
        "0dbc96d699a698ed95011330692533312094d941b5e3fb9c9bf25a965e10e3d1"
    )
    assert cfg["player"]["executable_sha256"] == (
        "7e42f64199b043204d62737d06aabc6b6ceb1f3c1c699f158a612b938ea87ac6"
    )
    assert cfg["formal_rule"]["all_ten_existing_criteria_must_pass"] is True
    assert cfg["formal_rule"]["every_scenario_warrants_min"] == 2
