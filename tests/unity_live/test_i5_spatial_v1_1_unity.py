"""Fresh FORMAL Unity I5 cycle for frozen Spatial V1.1."""

from __future__ import annotations

import faulthandler
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from unity_gate_support import GATE_RUNS, GATES, MISSION_CONFIG, leakage_scan, measured, reset_measured

from conrad.evaluation import partitions as P
from conrad.evaluation.decision_experiments.m1_action_integrated import (
    BASELINES,
    PRIMARY,
    SPECS,
    _merge,
    _spatial_diagnostics,
    _spatial_profile,
    drive_and_score,
    summarize,
    verdicts,
)
from conrad.orchestration.mission_config import runtime_config
from conrad.settings import REPO_ROOT, load_settings
from conrad.sim.mission.options import world_options
from conrad.sim.mission.unity_run import player_identity, prepare_unity, replay_unity_run, resolve_unity

GATE = "I5_SPATIAL_V1_1"
CONFIG = REPO_ROOT / "configs/eval/i5_spatial_v1_1_unity_formal.yaml"
RESULTS = GATES / GATE / "unity_i5_spatial_v1_1_results.json"
MATRIX_ARM = "egdc_structured"
MATRIX_CLASSES = ("continue", "request evidence", "replan", "change sensing", "return", "escalate")
C_ACTIONS = "actions exercised correctly inside integrated missions"
C_TRACE = "traceable decisions with low measured UIR"
C_COMPETITIVE = "competitive mission outcomes vs decision baselines"
MIN_PER_CONDITION = 50
HARD_TIMEOUT_S = 8 * 3600.0


@pytest.fixture(autouse=True)
def _hard_timeout() -> Iterator[None]:
    faulthandler.dump_traceback_later(HARD_TIMEOUT_S, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _config() -> dict[str, Any]:
    return dict(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def matrix() -> dict[str, Any]:
    path = REPO_ROOT / _config()["matrix_artifact"]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["partition"] == "final_test" and data["purpose"] == "final_evaluation"
    return data


def _recall(matrix: dict[str, Any], cls: str) -> dict[str, Any]:
    summary = matrix["summary"][MATRIX_ARM]["per_class"][cls]
    floor = float(matrix["acceptance"]["canonical_recall_floor"])
    return {
        "recall": summary["recall"],
        "n_canonical": summary["n_canonical"],
        "floor": floor,
        "chosen_as_itself": matrix["summary"][MATRIX_ARM]["confusion_matrix"][cls][cls],
    }


@pytest.mark.no_unity_player
@pytest.mark.parametrize("cls", MATRIX_CLASSES)
def test_action_matrix_class(matrix, cls):
    measurement = _recall(matrix, cls)
    measured(GATE, cls, **measurement)
    assert measurement["n_canonical"] >= MIN_PER_CONDITION
    assert measurement["recall"] is not None and measurement["recall"] >= measurement["floor"]
    assert measurement["chosen_as_itself"] >= measurement["floor"] * measurement["n_canonical"]


@pytest.mark.no_unity_player
def test_matrix_hard_constraints_inviolable(matrix):
    summary = matrix["summary"][MATRIX_ARM]
    hard = summary["hard_constraint_violations"]
    measured(GATE, "hard constraints inviolable", hard_constraints=hard, uir=summary["uir"])
    assert hard["chosen_action_audit"] == 0
    assert hard["injected_invalid_accepted"] == 0
    assert hard["total"] == 0
    assert summary["uir"]["unsupported_inference_rate"] == 0.0
    assert summary["uir"]["relied_world_claims"] > 0


def _worlds(cfg: dict[str, Any]) -> list[int]:
    seeds = [int(seed) for seed in cfg["final_worlds"]]
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        allowed = set(P.split(P.I5_UNITY_V3_DOMAIN, "final_test", "final_evaluation").world_seeds)
    assert set(seeds) <= allowed
    assert not set(seeds) & set(cfg["unused_final_worlds"])
    return seeds


def _resolved_spatial_options(cfg: dict[str, Any], scenario: str):
    settings = load_settings(MISSION_CONFIG)
    mission = _merge(dict(settings.sim.get("mission", {})), _spatial_profile(cfg, scenario))
    world_raw, runtime_raw = resolve_unity(scenario, mission)
    per_scenario = dict(cfg["spatial_v1"].get("runtime_by_scenario", {})).get(scenario, {})
    return world_options(world_raw), runtime_config(_merge(runtime_raw, dict(per_scenario)))


def _fly(seed: int, scenario: str, arm: str, cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    run_id = f"I5-SPATIAL-V1-1-UNITY-{scenario}-s{seed}-{arm}"
    run_dir = root / run_id
    scored = run_dir / "reports/i5_spatial_v1_1_score.json"
    if scored.exists() and (run_dir / "bundle_manifest.json").exists():
        return json.loads(scored.read_text(encoding="utf-8"))
    identity = player_identity()
    assert identity["player_exe_sha256"] == cfg["player"]["executable_sha256"]
    stored_world, stored_runtime = _resolved_spatial_options(cfg, scenario)
    session = prepare_unity(
        scenario,
        MISSION_CONFIG,
        run_id=run_id,
        runs_root=root,
        seed=seed,
        stored_world=stored_world,
        stored_runtime=stored_runtime,
    )
    try:
        row = drive_and_score(session, seed, scenario, arm, cfg)
        row["spatial_v1"] = _spatial_diagnostics(session, row["decision_states"])
        output = session.finish()
    except BaseException:
        session.abort()
        raise
    row["run_dir"] = str(Path(output["run_dir"]).relative_to(REPO_ROOT))
    scored.parent.mkdir(parents=True, exist_ok=True)
    scored.write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
    return row


@pytest.fixture(scope="module")
def flights() -> dict[str, Any]:
    cfg = _config()
    seeds = _worlds(cfg)
    scenarios = [str(value) for value in cfg["scenarios"]]
    arms = [str(value) for value in cfg["arms"]]
    assert set(scenarios) == set(SPECS) - {"I5-NOMINAL-READABLE"}
    assert arms == [PRIMARY, *BASELINES]
    rows = [
        _fly(seed, scenario, arm, cfg, GATE_RUNS / GATE)
        for seed in seeds
        for scenario in scenarios
        for arm in arms
    ]
    summary = summarize(rows, scenarios)
    gate_verdicts = verdicts(summary, cfg, scenarios, rows)
    identity = player_identity()
    result = {
        "experiment_id": cfg["experiment_id"],
        "evidence_kind": "FORMAL",
        "evidence_class": "FORMAL (built Unity V2 player, lock-step TCP, Spatial V1.1)",
        "data_status": "SYNTHETIC_ONLY",
        "partition": "final_test",
        "partition_file": cfg["partition_file"],
        "partition_digest": P.load_i5_unity_v3()["digest"],
        "config": str(CONFIG.relative_to(REPO_ROOT)),
        "config_sha256": _sha256(CONFIG),
        "worlds": seeds,
        "scenarios": scenarios,
        "arms": arms,
        "player": identity,
        **summary,
        "verdicts": gate_verdicts,
        "per_run": rows,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    reset_measured(
        GATE,
        {
            "experiment_id": cfg["experiment_id"],
            "worlds": seeds,
            "scenarios": scenarios,
            "arms": arms,
            "partition": cfg["partition_file"],
            "config_sha256": _sha256(CONFIG),
            "player": identity,
            "results": str(RESULTS.relative_to(REPO_ROOT)),
        },
    )
    return {"rows": rows, "summary": summary, "verdicts": gate_verdicts, "config": cfg}


def test_actions_exercised_correctly_inside_integrated_missions(flights):
    verdict = flights["verdicts"]
    rule = flights["config"]["formal_rule"]
    measured(
        GATE,
        C_ACTIONS,
        per_scenario_warrants=verdict["per_scenario_warrant_reached"],
        per_scenario_correct_given_warrant=verdict["per_scenario_correct_given_warrant_rate"],
        false_intact=verdict["spatial_false_intact_zero"],
        violations_total=verdict["violations_total"],
        nominal_over_escalations=verdict["nominal_over_escalations"],
        spatial_v1_runtime_selected=verdict["spatial_v1_runtime_selected"],
    )
    assert verdict["actions_exercised_correctly"]
    assert verdict["spatial_v1_runtime_selected"]
    assert verdict["spatial_false_intact_zero"]
    assert all(
        count >= rule["every_scenario_warrants_min"]
        for count in verdict["per_scenario_warrant_reached"].values()
    )


def test_traceable_decisions_with_low_measured_uir(flights):
    verdict = flights["verdicts"]
    measured(GATE, C_TRACE, traceable_fraction=verdict["traceable_fraction"], uir=verdict["uir"])
    assert verdict["traceable_low_uir"]


def test_competitive_mission_outcomes_vs_decision_baselines(flights):
    verdict = flights["verdicts"]
    measured(GATE, C_COMPETITIVE, competitive=verdict["competitive"])
    assert verdict["competitive_outcomes"]


def test_every_scenario_and_arm_flew_on_every_world(flights):
    cfg = flights["config"]
    observed = {(row["seed"], row["scenario"], row["arm"]) for row in flights["rows"]}
    expected = {
        (seed, scenario, arm)
        for seed in cfg["final_worlds"]
        for scenario in cfg["scenarios"]
        for arm in cfg["arms"]
    }
    measured(GATE, "coverage of the declared grid", flights=len(observed), declared=len(expected))
    assert len(flights["rows"]) == cfg["formal_rule"]["expected_rows"]
    assert observed == expected


def test_no_twin_truth_leakage_on_runtime_side(flights):
    scanned: dict[str, int] = {}
    violations: list[str] = []
    for row in flights["rows"]:
        run_dir = REPO_ROOT / row["run_dir"]
        meta = json.loads((run_dir / "truth/truth_record.json").read_text(encoding="utf-8"))["meta"]
        count, bad = leakage_scan(run_dir, set(meta["world_entity_ids"]), set())
        # Score rows intentionally contain only evaluation fields.  Use the
        # immutable bundle path as the audit label instead of assuming a
        # separate run_id field was copied into the score payload.
        scanned[row["run_dir"]] = count
        violations.extend(bad)
        assert count > 100
    measured(GATE, "leakage", runs=scanned, violations=violations[:20])
    assert not violations


def test_bundle_replays_deterministically(flights):
    row = next(item for item in flights["rows"] if item["arm"] == PRIMARY)
    replay = replay_unity_run(REPO_ROOT / row["run_dir"], GATE_RUNS / "replay" / GATE)
    measured(GATE, "replay", run=row["run_dir"], **replay)
    assert replay["equal"] and replay["same_player_binary"]


def test_results_file_is_versioned_and_written(flights):
    assert RESULTS.exists()
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert result["experiment_id"] == "I5-SPATIAL-V1-1-UNITY-FORMAL"
    assert result["player"]["player_exe_sha256"] == flights["config"]["player"]["executable_sha256"]
