"""Run and record the separately versioned Spatial V1.1 FORMAL Unity I5 cycle."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from record_unity_gate_evidence import read_outcomes

from conrad.evaluation.gates import CriterionResult, CriterionStatus, EvidenceClass, GateEvidence
from conrad.sim.mission.unity_run import git_commit, player_identity

ROOT = Path(__file__).resolve().parent.parent
MODULE = "tests/unity_live/test_i5_spatial_v1_1_unity.py"
ARTIFACT_DIR = ROOT / "artifacts/gates/I5_SPATIAL_V1_1"
XML = ARTIFACT_DIR / "unity_pytest.xml"
LOG = ARTIFACT_DIR / "unity_pytest.txt"
MEASURED = ARTIFACT_DIR / "unity_measured.json"
RESULTS = ARTIFACT_DIR / "unity_i5_spatial_v1_1_results.json"
EVIDENCE = ROOT / "artifacts/gates/I5/evidence_formal_spatial_v1_1.json"
MATRIX = ROOT / "artifacts/experiments/M1-ACTION-E001/m1_action_e001.json"
PREFIX = "test_i5_spatial_v1_1_unity::"
MATRIX_CLASSES = ("continue", "request evidence", "replan", "change sensing", "return", "escalate")
INTEGRATED = (
    (
        "actions exercised correctly inside integrated missions",
        "test_actions_exercised_correctly_inside_integrated_missions",
    ),
    ("traceable decisions with low measured UIR", "test_traceable_decisions_with_low_measured_uir"),
    (
        "competitive mission outcomes vs decision baselines",
        "test_competitive_mission_outcomes_vs_decision_baselines",
    ),
)
SUPPORT = (
    "test_every_scenario_and_arm_flew_on_every_world",
    "test_no_twin_truth_leakage_on_runtime_side",
    "test_bundle_replays_deterministically",
    "test_results_file_is_versioned_and_written",
)


def _status(outcomes: dict[str, str], nodes: list[str]) -> CriterionStatus:
    values = [outcomes.get(PREFIX + node, "NOT_RUN") for node in nodes]
    if "FAIL" in values:
        return CriterionStatus.FAIL
    if "SKIP" in values or "NOT_RUN" in values:
        return CriterionStatus.NOT_RUN
    return CriterionStatus.PASS


def _run() -> dict[str, str]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-v",
                "-rA",
                "-p",
                "no:cacheprovider",
                "--run-unity-live",
                "--run-formal-unity-gates",
                f"--junitxml={XML}",
                MODULE,
            ],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return read_outcomes(XML)


def record(*, run: bool = True) -> GateEvidence:
    if EVIDENCE.exists():
        raise SystemExit(f"refusing to overwrite immutable evidence {EVIDENCE}")
    outcomes = _run() if run else read_outcomes(XML)
    measured = (
        json.loads(MEASURED.read_text(encoding="utf-8"))
        if MEASURED.exists()
        else {"meta": {}, "criteria": {}}
    )
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    criteria: list[CriterionResult] = []
    for name in MATRIX_CLASSES:
        node = f"test_action_matrix_class[{name}]"
        row = matrix["summary"]["egdc_structured"]["per_class"][name]
        values = {
            "recall": row["recall"],
            "n_canonical": row["n_canonical"],
            "floor": matrix["acceptance"]["canonical_recall_floor"],
            "chosen_as_itself": matrix["summary"]["egdc_structured"]["confusion_matrix"][name][name],
        }
        criteria.append(
            CriterionResult(
                criterion=name,
                status=_status(outcomes, [node]),
                measured=json.dumps(values, sort_keys=True),
            )
        )
    hard = matrix["summary"]["egdc_structured"]
    criteria.append(
        CriterionResult(
            criterion="hard constraints inviolable",
            status=_status(outcomes, ["test_matrix_hard_constraints_inviolable"]),
            measured=json.dumps(
                {"hard_constraint_violations": hard["hard_constraint_violations"], "uir": hard["uir"]},
                sort_keys=True,
            ),
        )
    )
    for criterion, node in INTEGRATED:
        nodes = [node, *SUPPORT]
        values = {
            criterion: measured["criteria"].get(criterion),
            "grid": measured["criteria"].get("coverage of the declared grid"),
            "leakage": measured["criteria"].get("leakage"),
            "replay": measured["criteria"].get("replay"),
        }
        criteria.append(
            CriterionResult(
                criterion=criterion,
                status=_status(outcomes, nodes),
                measured=json.dumps(values, sort_keys=True),
            )
        )
    evidence = GateEvidence(
        gate_id="I5",
        evidence_class=EvidenceClass.FORMAL,
        execution_path="Model1 action matrix plus frozen Spatial V1.1 through built Unity V2 player",
        git_commit=git_commit(),
        criteria=tuple(criteria),
        artifacts=(
            MODULE,
            str(XML.relative_to(ROOT)),
            str(LOG.relative_to(ROOT)),
            str(MEASURED.relative_to(ROOT)),
            str(RESULTS.relative_to(ROOT)),
            "configs/eval/i5_spatial_v1_1_unity_formal.yaml",
            "configs/eval/partitions_i5_unity_v3.yaml",
            "artifacts/experiments/M1-ACTION-SPATIAL-V1-1-FINAL/final_verification.json",
        ),
        notes=(
            f"meta={json.dumps(measured.get('meta', {}), sort_keys=True)}; "
            f"player={json.dumps(player_identity(), sort_keys=True)}; "
            "fresh worlds 8701000-8701001; unused suffix 8701002-8701009 not opened; "
            "all 42 Unity flights sequential; SYNTHETIC_ONLY; L1_APPROXIMATE_PHYSICS; "
            "historical I5 FORMAL 9/10 FAIL remains preserved in evidence_formal.json"
        ),
    )
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    evidence = record(run="--no-run" not in sys.argv)
    statuses = [f"{item.criterion}:{item.status.value}" for item in evidence.criteria]
    print("I5 Spatial V1.1", statuses)
    if any(item.status is not CriterionStatus.PASS for item in evidence.criteria):
        raise SystemExit(1)
