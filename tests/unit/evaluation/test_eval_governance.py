from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import yaml

from conrad.evaluation.acceptance import (
    AcceptanceRecord,
    AcceptanceStatus,
    GateKind,
    Measurements,
    evaluate,
    load_acceptance_record,
)
from conrad.evaluation.claims import (
    Claim,
    ClaimLevel,
    ClaimRefused,
    TransferArm,
    TwinUsefulnessStatus,
    check_claim,
    twin_usefulness,
)
from conrad.evaluation.maturity import (
    EvidenceLink,
    MaturityLevel,
    SubsystemMaturity,
    supported_level,
    validate_subsystem,
)
from conrad.evaluation.metrics import Interval

SPEC_EXAMPLE = """
acceptance:
  gate_id: I4
  primary_metric: hidden_state_error
  direction: lower
  baseline: fixed_views
  minimum_effect: OPEN
  uncertainty_method: OPEN
  evaluation_split_hash: OPEN
  seeds: OPEN
  budgets: {time_s: OPEN, energy_j: OPEN}
  guardrails: {unsupported_confidence_max: OPEN, safety_violations_max: OPEN}
  evidence_artifact: null
  status: NOT_EVALUABLE
"""


def filled() -> AcceptanceRecord:
    return AcceptanceRecord(
        gate_id="I4",
        primary_metric="hidden_state_error",
        baseline="fixed_views",
        minimum_effect=0.05,
        uncertainty_method="paired_bootstrap",
        evaluation_split_hash="h",
        seeds=(1, 2, 3),
        budgets={"time_s": 100.0},
        guardrails={"unsupported_confidence_max": 0.1},
    )


def measured(**kw) -> Measurements:
    base = {
        "split_hash": "h",
        "seeds": (3, 2, 1),
        "baseline": "fixed_views",
        "baseline_evidence": "runs/base",
        "effect": Interval(point=0.1, low=0.07, high=0.13, n=3, confidence=0.95, method="b"),
        "uncertainty_method": "paired_bootstrap",
        "budgets_used": {"time_s": 90.0},
        "guardrail_values": {"unsupported_confidence_max": 0.05},
        "evidence_artifact": "runs/x/reports/r.json",
    }
    base.update(kw)
    return Measurements(**base)


def test_spec_example_with_open_thresholds_is_not_evaluable(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_text(SPEC_EXAMPLE)
    record = load_acceptance_record(path)
    result = evaluate(record, measured())
    assert result.status is AcceptanceStatus.NOT_EVALUABLE
    assert any("minimum_effect" in r for r in result.reasons)
    path.write_text(SPEC_EXAMPLE.replace("status: NOT_EVALUABLE", "status: PASS"))
    with pytest.raises(ValueError):
        load_acceptance_record(path)


def test_filled_record_pass_fail_and_missing_measurements():
    assert evaluate(filled(), measured()).status is AcceptanceStatus.PASS
    small = Interval(point=0.05, low=0.01, high=0.09, n=3, confidence=0.95, method="b")
    assert evaluate(filled(), measured(effect=small)).status is AcceptanceStatus.FAIL
    assert evaluate(filled(), measured(budgets_used={"time_s": 150.0})).status is AcceptanceStatus.FAIL
    assert evaluate(filled(), measured(guardrail_values={})).status is AcceptanceStatus.NOT_EVALUABLE
    assert evaluate(filled(), measured(split_hash="other")).status is AcceptanceStatus.NOT_EVALUABLE
    assert evaluate(filled(), measured(evidence_artifact=None)).status is AcceptanceStatus.NOT_EVALUABLE


def test_deterministic_contract_gate_requires_zero_violations():
    gate = AcceptanceRecord(
        gate_id="GS-05",
        kind=GateKind.DETERMINISTIC_CONTRACT,
        primary_metric="violations",
        suite_id="golden",
        max_violations=0,
    )
    ok = Measurements(suite_id="golden", violations=0, evidence_artifact="r")
    assert evaluate(gate, ok).status is AcceptanceStatus.PASS
    assert evaluate(gate, ok.model_copy(update={"violations": 1})).status is AcceptanceStatus.FAIL
    open_gate = gate.model_copy(update={"max_violations": "OPEN"})
    assert evaluate(open_gate, ok).status is AcceptanceStatus.NOT_EVALUABLE


def test_claim_ladder_refuses_overclaim_and_wording():
    claim = Claim(
        claim_id="CLAIM-TCDP-01",
        statement="TCDP improved hidden-state error in our synthetic benchmark.",
        mechanism="TCDP",
        evidence_required="E-2T-01.*",
        experiments_supporting=("2T-TCDP-E001",),
        current_validation_level=ClaimLevel.VALIDATED_ON_PUBLIC_REAL_DATA,
    )
    with pytest.raises(ClaimRefused):
        check_claim(claim, ClaimLevel.VALIDATED_IN_SIMULATION)
    assert check_claim(claim, ClaimLevel.VALIDATED_ON_PUBLIC_REAL_DATA) is claim
    bad = claim.model_copy(update={"statement": "A revolutionary result."})
    with pytest.raises(ClaimRefused):
        check_claim(bad, ClaimLevel.VALIDATED_IN_REPRESENTATIVE_ENVIRONMENT)
    with pytest.raises(ValueError):
        Claim(
            claim_id="CLAIM-X",
            statement="s",
            mechanism="m",
            evidence_required="e",
            current_validation_level=ClaimLevel.VALIDATED_IN_SIMULATION,
        )


def arm(group: str, results: dict[int, float], **kw) -> TransferArm:
    base = {
        "group": group,
        "architecture": "2T-v1",
        "real_train_manifest": "r" * 64,
        "real_test_split_hash": "t",
        "adaptation_budget": {"optimizer_steps": 1000.0},
        "metric": "err",
        "direction": "lower",
        "seed_results": results,
    }
    base.update(kw)
    return TransferArm(**base)


def test_twin_usefulness():
    rng = np.random.default_rng(0)
    real = {1: 1.0, 2: 1.1, 3: 0.95}
    no_real = twin_usefulness(arm("G4", real, real_train_manifest=None), arm("G2", real), rng)
    assert no_real.status is TwinUsefulnessStatus.NOT_EVALUABLE
    unmatched = twin_usefulness(
        arm("G4", real), arm("G2", real, adaptation_budget={"optimizer_steps": 2000.0}), rng
    )
    assert unmatched.status is TwinUsefulnessStatus.NOT_EVALUABLE
    better = twin_usefulness(arm("G4", real), arm("G2", {k: v - 0.2 for k, v in real.items()}), rng)
    assert better.status is TwinUsefulnessStatus.POSITIVE and better.comparison is not None


def test_maturity_level_needs_evidence(tmp_path):
    (tmp_path / "spec.txt").write_text("spec")
    (tmp_path / "test_x.py").write_text("t")
    digest = hashlib.sha256(b"t").hexdigest()
    links = (
        EvidenceLink(level=MaturityLevel.D0, artifact="spec.txt", description="spec"),
        EvidenceLink(level=MaturityLevel.D1, artifact="test_x.py", sha256=digest, description="impl"),
    )
    entry = SubsystemMaturity(subsystem="BUO", claimed_level=MaturityLevel.D2, evidence=links)
    problems = validate_subsystem(entry, tmp_path)
    assert problems == ["BUO: claims D2 but has no valid D2 evidence"]
    assert supported_level(entry, tmp_path) is MaturityLevel.D1
    ok = entry.model_copy(update={"claimed_level": MaturityLevel.D1})
    assert validate_subsystem(ok, tmp_path) == []
    (tmp_path / "test_x.py").write_text("changed")
    assert validate_subsystem(ok, tmp_path)
    yaml.safe_dump(json.loads(ok.model_dump_json()))  # serializable tracking entry
