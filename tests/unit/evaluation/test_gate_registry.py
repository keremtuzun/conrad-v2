from __future__ import annotations

from pathlib import Path

from conrad.evaluation.gates import (
    GATE_BY_ID,
    CriterionResult,
    CriterionStatus,
    EvidenceClass,
    GateEvidence,
    GateStatus,
    evaluate_gates,
    write_evidence,
)


def _ev(gate: str, cls: EvidenceClass, status: CriterionStatus = CriterionStatus.PASS) -> GateEvidence:
    crit = tuple(CriterionResult(criterion=c, status=status) for c in GATE_BY_ID[gate].criteria)
    return GateEvidence(gate_id=gate, evidence_class=cls, execution_path="t", git_commit="x", criteria=crit)


def test_surrogate_evidence_never_promotes(tmp_path: Path) -> None:
    for g in ("P0", "I0", "C1", "2S-FIRST"):
        write_evidence(_ev(g, EvidenceClass.FORMAL), tmp_path)
    write_evidence(_ev("I1", EvidenceClass.SURROGATE), tmp_path)
    r = evaluate_gates(tmp_path)
    assert r["C1"].official_status is GateStatus.PASS
    assert r["I1"].surrogate_status is GateStatus.PASS
    assert r["I1"].official_status is not GateStatus.PASS


def test_downstream_cannot_pass_without_upstream(tmp_path: Path) -> None:
    for g in ("P0", "I0", "C1", "2S-FIRST", "I1"):
        write_evidence(_ev(g, EvidenceClass.FORMAL), tmp_path)
    r = evaluate_gates(tmp_path)
    assert r["U0"].official_status is GateStatus.NOT_RUN
    assert r["I1"].official_status is GateStatus.BLOCKED_UPSTREAM and "U0" in r["I1"].blocking_upstream


def test_failed_criterion_fails_gate_and_open_threshold_is_not_evaluable(tmp_path: Path) -> None:
    write_evidence(_ev("P0", EvidenceClass.FORMAL), tmp_path)
    write_evidence(_ev("I0", EvidenceClass.FORMAL, CriterionStatus.NOT_EVALUABLE), tmp_path)
    assert evaluate_gates(tmp_path)["I0"].official_status is GateStatus.NOT_EVALUABLE
    write_evidence(_ev("I0", EvidenceClass.FORMAL, CriterionStatus.FAIL), tmp_path)
    assert evaluate_gates(tmp_path)["I0"].official_status is GateStatus.FAIL


def test_hardware_gates_are_external_without_formal_evidence(tmp_path: Path) -> None:
    r = evaluate_gates(tmp_path)
    assert r["I8"].official_status is GateStatus.BLOCKED_EXTERNAL
    assert r["I9"].official_status is GateStatus.BLOCKED_EXTERNAL


def test_recorder_labels_surrogate_runs_as_surrogate() -> None:
    import importlib.util

    root = Path(__file__).resolve().parent.parent.parent.parent
    spec = importlib.util.spec_from_file_location("rec", root / "scripts" / "record_gate_evidence.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = (root / "scripts" / "record_gate_evidence.py").read_text(encoding="utf-8")
    assert "EvidenceClass.SURROGATE if surrogate else EvidenceClass.FORMAL" in src
    assert set(mod.SURROGATE_PLAN).isdisjoint(set(mod.PLAN)), "a gate must not have both a formal and surrogate plan here"
