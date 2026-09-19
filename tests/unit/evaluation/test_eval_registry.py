from __future__ import annotations

import json

import numpy as np
import pytest

from conrad.evaluation.claims import ClaimLevel
from conrad.evaluation.metrics import paired_seed_comparison
from conrad.evaluation.registry import (
    SEED_HYPOTHESES,
    Answer,
    CostReport,
    EntryResult,
    ExperimentRecord,
    ExperimentRegistry,
    HypothesisRegistry,
    HypothesisStatus,
    MechanismComparison,
    NotebookEntry,
    Outcome,
    RegistryIntegrityError,
    ResearchNotebook,
    Tier,
    Verdict,
    assess,
    retain_with_adr,
)

BASE = {1: 1.0, 2: 1.2, 3: 0.9, 4: 1.1}


def cmp(delta: float, direction: str = "lower"):
    cand = {k: v + delta for k, v in BASE.items()}
    return paired_seed_comparison(BASE, cand, np.random.default_rng(0), metric="m", direction=direction)


def record(**kw) -> ExperimentRecord:
    base = {
        "experiment_id": "CORE-BUO-E001",
        "hypothesis_id": "H-CORE-02",
        "mechanism": "BUO",
        "hypothesis": "h",
        "tier": Tier.T1_DEVELOPMENT,
        "baselines": ("weighted averaging",),
        "primary_metric": "state_error",
        "direction": "lower",
        "seeds": (1, 2, 3, 4),
        "split_hash": "s",
        "config_digest": "c",
        "code_commit": "a" * 40,
        "git_dirty": False,
        "outcome": Outcome.REFUTES,
        "recorded_time_ns": 1,
    }
    base.update(kw)
    return ExperimentRecord(**base)


def test_hypotheses_seeded_untested():
    reg = HypothesisRegistry()
    assert len(reg.all()) == 18 and all(h.status is HypothesisStatus.UNTESTED for h in SEED_HYPOTHESES)
    assert reg.get("H-CORE-02").kill_rule and reg.by_tier("A")


def test_experiment_discipline():
    record(outcome=Outcome.FAILED_RUN, code_commit=None)  # failed runs are still recorded
    with pytest.raises(ValueError, match="lacks"):
        record(code_commit=None)
    with pytest.raises(ValueError, match="paired effect"):
        record(outcome=Outcome.SUPPORTS)
    with pytest.raises(ValueError, match="clean"):
        record(outcome=Outcome.SUPPORTS, effect=cmp(-0.3), git_dirty=True)
    with pytest.raises(ValueError):
        record(evidence_level=ClaimLevel.VALIDATED_IN_SIMULATION)
    with pytest.raises(ValueError):
        record(experiment_id="random-name")
    ok = record(outcome=Outcome.SUPPORTS, effect=cmp(-0.3), evidence_level=ClaimLevel.VALIDATED_IN_SIMULATION)
    assert ok.effect is not None and ok.effect.repeatable_benefit


def test_registry_append_only_and_tamper_detection(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.jsonl")
    reg.append(record(outcome=Outcome.FAILED_RUN))
    reg.append(record())
    reg.append(
        record(
            experiment_id="2S-UAHSM-E001", hypothesis_id="H-2S-01", mechanism="UAHSM", outcome=Outcome.PLANNED
        )
    )
    assert reg.verify() == []
    latest = reg.latest("CORE-BUO-E001")
    assert latest is not None
    assert len(reg.history("CORE-BUO-E001")) == 2 and latest.outcome is Outcome.REFUTES
    assert [r.experiment_id for r in reg.query(prefix="2S-")] == ["2S-UAHSM-E001"]
    assert len(reg.query(hypothesis_id="H-CORE-02", outcome=Outcome.FAILED_RUN)) == 1
    lines = (tmp_path / "exp.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    row["record"]["outcome"] = "SUPPORTS"
    lines[0] = json.dumps(row, sort_keys=True)
    (tmp_path / "exp.jsonl").write_text("\n".join(lines) + "\n")
    assert reg.verify()
    with pytest.raises(RegistryIntegrityError):
        reg.append(record())


def matched(**kw) -> MechanismComparison:
    base = {
        "mechanism": "BUO",
        "baseline": "GRU update",
        "matched": True,
        "task_error": cmp(-0.3),
        "calibration": cmp(-0.2),
        "ood": cmp(-0.1),
        "cost": CostReport(latency_ms=5, memory_mb=10, budget_latency_ms=20, budget_memory_mb=50),
    }
    base.update(kw)
    return MechanismComparison(**base)


def test_kill_protocol_four_questions():
    good = assess(matched())
    assert good.verdict is Verdict.RETAIN_CANDIDATE
    assert (good.task_error, good.calibration, good.ood, good.cost_justified) == (Answer.YES,) * 4
    assert assess(matched(task_error=cmp(0.0))).verdict is Verdict.KILL_CANDIDATE
    assert assess(matched(calibration=cmp(0.3))).verdict is Verdict.KILL_CANDIDATE
    assert assess(matched(leakage_detected=True)).verdict is Verdict.KILL_CANDIDATE
    over = CostReport(latency_ms=50, memory_mb=10, budget_latency_ms=20, budget_memory_mb=50)
    assert assess(matched(cost=over)).verdict is Verdict.KILL_CANDIDATE
    assert (
        assess(matched(cost=over.model_copy(update={"documented_mission_benefit": "ADR-9"}))).verdict
        is Verdict.RETAIN_CANDIDATE
    )
    assert assess(matched(ood=None)).verdict is Verdict.NOT_EVALUABLE
    assert assess(matched(cost=CostReport())).verdict is Verdict.NOT_EVALUABLE
    assert assess(matched(matched=False)).verdict is Verdict.NOT_EVALUABLE


def test_kill_candidate_retention_requires_adr():
    kill = assess(matched(safety_regression=True))
    with pytest.raises(ValueError):
        retain_with_adr(kill, adr_ref="ADR-7", mission_condition="", compensating_control="x")
    kept = retain_with_adr(
        kill, adr_ref="ADR-7", mission_condition="turbid harbour", compensating_control="operator gate"
    )
    assert kept.verdict is Verdict.RETAINED_BY_ADR
    with pytest.raises(ValueError):
        retain_with_adr(assess(matched()), adr_ref="a", mission_condition="b", compensating_control="c")


def entry(t: int, result=EntryResult.NEGATIVE) -> NotebookEntry:
    return NotebookEntry(
        time_ns=t,
        contributor="kerem",
        method="ablation",
        result=result,
        result_summary="no gain",
        limitation="tiny data",
        decision="simplify",
        next_action="rerun at T2",
    )


def test_notebook_append_only(tmp_path):
    nb = ResearchNotebook(tmp_path / "entries.jsonl")
    nb.append(entry(1))
    nb.append(entry(2, EntryResult.INCONCLUSIVE))
    assert [e.result for e in nb.entries()] == [EntryResult.NEGATIVE, EntryResult.INCONCLUSIVE]
    with pytest.raises(ValueError, match="chronological"):
        nb.append(entry(0))
    text = (tmp_path / "entries.jsonl").read_text().replace("no gain", "big gain")
    (tmp_path / "entries.jsonl").write_text(text)
    with pytest.raises(ValueError, match="edited"):
        nb.entries()
