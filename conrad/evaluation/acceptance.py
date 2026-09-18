"""Acceptance records (ch28 Acceptance Records).

An unfilled required threshold yields NOT_EVALUABLE, never PASS. Deterministic contract gates
may require zero violations in their suite. Research gates need matched budgets, baseline
evidence, an effect size and an uncertainty interval.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from conrad.evaluation.metrics.bootstrap import Interval
from conrad.schemas.base import ConradModel

OPEN: Literal["OPEN"] = "OPEN"
OpenFloat = float | Literal["OPEN"]
OpenStr = str | Literal["OPEN"]


class GateKind(str, Enum):
    RESEARCH = "RESEARCH"
    DETERMINISTIC_CONTRACT = "DETERMINISTIC_CONTRACT"


class AcceptanceStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class AcceptanceRecord(ConradModel):
    gate_id: str = Field(min_length=1)
    kind: GateKind = GateKind.RESEARCH
    primary_metric: str = Field(min_length=1)
    direction: Literal["lower", "higher"] = "lower"
    baseline: str = OPEN
    minimum_effect: OpenFloat = OPEN
    uncertainty_method: OpenStr = OPEN
    evaluation_split_hash: OpenStr = OPEN
    seeds: tuple[int, ...] | Literal["OPEN"] = OPEN
    budgets: dict[str, OpenFloat] = Field(default_factory=dict)
    guardrails: dict[str, OpenFloat] = Field(default_factory=dict)
    suite_id: OpenStr = OPEN
    max_violations: int | Literal["OPEN"] = OPEN
    evidence_artifact: str | None = None
    status: AcceptanceStatus = AcceptanceStatus.NOT_EVALUABLE

    def open_fields(self) -> list[str]:
        if self.kind is GateKind.DETERMINISTIC_CONTRACT:
            names = {"suite_id": self.suite_id, "max_violations": self.max_violations}
            return [k for k, v in names.items() if v == OPEN]
        fields = {
            "baseline": self.baseline,
            "minimum_effect": self.minimum_effect,
            "uncertainty_method": self.uncertainty_method,
            "evaluation_split_hash": self.evaluation_split_hash,
            "seeds": self.seeds,
        }
        out = [k for k, v in fields.items() if v == OPEN]
        out += [f"budgets.{k}" for k, v in self.budgets.items() if v == OPEN]
        out += [f"guardrails.{k}" for k, v in self.guardrails.items() if v == OPEN]
        return out


class Measurements(ConradModel):
    """What a finished evaluation actually produced. Missing means not measured."""

    split_hash: str | None = None
    seeds: tuple[int, ...] = ()
    baseline: str | None = None
    baseline_evidence: str | None = None
    effect: Interval | None = Field(default=None, description="benefit interval, positive = better")
    uncertainty_method: str | None = None
    budgets_used: dict[str, float] = Field(default_factory=dict)
    guardrail_values: dict[str, float] = Field(default_factory=dict)
    suite_id: str | None = None
    violations: int | None = None
    evidence_artifact: str | None = None


class AcceptanceResult(ConradModel):
    gate_id: str
    status: AcceptanceStatus
    reasons: tuple[str, ...]


def _result(record: AcceptanceRecord, status: AcceptanceStatus, reasons: list[str]) -> AcceptanceResult:
    return AcceptanceResult(gate_id=record.gate_id, status=status, reasons=tuple(reasons))


def evaluate(record: AcceptanceRecord, measurements: Measurements) -> AcceptanceResult:
    opened = record.open_fields()
    if opened:
        return _result(record, AcceptanceStatus.NOT_EVALUABLE, [f"OPEN threshold: {f}" for f in opened])
    if not measurements.evidence_artifact:
        return _result(record, AcceptanceStatus.NOT_EVALUABLE, ["no evidence_artifact"])
    if record.kind is GateKind.DETERMINISTIC_CONTRACT:
        return _evaluate_contract(record, measurements)
    return _evaluate_research(record, measurements)


def _evaluate_contract(record: AcceptanceRecord, m: Measurements) -> AcceptanceResult:
    if m.suite_id != record.suite_id or m.violations is None:
        return _result(record, AcceptanceStatus.NOT_EVALUABLE, [f"suite {record.suite_id!r} was not run"])
    assert isinstance(record.max_violations, int)
    if m.violations > record.max_violations:
        return _result(
            record, AcceptanceStatus.FAIL, [f"{m.violations} violations > {record.max_violations}"]
        )
    return _result(record, AcceptanceStatus.PASS, [f"{m.violations} violations"])


def _evaluate_research(record: AcceptanceRecord, m: Measurements) -> AcceptanceResult:
    missing: list[str] = []
    if m.split_hash != record.evaluation_split_hash:
        missing.append(f"evaluated split {m.split_hash!r} != {record.evaluation_split_hash!r}")
    if tuple(sorted(m.seeds)) != tuple(sorted(record.seeds)):
        missing.append("seed set differs from the declared seeds")
    if m.baseline != record.baseline or not m.baseline_evidence:
        missing.append(f"no baseline implementation evidence for {record.baseline!r}")
    if m.effect is None or m.effect.low is None or m.effect.high is None:
        missing.append("no effect size with an uncertainty interval")
    if m.uncertainty_method != record.uncertainty_method:
        missing.append(f"uncertainty method {m.uncertainty_method!r} != {record.uncertainty_method!r}")
    missing += [f"budget {k} not measured" for k in record.budgets if k not in m.budgets_used]
    missing += [f"guardrail {k} not measured" for k in record.guardrails if k not in m.guardrail_values]
    if missing:
        return _result(record, AcceptanceStatus.NOT_EVALUABLE, missing)
    failures: list[str] = []
    for key, limit in record.budgets.items():
        assert isinstance(limit, float | int)
        if m.budgets_used[key] > limit:
            failures.append(f"budget {key}: {m.budgets_used[key]} > {limit}")
    for key, limit in record.guardrails.items():
        assert isinstance(limit, float | int)
        if m.guardrail_values[key] > limit:
            failures.append(f"guardrail {key}: {m.guardrail_values[key]} > {limit}")
    assert (
        m.effect is not None and m.effect.low is not None and isinstance(record.minimum_effect, float | int)
    )
    if m.effect.low < record.minimum_effect:
        failures.append(f"effect lower bound {m.effect.low:.4g} < minimum_effect {record.minimum_effect}")
    if failures:
        return _result(record, AcceptanceStatus.FAIL, failures)
    return _result(record, AcceptanceStatus.PASS, [f"effect lower bound {m.effect.low:.4g}"])


def load_acceptance_record(path: str | Path) -> AcceptanceRecord:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    section = data.get("acceptance", data) if isinstance(data, dict) else None
    if not isinstance(section, dict):
        raise ValueError(f"{path} has no acceptance mapping")
    record = AcceptanceRecord.model_validate(section)
    if record.status is AcceptanceStatus.PASS:
        raise ValueError(
            f"{record.gate_id}: a stored record may not assert PASS; status comes from evaluate()"
        )
    return record
