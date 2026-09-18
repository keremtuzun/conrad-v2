"""Outcome monitoring: expected -> actual (ch16 'Outcome monitoring', ch17 'Outcome learning').

implementation_status: FROZEN_CONTRACT (record) / deterministic
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from conrad.decision.context import DecisionSummary
from conrad.schemas.base import ConradModel
from conrad.schemas.decision import ActionType, DecisionRecord


class OutcomeRecord(ConradModel):
    decision_id: UUID
    action_type: ActionType | None
    expected: dict[str, Any]
    actual: dict[str, Any] | None = None
    discrepancies: dict[str, float] = Field(default_factory=dict)
    within_tolerance: bool | None = None
    reported_time_ns: int | None = None


class OutcomeMonitor:
    def __init__(self, tolerance: float = 0.25, history_limit: int = 64) -> None:
        self.tolerance = tolerance
        self.history_limit = history_limit
        self._open: dict[UUID, OutcomeRecord] = {}
        self._records: dict[UUID, DecisionRecord] = {}
        self.closed: list[OutcomeRecord] = []

    def register(self, record: DecisionRecord) -> OutcomeRecord:
        expected = {} if record.chosen is None else dict(record.chosen.expected_outcome)
        outcome = OutcomeRecord(
            decision_id=record.decision_id,
            action_type=None if record.chosen is None else record.chosen.action_type,
            expected=expected,
        )
        self._open[record.decision_id] = outcome
        self._records[record.decision_id] = record
        return outcome

    def report(self, decision_id: UUID, actual: dict[str, Any], time_ns: int) -> OutcomeRecord:
        if decision_id not in self._open:
            raise KeyError(f"no open decision {decision_id}")
        pending = self._open.pop(decision_id)
        diffs: dict[str, float] = {}
        for key, want in pending.expected.items():
            got = actual.get(key)
            if isinstance(want, int | float) and isinstance(got, int | float):
                diffs[key] = float(got) - float(want)
        ok = all(abs(d) <= self.tolerance for d in diffs.values()) if diffs else None
        closed = pending.model_copy(
            update={
                "actual": dict(actual),
                "discrepancies": diffs,
                "within_tolerance": ok,
                "reported_time_ns": time_ns,
            }
        )
        self.closed.append(closed)
        return closed

    @property
    def open_decisions(self) -> tuple[UUID, ...]:
        return tuple(self._open)

    def history(self) -> tuple[DecisionSummary, ...]:
        """Compact H_t for the next DecisionContext (most recent last)."""
        outcomes = {o.decision_id: o.within_tolerance for o in self.closed}
        out = []
        for record in list(self._records.values())[-self.history_limit :]:
            out.append(
                DecisionSummary(
                    decision_id=record.decision_id,
                    time_ns=record.timestamp.time_ns,
                    action_type=None if record.chosen is None else record.chosen.action_type,
                    target_belief_ids=() if record.chosen is None else record.chosen.target_belief_ids,
                    abstained=record.abstained,
                    outcome_ok=outcomes.get(record.decision_id),
                )
            )
        return tuple(out)

    def mean_abs_discrepancy(self) -> dict[str, float]:
        sums: dict[str, list[float]] = {}
        for o in self.closed:
            for k, v in o.discrepancies.items():
                sums.setdefault(k, []).append(abs(v))
        return {k: sum(v) / len(v) for k, v in sums.items()}
