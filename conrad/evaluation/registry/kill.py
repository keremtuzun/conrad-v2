"""Mandatory comparison and kill protocol (ch35 Priority 5, ch36 Model governance).

Four separate questions: task error, calibration / unsupported confidence, OOD, cost. Automatic
KILL_CANDIDATE: no repeatable benefit, calibration regression, leakage, safety / deadline
regression, or cost outside budget without a documented mission benefit. Retention needs an ADR.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from conrad.evaluation.metrics.bootstrap import PairedComparison
from conrad.schemas.base import ConradModel


class Answer(str, Enum):
    YES = "YES"
    NO = "NO"
    REGRESSION = "REGRESSION"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class Verdict(str, Enum):
    RETAIN_CANDIDATE = "RETAIN_CANDIDATE"
    KILL_CANDIDATE = "KILL_CANDIDATE"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    RETAINED_BY_ADR = "RETAINED_BY_ADR"


class CostReport(ConradModel):
    latency_ms: float | None = None
    memory_mb: float | None = None
    compute_ratio_vs_baseline: float | None = None
    budget_latency_ms: float | None = Field(
        default=None, description="RobotConfig-specific budget; None = OPEN"
    )
    budget_memory_mb: float | None = None
    documented_mission_benefit: str | None = None


class MechanismComparison(ConradModel):
    mechanism: str
    baseline: str
    matched: bool = Field(
        description="same manifest, scenarios, seeds, compute budget, interface, metric code"
    )
    task_error: PairedComparison | None = None
    calibration: PairedComparison | None = None
    ood: PairedComparison | None = None
    cost: CostReport = CostReport()
    leakage_detected: bool = False
    safety_regression: bool = False
    deadline_regression: bool = False


class KillReport(ConradModel):
    mechanism: str
    baseline: str
    task_error: Answer
    calibration: Answer
    ood: Answer
    cost_justified: Answer
    verdict: Verdict
    reasons: tuple[str, ...]
    adr_ref: str | None = None
    mission_condition: str | None = None
    compensating_control: str | None = None


def _answer(comparison: PairedComparison | None) -> Answer:
    if comparison is None:
        return Answer.NOT_EVALUABLE
    if comparison.repeatable_regression:
        return Answer.REGRESSION
    return Answer.YES if comparison.repeatable_benefit else Answer.NO


def _cost(cost: CostReport) -> tuple[Answer, list[str]]:
    over: list[str] = []
    if (
        cost.budget_latency_ms is not None
        and cost.latency_ms is not None
        and cost.latency_ms > cost.budget_latency_ms
    ):
        over.append(f"latency {cost.latency_ms} ms > budget {cost.budget_latency_ms} ms")
    if (
        cost.budget_memory_mb is not None
        and cost.memory_mb is not None
        and cost.memory_mb > cost.budget_memory_mb
    ):
        over.append(f"memory {cost.memory_mb} MB > budget {cost.budget_memory_mb} MB")
    if over:
        return (Answer.YES if cost.documented_mission_benefit else Answer.NO), over
    if cost.budget_latency_ms is None or cost.budget_memory_mb is None or cost.latency_ms is None:
        return Answer.NOT_EVALUABLE, []
    return Answer.YES, []


def assess(c: MechanismComparison) -> KillReport:
    task, calib, ood = _answer(c.task_error), _answer(c.calibration), _answer(c.ood)
    cost_answer, over = _cost(c.cost)
    kill: list[str] = []
    if c.leakage_detected:
        kill.append("truth/provenance leakage")
    if c.safety_regression:
        kill.append("safety regression")
    if c.deadline_regression:
        kill.append("deadline / real-time regression")
    if calib is Answer.REGRESSION:
        kill.append("calibration or unsupported-confidence regression")
    if task in (Answer.NO, Answer.REGRESSION):
        kill.append("no repeatable task-error benefit over the matched baseline")
    if cost_answer is Answer.NO:
        kill.append("cost outside budget without documented mission benefit: " + "; ".join(over))
    if kill:
        verdict = Verdict.KILL_CANDIDATE
    elif not c.matched:
        verdict, kill = Verdict.NOT_EVALUABLE, ["comparison is not matched"]
    elif Answer.NOT_EVALUABLE in (task, calib, ood, cost_answer):
        verdict = Verdict.NOT_EVALUABLE
        kill = [
            f"{n} not evaluable"
            for n, a in (("task_error", task), ("calibration", calib), ("ood", ood), ("cost", cost_answer))
            if a is Answer.NOT_EVALUABLE
        ]
    else:
        verdict = Verdict.RETAIN_CANDIDATE
    return KillReport(
        mechanism=c.mechanism,
        baseline=c.baseline,
        task_error=task,
        calibration=calib,
        ood=ood,
        cost_justified=cost_answer,
        verdict=verdict,
        reasons=tuple(kill),
    )


def retain_with_adr(
    report: KillReport, *, adr_ref: str, mission_condition: str, compensating_control: str
) -> KillReport:
    """The only way to keep a KILL_CANDIDATE: an ADR naming the mission condition and control."""
    if report.verdict is not Verdict.KILL_CANDIDATE:
        raise ValueError("only a KILL_CANDIDATE needs ADR retention")
    if not (adr_ref.strip() and mission_condition.strip() and compensating_control.strip()):
        raise ValueError("ADR retention needs adr_ref, mission_condition and compensating_control")
    return report.model_copy(
        update={
            "verdict": Verdict.RETAINED_BY_ADR,
            "adr_ref": adr_ref,
            "mission_condition": mission_condition,
            "compensating_control": compensating_control,
        }
    )
