"""Interchangeable ranking policies behind one interface (ch16 'Structured reasoning architecture').

The I/O is frozen: input = claim graph + legal candidates, output = ranked proposals with their
supporting claims. A policy can only RANK. It cannot add an action type, cannot mark a claim
grounded, and cannot override the ConstraintEngine.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from conrad.decision.claims import ClaimGraph
from conrad.decision.config import DecisionConfig
from conrad.decision.consequence import ConsequenceVector
from conrad.decision.context import DecisionContext
from conrad.schemas.decision import ActionProposal, ActionType


class DecisionPolicy(Protocol):
    name: str

    def rank(
        self,
        graph: ClaimGraph,
        candidates: Sequence[ActionProposal],
        consequences: Sequence[ConsequenceVector],
        ctx: DecisionContext,
    ) -> list[ActionProposal]:
        """Return the candidates with ``score``/``priority`` filled, best first. Must be deterministic."""
        ...


def _with_score(action: ActionProposal, score: float, c: ConsequenceVector) -> ActionProposal:
    priority = min(1.0, max(0.0, max(c.mission, c.information)))
    return action.model_copy(
        update={
            "score": float(score),
            "priority": priority,
            "expected_outcome": {
                "mission": c.mission,
                "information": c.information,
                "risk": c.risk,
                "uncertainty_exposure": c.uncertainty_exposure,
                "time": c.time,
                "energy": c.energy,
            },
        }
    )


def _stable_sort(actions: list[ActionProposal]) -> list[ActionProposal]:
    order = {t: i for i, t in enumerate(ActionType)}
    return sorted(actions, key=lambda a: (-a.score, order[a.action_type], a.action_id.int))


class StructuredReasoningPolicy:
    """Default runtime policy: configured utility over the consequence vector. No learned parameter."""

    name = "structured_reasoning"

    def __init__(self, config: DecisionConfig) -> None:
        self.config = config

    def rank(
        self,
        graph: ClaimGraph,
        candidates: Sequence[ActionProposal],
        consequences: Sequence[ConsequenceVector],
        ctx: DecisionContext,
    ) -> list[ActionProposal]:
        scored = [
            _with_score(a, c.scalarize(self.config.utility), c)
            for a, c in zip(candidates, consequences, strict=True)
        ]
        return _stable_sort(scored)


class NaiveActOnClaimsPolicy:
    """M1-UIR-E001 baseline: reads the claimed value, ignores grounding, staleness and uncertainty cause.

    It continues unless a claimed value itself looks alarming, and it cites every belief claim it
    looked at as support, including UNSUPPORTED ones. Never use this outside the experiment.
    """

    name = "naive_act_on_claims"

    def __init__(self, alarm_values: Sequence[str] = ("DAMAGED", "SEVERE", "CRITICAL")) -> None:
        self.alarm_values = {v.upper() for v in alarm_values}

    def rank(
        self,
        graph: ClaimGraph,
        candidates: Sequence[ActionProposal],
        consequences: Sequence[ConsequenceVector],
        ctx: DecisionContext,
    ) -> list[ActionProposal]:
        alarming = any(
            isinstance(c.structured_value.get("value"), str)
            and str(c.structured_value["value"]).upper() in self.alarm_values
            for c in graph.world_claims()
        )
        scored = []
        for a, c in zip(candidates, consequences, strict=True):
            score = 0.0
            if a.action_type is ActionType.CONTINUE_MISSION:
                score = 0.2 if alarming else 1.0
            elif a.action_type is ActionType.ESCALATE_TO_OPERATOR:
                score = 1.0 if alarming else 0.1
            scored.append(_with_score(a, score, c))
        return _stable_sort(scored)
