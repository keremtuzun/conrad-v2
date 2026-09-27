"""P7 cross-domain uncertainty and hypothesis reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from conrad.schemas.belief import BeliefMessage
from conrad.schemas.uncertainty import Uncertainty


def provenance_aware_uncertainty_update(base: Uncertainty, evidence: Uncertainty, independent: bool) -> Uncertainty:
    """Update four channels without collapsing them into a single confidence scalar."""

    independence_gain = 0.7 if independent else 0.95
    contradiction = max(base.contradiction, evidence.contradiction)
    if not independent:
        contradiction = max(contradiction, min(1.0, base.contradiction + 0.05))
    return Uncertainty(
        aleatoric=max(base.aleatoric, evidence.aleatoric),
        epistemic=max(0.0, min(base.epistemic, evidence.epistemic) * independence_gain),
        contradiction=contradiction,
        observational=max(0.0, min(base.observational, evidence.observational) * independence_gain),
    )


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: UUID
    statement: str
    support: float
    contradiction: float
    source_belief_ids: tuple[UUID, ...]

    @property
    def score(self) -> float:
        return self.support - self.contradiction


@dataclass(frozen=True)
class CompetingHypotheses:
    hypotheses: tuple[Hypothesis, ...]

    def ranked(self) -> tuple[Hypothesis, ...]:
        return tuple(sorted(self.hypotheses, key=lambda h: (-h.score, str(h.hypothesis_id))))

    def unresolved(self, margin: float = 0.1) -> bool:
        ranked = self.ranked()
        return len(ranked) >= 2 and abs(ranked[0].score - ranked[1].score) < margin


class CrossDomainReasoner:
    """Relates belief messages across domains without fabricating causal truth."""

    version = "p7-cross-domain-reasoner-v0"

    def contradiction_between(self, left: BeliefMessage, right: BeliefMessage) -> float:
        l_state = {c.name: c.value for c in left.state_summary}
        r_state = {c.name: c.value for c in right.state_summary}
        overlap = set(l_state) & set(r_state)
        if not overlap:
            return max(left.uncertainty.contradiction, right.uncertainty.contradiction)
        mismatches = sum(1 for key in overlap if l_state[key] != r_state[key])
        return max(left.uncertainty.contradiction, right.uncertainty.contradiction, mismatches / len(overlap))

    def build_hypotheses(self, beliefs: tuple[BeliefMessage, ...], ids: tuple[UUID, ...]) -> CompetingHypotheses:
        hypotheses: list[Hypothesis] = []
        for idx, belief in enumerate(beliefs):
            support = max(0.0, 1.0 - belief.uncertainty.epistemic - belief.uncertainty.observational * 0.5)
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=ids[idx],
                    statement=f"{belief.domain.value}:{belief.knowledge_status.value}",
                    support=support,
                    contradiction=belief.uncertainty.contradiction,
                    source_belief_ids=(belief.belief_id,),
                )
            )
        return CompetingHypotheses(tuple(hypotheses))
