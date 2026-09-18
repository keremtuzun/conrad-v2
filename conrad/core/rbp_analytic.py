"""Analytic relational propagation baseline over interpretable estimates (ch9, H-CORE-03 baseline).

Rules enforced here:
* results are INFERRED, never OBSERVED;
* a directly OBSERVED property is never overwritten by a relational message;
* observational uncertainty / coverage / independence counts are untouched: a neighbour's state is
  not a sensing event on this entity;
* a message is discounted by the source's own uncertainty and by the edge confidence.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from uuid import UUID

from conrad.core.config import RbpConfig
from conrad.core.state import AnalyticBeliefState, PropertyEstimate
from conrad.schemas.belief import KnowledgeStatus, Relationship


@dataclass(frozen=True)
class RelationalInference:
    belief_id: UUID
    state: AnalyticBeliefState
    source_belief_ids: tuple[UUID, ...]
    relationship_ids: tuple[UUID, ...]
    inferred_properties: tuple[str, ...]
    disagreeing_properties: tuple[str, ...]


def _message(src: PropertyEstimate, rel: Relationship, cfg: RbpConfig) -> tuple[float, float] | None:
    """(mean, variance) implied for the target, or None when the edge carries no usable strength."""
    strength = rel.confidence * cfg.analytic_gain
    if strength <= 0:
        return None
    rel_var = rel.attributes.get("relation_variance", 0.0)
    offset = rel.attributes.get("offset", 0.0)
    return src.mean + offset, (src.variance + rel_var) / strength


class AnalyticRBP:
    def __init__(self, config: RbpConfig | None = None, conflict_sigma: float = 3.0) -> None:
        self.cfg = config or RbpConfig()
        self.conflict_sigma = conflict_sigma

    def propagate(
        self, states: Mapping[UUID, AnalyticBeliefState], relationships: Sequence[Relationship]
    ) -> list[RelationalInference]:
        """One hop over directed edges source -> target. Sources are read from the INPUT states only,
        so the result does not depend on edge order."""
        incoming: dict[UUID, list[Relationship]] = {}
        for rel in sorted(relationships, key=lambda r: r.relationship_id.int):
            if rel.source_belief_id in states and rel.target_belief_id in states:
                incoming.setdefault(rel.target_belief_id, []).append(rel)
        out: list[RelationalInference] = []
        for target_id in sorted(incoming, key=lambda b: b.int):
            inference = self._infer(target_id, states, incoming[target_id])
            if inference is not None:
                out.append(inference)
        return out

    def _infer(
        self, target_id: UUID, states: Mapping[UUID, AnalyticBeliefState], rels: Sequence[Relationship]
    ) -> RelationalInference | None:
        target = states[target_id]
        pooled: dict[str, tuple[float, float, str | None]] = {}  # name -> (precision*mean, precision, units)
        used_rel: dict[str, set[UUID]] = {}
        used_src: dict[str, set[UUID]] = {}
        for rel in rels:
            for name, est in states[rel.source_belief_id].estimates.items():
                msg = _message(est, rel, self.cfg)
                if msg is None:
                    continue
                mean, var = msg
                pm, p, _ = pooled.get(name, (0.0, 0.0, est.units))
                pooled[name] = (pm + mean / var, p + 1.0 / var, est.units)
                used_rel.setdefault(name, set()).add(rel.relationship_id)
                used_src.setdefault(name, set()).add(rel.source_belief_id)
        estimates = dict(target.estimates)
        inferred: list[str] = []
        disagree: list[str] = []
        for name in sorted(pooled):
            pm, p, units = pooled[name]
            m_mean, m_var = pm / p, 1.0 / p
            existing = estimates.get(name)
            if existing is not None and existing.status is KnowledgeStatus.OBSERVED:
                nu = abs(m_mean - existing.mean) / math.sqrt(existing.variance + m_var)
                if nu > self.conflict_sigma:
                    disagree.append(name)
                continue  # direct observation outranks relational inference
            if existing is None or existing.status is KnowledgeStatus.INFERRED:
                # a previous relational estimate is REPLACED, never fused again (no double counting)
                estimates[name] = PropertyEstimate(m_mean, m_var, units, KnowledgeStatus.INFERRED)
            else:
                k = existing.variance / (existing.variance + m_var)
                estimates[name] = PropertyEstimate(
                    existing.mean + k * (m_mean - existing.mean),
                    (1 - k) * existing.variance,
                    existing.units or units,
                    KnowledgeStatus.INFERRED,
                )
            inferred.append(name)
        if not inferred:
            return None
        rel_ids = sorted({r for n in inferred for r in used_rel[n]}, key=lambda u: u.int)
        src_ids = sorted({s for n in inferred for s in used_src[n]}, key=lambda u: u.int)
        # uo / coverage / independence_groups deliberately unchanged
        new_state = replace(target, estimates=estimates)
        return RelationalInference(
            target_id, new_state, tuple(src_ids), tuple(rel_ids), tuple(inferred), tuple(disagree)
        )
