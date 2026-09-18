"""AnalyticBUO: non-learned, precision-weighted belief update with explicit UA/UE/UC/UO arithmetic.

This is the DEFAULT runtime operator until a learned BUO is validated (brief rule 11). It works on
interpretable ``Evidence.measurements`` only. Constants are ENGINEERING_ESTIMATE configuration.

Behaviour guaranteed by tests: corrupted evidence -> UA up; OOD score -> UE up; credible
disagreement between reliable evidence -> UC up (never averaged away); low coverage -> UO up;
duplicates ignored; repeated same-source evidence never manufactures certainty.

implementation_status: EXPERIMENTAL_CANDIDATE (baseline family: Bayesian/Kalman-style fusion, H-CORE-02)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from uuid import UUID

from conrad.core.config import AnalyticBuoConfig
from conrad.core.state import AnalyticBeliefState, ContradictionEntry, PropertyEstimate
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.observation import Evidence, EvidenceValidity


@dataclass(frozen=True)
class EvidenceAssessment:
    """Per-evidence diagnostics. ``trust`` and ``innovation`` are separate quantities (H-CORE-02)."""

    evidence_id: UUID
    trust: float
    innovation: float
    independent: bool
    outcome: str  # SUPPORT | CONFLICT | AMBIGUOUS | NO_MEASUREMENT | INVALID


@dataclass(frozen=True)
class AnalyticUpdateResult:
    state: AnalyticBeliefState
    consumed_evidence_ids: tuple[UUID, ...]
    rejected_duplicate_ids: tuple[UUID, ...]
    support_ids: tuple[UUID, ...]
    conflict_ids: tuple[UUID, ...]
    assessments: tuple[EvidenceAssessment, ...]
    temporal_surprise: float

    @property
    def changed(self) -> bool:
        return bool(self.consumed_evidence_ids)


def evidence_trust(ev: Evidence, cfg: AnalyticBuoConfig) -> float:
    """Credibility of the evidence as evidence. Independent of agreement with the belief."""
    if ev.validity is EvidenceValidity.INVALID:
        return 0.0
    scale = 0.5 if ev.validity is EvidenceValidity.DEGRADED else 1.0
    return max(0.0, min(1.0, ev.reliability * scale))


def measurement_variance(ev: Evidence, cfg: AnalyticBuoConfig) -> float:
    trust = max(evidence_trust(ev, cfg), cfg.reliability_floor)
    return max(ev.aleatoric_uncertainty, cfg.min_variance) / trust


class AnalyticBUO:
    def __init__(self, config: AnalyticBuoConfig | None = None) -> None:
        self.cfg = config or AnalyticBuoConfig()

    # ------------------------------------------------------------------ public
    def update(self, state: AnalyticBeliefState, evidence: Sequence[Evidence]) -> AnalyticUpdateResult:
        """Fuse simultaneous evidence. Order of ``evidence`` does not change the result class
        (evidence is processed in a canonical order so the operator is permutation invariant)."""
        cfg = self.cfg
        fresh, duplicates = self._reject_duplicates(state, evidence)
        if not fresh:
            return AnalyticUpdateResult(state, (), tuple(duplicates), (), (), (), 0.0)

        ordered = sorted(fresh, key=lambda e: (e.timestamp.time_ns, e.evidence_id.int))
        groups = set(state.independence_groups)
        estimates = dict(state.estimates)
        contradictions = list(state.contradictions)
        coverage, uc = state.coverage, state.uc
        support: list[UUID] = []
        conflict: list[UUID] = []
        assessments: list[EvidenceAssessment] = []
        surprises: list[float] = []

        for ev in ordered:
            trust = evidence_trust(ev, cfg)
            group = ev.independence_group or str(ev.evidence_id)
            independent = group not in groups
            if ev.validity is EvidenceValidity.INVALID:
                assessments.append(EvidenceAssessment(ev.evidence_id, trust, 0.0, False, "INVALID"))
                continue
            worst = 0.0
            outcome = "NO_MEASUREMENT"
            for name in sorted(ev.measurements):
                est, nu, kind = self._fuse(estimates.get(name), ev, name, independent)
                estimates[name] = est
                worst = max(worst, nu)
                if kind == "CONFLICT":
                    outcome = "CONFLICT"
                    excess = nu / cfg.contradiction_sigma - 1.0
                    uc += cfg.contradiction_gain * trust * (1.0 + excess)
                    contradictions.append(
                        ContradictionEntry(ev.evidence_id, name, ev.measurements[name], nu, ev.reliability)
                    )
                else:
                    if kind == "SUPPORT" and independent and trust >= cfg.credible_reliability:
                        # independent credible agreement resolves remembered disagreement on this property
                        contradictions = [c for c in contradictions if c.property_name != name]
                    if outcome != "CONFLICT":
                        outcome = kind if outcome in ("NO_MEASUREMENT", kind) else "AMBIGUOUS"
            if outcome == "SUPPORT":
                support.append(ev.evidence_id)
                if independent:
                    uc *= 1.0 - cfg.contradiction_relief * trust
            elif outcome == "CONFLICT":
                conflict.append(ev.evidence_id)
            if independent:
                groups.add(group)
                occlusion = ev.sensor_context.occlusion or 0.0
                gain = cfg.coverage_per_independent_obs * (1.0 - occlusion) * trust
                coverage = 1.0 - (1.0 - coverage) * (1.0 - gain)
            surprises.append(worst)
            assessments.append(EvidenceAssessment(ev.evidence_id, trust, worst, independent, outcome))

        new_state = replace(
            state,
            estimates=estimates,
            ua=self._ua(state, ordered),
            ue=self._ue(state, ordered),
            uc=max(uc, 0.0),
            uo=max(0.0, 1.0 - coverage),
            coverage=coverage,
            independence_groups=frozenset(groups),
            consumed_evidence_ids=state.consumed_evidence_ids | {e.evidence_id for e in ordered},
            contradictions=tuple(contradictions[-cfg.max_contradiction_memory :]),
        )
        return AnalyticUpdateResult(
            new_state,
            tuple(e.evidence_id for e in ordered),
            tuple(duplicates),
            tuple(support),
            tuple(conflict),
            tuple(assessments),
            max(surprises) if surprises else 0.0,
        )

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _reject_duplicates(
        state: AnalyticBeliefState, evidence: Sequence[Evidence]
    ) -> tuple[list[Evidence], list[UUID]]:
        seen = set(state.consumed_evidence_ids)
        fresh: list[Evidence] = []
        duplicates: list[UUID] = []
        for ev in evidence:
            if ev.evidence_id in seen:
                duplicates.append(ev.evidence_id)
            else:
                seen.add(ev.evidence_id)
                fresh.append(ev)
        return fresh, duplicates

    def _fuse(
        self, prior: PropertyEstimate | None, ev: Evidence, name: str, independent: bool
    ) -> tuple[PropertyEstimate, float, str]:
        cfg = self.cfg
        y = ev.measurements[name]
        r = measurement_variance(ev, cfg)
        units = ev.measurement_units.get(name)
        if prior is None:
            return PropertyEstimate(y, r, units, KnowledgeStatus.OBSERVED), 0.0, "SUPPORT"
        innov = y - prior.mean
        nu = abs(innov) / math.sqrt(prior.variance + r)
        k = prior.variance / (prior.variance + r)
        credible = ev.reliability >= cfg.credible_reliability and ev.validity is EvidenceValidity.VALID
        if nu > cfg.contradiction_sigma:
            if not credible:
                # unreliable disagreement: ambiguous, inflated R already limits its influence
                mean = prior.mean + k * innov
                var = prior.variance if not independent else max((1 - k) * prior.variance, cfg.min_variance)
                return PropertyEstimate(mean, max(var, prior.variance), units), nu, "AMBIGUOUS"
            # credible contradiction: moment-matched two-hypothesis mixture; variance GROWS
            w = k * cfg.contradiction_state_weight
            mean = prior.mean + w * innov
            var = (1 - w) * prior.variance + w * r + w * (1 - w) * innov * innov
            return PropertyEstimate(mean, var, units), nu, "CONFLICT"
        mean = prior.mean + k * innov
        if independent:
            var = max((1 - k) * prior.variance, cfg.min_variance)
        else:
            # same independence group: correlated repeat, never below the best single source
            var = min(prior.variance, r)
        return PropertyEstimate(mean, var, units), nu, "SUPPORT"

    def _ua(self, state: AnalyticBeliefState, evidence: Sequence[Evidence]) -> float:
        observed = [ev.aleatoric_uncertainty + (1.0 - evidence_trust(ev, self.cfg)) for ev in evidence]
        target = sum(observed) / len(observed)
        if not state.consumed_evidence_ids:
            return target
        s = self.cfg.ua_smoothing
        return (1 - s) * state.ua + s * target

    def _ue(self, state: AnalyticBeliefState, evidence: Sequence[Evidence]) -> float:
        scores = [ev.sensor_context.ood_score for ev in evidence if ev.sensor_context.ood_score is not None]
        if not scores:
            return state.ue  # OOD was not measured; never replaced by a nominal value
        target = max(scores)
        if not state.consumed_evidence_ids:
            return target
        s = self.cfg.ue_smoothing
        return (1 - s) * state.ue + s * target
