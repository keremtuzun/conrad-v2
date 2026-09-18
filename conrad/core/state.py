"""Interpretable per-belief state used by the analytic operators, and its mapping to PropertyClaims.

A belief's interpretable state is a set of scalar Gaussian property estimates plus the explicit
four-channel uncertainty. It is derived from evidence MEASUREMENTS, never from evidence embeddings.

Contract gap: ``PropertyClaim`` has no variance field, so variances persist as companion claims
named ``<property>.variance`` and the contradiction memory as ``contradiction.evidence_ids``.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from uuid import UUID

from conrad.schemas.belief import KnowledgeStatus, PropertyClaim
from conrad.schemas.uncertainty import Uncertainty

VARIANCE_SUFFIX = ".variance"
CONTRADICTION_CLAIM = "contradiction.evidence_ids"


@dataclass(frozen=True)
class PropertyEstimate:
    mean: float
    variance: float
    units: str | None = None
    status: KnowledgeStatus = KnowledgeStatus.OBSERVED


@dataclass(frozen=True)
class ContradictionEntry:
    """Unresolved credible disagreement kept in memory instead of being averaged away (ch5)."""

    evidence_id: UUID
    property_name: str
    measured_value: float
    normalised_innovation: float
    reliability: float


@dataclass(frozen=True)
class AnalyticBeliefState:
    estimates: dict[str, PropertyEstimate] = field(default_factory=dict)
    ua: float = 0.0
    ue: float = 1.0
    uc: float = 0.0
    uo: float = 1.0
    coverage: float = 0.0
    independence_groups: frozenset[str] = frozenset()
    consumed_evidence_ids: frozenset[UUID] = frozenset()
    contradictions: tuple[ContradictionEntry, ...] = ()

    @property
    def channels(self) -> tuple[float, float, float, float]:
        return (self.ua, self.ue, self.uc, self.uo)

    @property
    def independent_observation_count(self) -> int:
        return len(self.independence_groups)

    def with_status(self, status: KnowledgeStatus) -> AnalyticBeliefState:
        return replace(self, estimates={k: replace(v, status=status) for k, v in self.estimates.items()})


def unknown_state() -> AnalyticBeliefState:
    """Never-observed prior: no estimates, UE and UO maximal on the [0, 1] scale."""
    return AnalyticBeliefState()


def state_to_claims(
    state: AnalyticBeliefState, uncertainty: Uncertainty, provenance_id: UUID
) -> tuple[PropertyClaim, ...]:
    claims: list[PropertyClaim] = []
    for name in sorted(state.estimates):
        est = state.estimates[name]
        claims.append(
            PropertyClaim(
                name=name,
                value=est.mean,
                units=est.units,
                status=est.status,
                uncertainty=uncertainty,
                provenance_id=provenance_id,
            )
        )
        claims.append(
            PropertyClaim(
                name=name + VARIANCE_SUFFIX,
                value=est.variance,
                units=None if est.units is None else f"({est.units})^2",
                status=est.status,
                uncertainty=uncertainty,
                provenance_id=provenance_id,
            )
        )
    if state.contradictions and claims:
        claims.append(
            PropertyClaim(
                name=CONTRADICTION_CLAIM,
                value=",".join(sorted({f"{c.evidence_id}:{c.property_name}" for c in state.contradictions})),
                status=claims[0].status,
                uncertainty=uncertainty,
                provenance_id=provenance_id,
            )
        )
    return tuple(claims)


def estimates_from_claims(claims: tuple[PropertyClaim, ...]) -> dict[str, PropertyEstimate]:
    """Rebuild estimates from persisted claims (used after a working-memory reset, CC-09)."""
    by_name = {c.name: c for c in claims}
    out: dict[str, PropertyEstimate] = {}
    for name, claim in by_name.items():
        if name.endswith(VARIANCE_SUFFIX) or name == CONTRADICTION_CLAIM:
            continue
        var_claim = by_name.get(name + VARIANCE_SUFFIX)
        if isinstance(claim.value, bool) or not isinstance(claim.value, (int, float)):
            continue
        if var_claim is None or isinstance(var_claim.value, (bool, str)) or var_claim.value is None:
            continue
        out[name] = PropertyEstimate(float(claim.value), float(var_claim.value), claim.units, claim.status)
    return out


def contradiction_refs_from_claims(claims: tuple[PropertyClaim, ...]) -> tuple[tuple[UUID, str], ...]:
    """Persisted contradiction memory as (evidence_id, property_name) pairs."""
    for c in claims:
        if c.name == CONTRADICTION_CLAIM and isinstance(c.value, str) and c.value:
            pairs = (part.split(":", 1) for part in c.value.split(","))
            return tuple((UUID(eid), prop) for eid, prop in pairs)
    return ()
