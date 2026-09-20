"""Gate I6 multi-domain reasoning: 2E imaging conditions join Model 1's claim graph through the Belief Bus.

DEPLOYMENT PLANE. Two pieces, both switched on only by ``MissionRuntimeConfig.multidomain.enabled`` (off by default,
so every other gate's mission is unchanged):

* ``sensing_requirements``: for every critical TECHNICAL inspection requirement, one ECOLOGICAL requirement "imaging
  conditions at the same inspection station". Its primary belief is Model2E's turbidity field belief (a belief ID
  minted by the 2E child, known to its own runtime) and its context domain is SPATIAL. Its consequence is set below
  ``DecisionConfig.consequence_matters_above``, so it never drives sensing actions by itself: it only puts grounded
  2E claims (and 2S coverage context) into every EGDC claim graph next to the 2T claims.
* ``SensingConditionsGate``: before an EGDC information request on a critical component is sent to MCBR, the gate
  reads the 2E turbidity claims (mean and sd) of THAT decision's claim graph and computes the beam visibility of
  the upper credible turbidity (mean + k sd) at the closest feasible inspection range with the deployment's own
  optical model (Model2E's coupling constants). Below the
  configured minimum the inspection is DEFERRED: the request is not carried out, not counted as an attempt, and the
  deferral is recorded with the TECHNICAL, SPATIAL and ECOLOGICAL claim and belief IDs it rests on.

The gate never writes a belief. It reads claims Model 1 already built from bus snapshots.

Why the rule is here and not inside EGDC: ``conrad.decision`` is owned by the I5 workstream at the time of writing.
The rule consumes only the EGDC decision record, so it can move into the EGDC constraint/candidate stage unchanged.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from conrad.decision.context import MissionRequirement
from conrad.decision.egdc import DecisionOutcome
from conrad.orchestration.belief_bus import BeliefBus
from conrad.orchestration.multidomain_settings import MultiDomainSettings
from conrad.orchestration.services import RuntimeServices
from conrad.schemas.decision import ClaimType, DecisionClaim, GroundingStatus, InformationNeed
from conrad.schemas.events import EventType
from conrad.schemas.ids import IdFactory
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain

MODULE = "conrad.orchestration.multidomain"
MODEL_VERSION = "sensing-conditions-gate-0.1.0"
DEFERRED = "DEFERRED_LOW_VISIBILITY"
PROCEED = "PROCEED"
NO_BELIEF = "PROCEED_NO_GROUNDED_2E_BELIEF"


def sensing_requirements(
    requirements: Sequence[MissionRequirement],
    critical: Sequence[UUID],
    turbidity_belief_id: UUID,
    settings: MultiDomainSettings,
    ids: IdFactory,
) -> tuple[MissionRequirement, ...]:
    """One ECOLOGICAL imaging-conditions requirement per critical TECHNICAL requirement (same station)."""
    crit = set(critical)
    out = []
    for req in requirements:
        if req.domain is not Domain.TECHNICAL or not crit.intersection(req.target_entity_ids):
            continue
        out.append(
            MissionRequirement(
                requirement_id=ids.new(),
                description=f"imaging conditions at the inspection station of {req.description}",
                domain=Domain.ECOLOGICAL,
                target_belief_ids=(turbidity_belief_id,),
                region=req.region,
                properties=(settings.turbidity_property, settings.turbidity_sd_property),
                consequence=settings.sensing_consequence,
                context_domains=(Domain.SPATIAL,),
            )
        )
    return tuple(out)


def claim_domains(claims: Sequence[DecisionClaim], bus: BeliefBus) -> dict[str, list[UUID]]:
    """Belief-backed claims of a decision record grouped by the domain that OWNS each cited belief (bus heads)."""
    out: dict[str, list[UUID]] = {}
    for c in claims:
        for b in c.source_belief_ids:
            head = bus.head(b)
            if head is not None:
                out.setdefault(head.domain.value, []).append(c.claim_id)
                break
    return out


class SensingConditionsGate:
    def __init__(
        self,
        s: RuntimeServices,
        bus: BeliefBus,
        settings: MultiDomainSettings,
        attenuation_clear_per_m: float,
        attenuation_per_m_per_ntu: float,
    ) -> None:
        self.s, self.bus, self.cfg = s, bus, settings
        self.c0, self.c1 = attenuation_clear_per_m, attenuation_per_m_per_ntu
        self.records: list[dict[str, Any]] = []

    def visibility(self, turbidity_ntu: float) -> float:
        return math.exp(-(self.c0 + self.c1 * max(turbidity_ntu, 0.0)) * self.cfg.inspection_range_m)

    def _turbidity(self, claims: Sequence[DecisionClaim]) -> tuple[float | None, float | None, list[UUID]]:
        mean: float | None = None
        sd: float | None = None
        used: list[UUID] = []
        for c in claims:
            v = c.structured_value
            if (
                c.claim_type is not ClaimType.BELIEF_CLAIM
                or c.grounding is not GroundingStatus.GROUNDED
                or v.get("domain") != Domain.ECOLOGICAL.value
                or not isinstance(v.get("value"), (int, float))
            ):
                continue
            if v.get("property") == self.cfg.turbidity_property:
                mean = max(float(v["value"]), mean if mean is not None else -math.inf)
                used.append(c.claim_id)
            elif v.get("property") == self.cfg.turbidity_sd_property:
                sd = float(v["value"])
                used.append(c.claim_id)
        return mean, sd, used

    def check(self, need: InformationNeed, out: DecisionOutcome, now: TimeStamp) -> str | None:
        """Return a deferral reason, or None to let the information request go to MCBR."""
        rec = out.record
        turb, sd, eco_claims = self._turbidity(rec.claims)
        by_domain = claim_domains(rec.claims, self.bus)
        targets = set(need.target_belief_ids)
        tech_claims = [
            c.claim_id
            for c in rec.claims
            if c.claim_type is ClaimType.BELIEF_CLAIM and targets.intersection(c.source_belief_ids)
        ]
        spatial_claims = by_domain.get(Domain.SPATIAL.value, [])
        upper = None if turb is None else turb + self.cfg.turbidity_sd_multiplier * (sd or 0.0)
        vis = None if upper is None else self.visibility(upper)
        if vis is None:
            verdict = NO_BELIEF
        else:
            verdict = DEFERRED if vis < self.cfg.min_visibility else PROCEED
        cited = {
            "TECHNICAL": [str(c) for c in tech_claims],
            "SPATIAL": [str(c) for c in spatial_claims],
            "ECOLOGICAL": [str(c) for c in eco_claims],
        }
        beliefs = {
            d: sorted(
                {str(b) for c in rec.claims if str(c.claim_id) in set(cited[d]) for b in c.source_belief_ids}
            )
            for d in cited
        }
        prov = ProvenanceRecord(
            record_id=self.s.ids.new(),
            source_type=SourceType.DECISION,
            source_ids=(rec.decision_id, need.need_id),
            operation="orchestration.sensing_conditions_gate",
            module=MODULE,
            model_version=MODEL_VERSION,
            timestamp=now,
            parent_records=(out.provenance.record_id,),
            subject_id=rec.decision_id,
        )
        self.s.repo.put_provenance(self.s.run_id, [prov])
        row = {
            "t_s": now.time_ns / 1e9,
            "decision_id": str(rec.decision_id),
            "need_id": str(need.need_id),
            "verdict": verdict,
            "turbidity_ntu_belief": turb,
            "turbidity_sd_ntu_belief": sd,
            "turbidity_upper_ntu": upper,
            "visibility_at_range": vis,
            "inspection_range_m": self.cfg.inspection_range_m,
            "min_visibility": self.cfg.min_visibility,
            "claims": cited,
            "beliefs": beliefs,
            "provenance_id": str(prov.record_id),
        }
        self.records.append(row)
        if verdict == DEFERRED:
            self.s.emit(
                EventType.ACTION_REJECTED,
                MODULE,
                need.trace_id,
                {
                    "decision_id": str(rec.decision_id),
                    "need_id": str(need.need_id),
                    "route": "MCBR",
                    "reason": DEFERRED,
                    "visibility_at_range": vis,
                    "turbidity_ntu_belief": turb,
                    "provenance_id": str(prov.record_id),
                    "cited_beliefs": beliefs,
                },
            )
            return DEFERRED
        return None
