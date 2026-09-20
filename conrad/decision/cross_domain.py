"""Cross-domain reasoning that no single child owns (ch17 'Cross-domain reasoning').

2T may report a confident condition while 2S reports the region unobserved. Model 1 may conclude
"current evidence is insufficient", never "the pipe is damaged": the result is a CONTRADICTS edge
and an EXTEND_COVERAGE-worthy disagreement, not a new world claim.
"""

from __future__ import annotations

from uuid import UUID

from conrad.decision.config import DecisionConfig
from conrad.decision.context import DecisionContext, MissionRequirement
from conrad.decision.graph import ClaimGraph
from conrad.schemas.belief import BeliefMessage, KnowledgeStatus
from conrad.schemas.decision import ClaimEdgeType, ClaimType, DecisionClaim, GroundingStatus
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.ids import IdFactory


def cross_domain_disagreement(
    ids: IdFactory,
    config: DecisionConfig,
    ctx: DecisionContext,
    req: MissionRequirement,
    primary: list[BeliefMessage],
    graph: ClaimGraph,
    primary_claim_ids: list[UUID],
) -> bool:
    """2T says 'known' while 2S reports the region unobserved -> insufficient evidence, not damage."""
    if not primary or not req.context_domains:
        return False
    primary_confident = all(m.uncertainty.observational < config.thresholds.observational for m in primary)
    disagreement = False
    for domain in req.context_domains:
        for m in _context_for(req, ctx.beliefs(domain)):
            if m.spatial is None:
                continue
            unobserved = (
                m.spatial.coverage < config.min_spatial_coverage
                or m.knowledge_status is KnowledgeStatus.UNKNOWN
            )
            grounded = m.knowledge_status is not KnowledgeStatus.UNKNOWN
            node = DecisionClaim(
                claim_id=ids.new(),
                claim_type=ClaimType.BELIEF_CLAIM,
                statement=f"{domain.value} coverage context for requirement",
                structured_value={
                    "requirement_id": str(req.requirement_id),
                    "property": "coverage",
                    "value": m.spatial.coverage,
                    "domain": domain.value,
                    "context": True,
                    "issues": [],
                    "stale": False,
                },
                source_belief_ids=(m.belief_id,),
                source_belief_revisions=(m.revision,),
                evidence_refs=m.evidence_support,
                uncertainty=m.uncertainty,
                timestamp=m.timestamp,
                grounding=GroundingStatus.GROUNDED if grounded else GroundingStatus.UNSUPPORTED,
            )
            if not graph.add(node):
                continue
            if unobserved and primary_confident:
                disagreement = True
                for claim_id in primary_claim_ids:
                    graph.connect(node.claim_id, claim_id, ClaimEdgeType.CONTRADICTS)
    return disagreement


def _overlaps(a: SpatialSupport, b: SpatialSupport) -> bool:
    if a.frame_id != b.frame_id:
        return True  # cannot compare frames here: keep it (conservative)
    return all(
        abs(ca - cb) <= ha + hb
        for ca, cb, ha, hb in zip(a.center_m, b.center_m, a.half_extent_m, b.half_extent_m, strict=True)
    )


def _context_for(req: MissionRequirement, beliefs: tuple[BeliefMessage, ...]) -> list[BeliefMessage]:
    """The context beliefs that speak about THIS requirement (I5 iteration 2).

    A context belief associated with the requirement's registry component is the most specific statement about
    it, so when one exists only those are used. Otherwise only context beliefs whose support overlaps the
    requirement region count (a belief without a support, or a requirement without a region, is kept). Before
    this, every context belief in the snapshot counted, so a poorly covered block around ANOTHER component
    marked the critical requirement as contradicted for the whole mission."""
    own = [m for m in beliefs if m.world_entity_id is not None and m.world_entity_id in req.target_entity_ids]
    if own:
        return own
    if req.region is None:
        return list(beliefs)
    return [m for m in beliefs if m.spatial_support is None or _overlaps(m.spatial_support, req.region)]
