"""UIR instrumentation (ch16 'Unsupported Inference Rate', ch17 'Unsupported-inference metric').

UIR = #unsupported world claims relied on / #world claims relied on. Target UIR -> 0.
"""

from __future__ import annotations

from collections.abc import Iterable

from conrad.schemas.base import ConradModel
from conrad.schemas.decision import WORLD_DEPENDENT_CLAIMS, DecisionRecord, GroundingStatus


class UIRReport(ConradModel):
    """UIR instrumentation (ch16 'Unsupported Inference Rate', ch17 'Unsupported-inference metric')."""

    world_claims: int
    unsupported_world_claims: int
    relied_world_claims: int
    relied_unsupported_claims: int
    unsupported_inference_rate: float
    unsupported_claim_fraction: float


def uir_report(records: Iterable[DecisionRecord]) -> UIRReport:
    """Count world-dependent claims a decision RELIED ON (``chosen.supporting_claims``) that are UNSUPPORTED.

    A claim that is honestly labelled UNSUPPORTED and only motivates information acquisition is not an
    unsupported inference; using it as a supporting fact is. ``unsupported_claim_fraction`` is the raw
    share of UNSUPPORTED world claims in the graphs and measures upstream quality, not Model 1.
    """
    world = unsupported = relied = relied_unsupported = 0
    for record in records:
        by_id = {c.claim_id: c for c in record.claims}
        for c in record.claims:
            if c.claim_type in WORLD_DEPENDENT_CLAIMS:
                world += 1
                unsupported += c.grounding is GroundingStatus.UNSUPPORTED
        if record.chosen is None:
            continue
        for claim_id in record.chosen.supporting_claims:
            claim = by_id.get(claim_id)
            if claim is None:
                # a world claim the record cannot show is by definition not traceable
                relied += 1
                relied_unsupported += 1
            elif claim.claim_type in WORLD_DEPENDENT_CLAIMS:
                relied += 1
                relied_unsupported += claim.grounding is GroundingStatus.UNSUPPORTED
    return UIRReport(
        world_claims=world,
        unsupported_world_claims=unsupported,
        relied_world_claims=relied,
        relied_unsupported_claims=relied_unsupported,
        unsupported_inference_rate=relied_unsupported / relied if relied else 0.0,
        unsupported_claim_fraction=unsupported / world if world else 0.0,
    )


def unsupported_inference_rate(records: Iterable[DecisionRecord]) -> float:
    return uir_report(records).unsupported_inference_rate
