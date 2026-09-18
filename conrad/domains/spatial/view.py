"""BeliefQuery filtering over the latest published spatial messages. BELIEF PLANE.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from conrad.schemas.belief import BeliefMessage, BeliefQuery, KnowledgeStatus
from conrad.schemas.frames import FrameError, SpatialSupport
from conrad.schemas.world import Domain


def overlaps(a: SpatialSupport, b: SpatialSupport) -> bool:
    if a.frame_id != b.frame_id:
        raise FrameError(f"cannot intersect regions in {a.frame_id!r} and {b.frame_id!r}")
    ca, cb = np.asarray(a.center_m), np.asarray(b.center_m)
    ha, hb = np.asarray(a.half_extent_m), np.asarray(b.half_extent_m)
    return bool(np.all(np.abs(ca - cb) <= ha + hb))


def filter_messages(messages: Sequence[BeliefMessage], q: BeliefQuery, frame_id: str) -> list[BeliefMessage]:
    """Region / entity / belief-id / uncertainty-threshold / time filters; deterministic order."""
    if q.domain is not None and q.domain is not Domain.SPATIAL:
        return []
    if q.region is not None and q.region.frame_id != frame_id:
        raise FrameError(f"query region is in {q.region.frame_id!r}; the map frame is {frame_id!r}")
    out = []
    for m in messages:
        if q.belief_ids and m.belief_id not in q.belief_ids:
            continue
        if q.entity_ids and m.world_entity_id not in q.entity_ids:
            continue
        if q.region is not None and (m.spatial_support is None or not overlaps(q.region, m.spatial_support)):
            continue
        if (
            q.min_observational_uncertainty is not None
            and m.uncertainty.observational < q.min_observational_uncertainty
        ):
            continue
        if (
            q.min_contradiction_uncertainty is not None
            and m.uncertainty.contradiction < q.min_contradiction_uncertainty
        ):
            continue
        if q.time_range_ns is not None and not (
            q.time_range_ns[0] <= m.timestamp.time_ns <= q.time_range_ns[1]
        ):
            continue
        if not q.include_predictions and m.knowledge_status is KnowledgeStatus.PREDICTED:
            continue
        out.append(m)
    out.sort(key=lambda m: str(m.belief_id))
    return out[: q.max_results]
