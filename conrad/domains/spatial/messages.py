"""Region / object BeliefMessages from base-grid cells (Model2S -> Belief Bus). BELIEF PLANE.

A region message carries one ``occupancy.<status>`` claim per knowledge status present (mean occupancy
over those cells, status-specific uncertainty), an ``occupancy.unknown`` claim (value None) when any cell
of the region is UNKNOWN, and a ``coverage`` claim. Cells never created count as UNKNOWN with zero
coverage. The state embedding is an interpretable summary (no learned latent in V1) and never copies an
evidence embedding.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from uuid import UUID

import numpy as np

from conrad.domains.spatial.grid import INFERRED, OBSERVED, PREDICTED, UNKNOWN, SparseLevel
from conrad.domains.spatial.keys import FloatArr, IntArr
from conrad.schemas.belief import (
    Availability,
    BeliefCell,
    BeliefMessage,
    KnowledgeStatus,
    Lifecycle,
    PropertyClaim,
    SpatialPayload,
    summarize_status,
)
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.uncertainty import Uncertainty, UncertaintyRepresentation, unknown_uncertainty
from conrad.schemas.world import Domain

EMBEDDING_FEATURES = (
    "frac_observed",
    "frac_inferred",
    "frac_predicted",
    "frac_unknown",
    "mean_p_known",
    "coverage",
    "UA",
    "UE",
    "UC",
    "UO",
    "occupied_frac_known",
    "log1p_obs_div10",
)
_STATUS = (
    (OBSERVED, KnowledgeStatus.OBSERVED),
    (INFERRED, KnowledgeStatus.INFERRED),
    (PREDICTED, KnowledgeStatus.PREDICTED),
)


def _unc(ch: FloatArr) -> Uncertainty:
    a, e, c, o = (float(min(max(x, 0.0), 1.0)) for x in ch)
    return Uncertainty(
        aleatoric=a,
        epistemic=e,
        contradiction=c,
        observational=o,
        representation_type=UncertaintyRepresentation.ANALYTIC,
    )


@dataclass(frozen=True)
class RegionSummary:
    center_m: tuple[float, float, float]
    half_extent_m: tuple[float, float, float]
    n_cells: int
    status: np.ndarray
    probability: FloatArr
    uncertainty: FloatArr
    coverage_mean: float
    pose_sigma_m: float | None
    observation_count: int
    semantic_class: str | None

    @property
    def unknown_count(self) -> int:
        return self.n_cells - int((self.status != UNKNOWN).sum())

    @property
    def uncertainty_total(self) -> Uncertainty:
        """UA/UE/UC over known cells; UO over the whole region volume (absent cells have UO = 1)."""
        known = self.status != UNKNOWN
        ch = self.uncertainty[known].mean(axis=0) if known.any() else np.array([0.0, 1.0, 0.0, 1.0])
        uo_sum = float(self.uncertainty[:, 3].sum()) + (self.n_cells - len(self.status))
        ch = np.array([ch[0], ch[1], ch[2], uo_sum / max(self.n_cells, 1)])
        return _unc(ch)

    def embedding(self) -> tuple[float, ...]:
        n = max(self.n_cells, 1)
        known = self.status != UNKNOWN
        frac = [float((self.status == s).sum()) / n for s, _ in _STATUS]
        u = self.uncertainty_total.as_tuple()
        p_known = float(self.probability[known].mean()) if known.any() else 0.5
        occ = float((self.probability[known] >= 0.5).mean()) if known.any() else 0.0
        vals = [
            *frac,
            self.unknown_count / n,
            p_known,
            self.coverage_mean,
            *u,
            occ,
            math.log1p(self.observation_count) / 10,
        ]
        return tuple(float(v) for v in vals)


def summarize(
    level: SparseLevel, rows: IntArr, n_cells: int, center: FloatArr, half: FloatArr
) -> RegionSummary:
    rows = rows[rows >= 0]
    st = level.status(rows)
    sem = [level.semantic[r] for r in rows.tolist() if r in level.semantic]
    sig = level.mean_pose_sigma(rows)
    has = level.f["w"][rows] > 0
    return RegionSummary(
        center_m=(float(center[0]), float(center[1]), float(center[2])),
        half_extent_m=(float(half[0]), float(half[1]), float(half[2])),
        n_cells=n_cells,
        status=st,
        probability=level.probability(rows),
        uncertainty=level.uncertainty(rows),
        coverage_mean=float(level.coverage(rows).sum()) / max(n_cells, 1),
        pose_sigma_m=float(sig[has].mean()) if has.any() else None,
        observation_count=int(level.n_obs[rows].max(initial=0)),
        semantic_class=max(set(sem), key=sem.count) if sem else None,
    )


def region_claims(
    s: RegionSummary, provenance_id: UUID, predicted_only: bool = False
) -> tuple[PropertyClaim, ...]:
    claims: list[PropertyClaim] = []
    for code, ks in _STATUS:
        sel = s.status == code
        if predicted_only and code != PREDICTED:
            continue
        if sel.any():
            claims.append(
                PropertyClaim(
                    name=f"occupancy.{ks.value.lower()}",
                    value=float(s.probability[sel].mean()),
                    units="probability",
                    status=ks,
                    uncertainty=_unc(s.uncertainty[sel].mean(axis=0)),
                    provenance_id=provenance_id,
                )
            )
    if not predicted_only and s.unknown_count > 0:
        claims.append(
            PropertyClaim(
                name="occupancy.unknown",
                value=None,
                status=KnowledgeStatus.UNKNOWN,
                uncertainty=unknown_uncertainty(),
            )
        )
    direct = (s.status == OBSERVED).any() or s.coverage_mean > 0
    if not predicted_only:
        claims.append(
            PropertyClaim(
                name="coverage",
                value=s.coverage_mean if direct else None,
                units="fraction",
                status=KnowledgeStatus.OBSERVED if direct else KnowledgeStatus.UNKNOWN,
                uncertainty=s.uncertainty_total,
                provenance_id=provenance_id if direct else None,
            )
        )
    return tuple(claims)


@dataclass(frozen=True)
class MessageSpec:
    belief_id: UUID
    revision: int
    registry_id: UUID | None
    provenance_id: UUID
    evidence: tuple[UUID, ...]
    change_summary: str | None
    prediction_summary: str | None
    availability: Availability


def build(
    s: RegionSummary,
    spec: MessageSpec,
    message_id: UUID,
    now: TimeStamp,
    model_version: str,
    resolution_m: float,
    entity_type: str,
    frame_id: str,
    predicted_only: bool = False,
) -> tuple[BeliefMessage, BeliefCell]:
    claims = region_claims(s, spec.provenance_id, predicted_only)
    status = summarize_status(claims)
    evidence = () if status is KnowledgeStatus.PREDICTED else spec.evidence
    support = SpatialSupport(
        frame_id=frame_id, center_m=s.center_m, half_extent_m=s.half_extent_m, position_sigma_m=s.pose_sigma_m
    )
    known = s.status != UNKNOWN
    unc = s.uncertainty_total
    msg = BeliefMessage(
        message_id=message_id,
        belief_id=spec.belief_id,
        revision=spec.revision,
        world_entity_id=spec.registry_id,
        domain=Domain.SPATIAL,
        timestamp=now,
        state_summary=claims,
        state_embedding=s.embedding(),
        knowledge_status=status,
        uncertainty=unc,
        evidence_support=evidence,
        change_summary=spec.change_summary,
        prediction_summary=spec.prediction_summary,
        provenance_refs=(spec.provenance_id,),
        spatial_support=support,
        lifecycle=Lifecycle.ACTIVE,
        model_version=model_version,
        publisher_availability=spec.availability,
        spatial=SpatialPayload(
            occupancy_probability=float(s.probability[known].mean()) if known.any() else None,
            coverage=min(max(s.coverage_mean, 0.0), 1.0),
            resolution_m=resolution_m,
            semantic_class=s.semantic_class,
            observation_count=s.observation_count,
        ),
    )
    cell = BeliefCell(
        belief_id=spec.belief_id,
        domain=Domain.SPATIAL,
        entity_type=entity_type,
        registry_entity_id=spec.registry_id,
        lifecycle=Lifecycle.ACTIVE,
        revision=spec.revision,
        timestamp=now,
        state_embedding=msg.state_embedding,
        claims=claims,
        knowledge_status=status,
        uncertainty=unc,
        spatial_support=support,
        provenance_root=spec.provenance_id,
        model_version=model_version,
    )
    return msg, cell
