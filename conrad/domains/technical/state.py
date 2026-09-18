"""Per-component persistent belief working state (interpretable, analytic).

Each quantity carries a 2-state [level, rate] Gaussian so temporal prediction can learn the rate from
repeated inspections. ``status`` is the property-level knowledge status; ``direct_lineage`` records that
the value descends from direct evidence (so TCDP never overrides it).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from uuid import UUID

from conrad.domains.technical.config import Model2TConfig
from conrad.domains.technical.registry import (
    CORROSION_DEPTH,
    CRACK_LENGTH,
    QUANTITIES,
    ComponentSpec,
)
from conrad.schemas.belief import KnowledgeStatus, Lifecycle
from conrad.schemas.uncertainty import Uncertainty


@dataclass
class Estimate:
    level: float
    level_var: float
    rate: float = 0.0
    rate_var: float = 0.0
    cov: float = 0.0
    status: KnowledgeStatus = KnowledgeStatus.UNKNOWN
    provenance_id: UUID | None = None
    direct_lineage: bool = False
    updated_ns: int = 0

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.level_var, 0.0))

    @property
    def known(self) -> bool:
        return self.status is not KnowledgeStatus.UNKNOWN

    def copy(self) -> Estimate:
        return Estimate(**self.__dict__)


@dataclass
class ComponentBelief:
    belief_id: UUID
    spec: ComponentSpec
    estimates: dict[str, Estimate]
    prior: dict[str, Estimate]
    """Latent population prior, predicted forward in time; never reported (status UNKNOWN)."""
    ua: float = 0.0
    ue: float = 0.2
    uc: float = 0.0
    uo_context: float = 0.0
    ua_context: float = 0.0
    surface_var_gain: float = 1.0
    direct_support: float = 0.0
    propagated_support: float = 0.0
    conflicts: list[UUID] = field(default_factory=list)
    consumed: set[UUID] = field(default_factory=set)
    groups: set[str] = field(default_factory=set)
    last_evidence: tuple[UUID, ...] = ()
    reported_level: dict[str, float] = field(default_factory=dict)
    change_state: str = "STABLE"
    change_status: KnowledgeStatus = KnowledgeStatus.UNKNOWN
    change_provenance: UUID | None = None
    lifecycle: Lifecycle = Lifecycle.CONFIRMED
    revision: int = -1
    provenance_root: UUID | None = None
    head_time_ns: int = 0
    relationship_ids: tuple[UUID, ...] = ()

    @property
    def valid(self) -> frozenset[str]:
        return self.spec.valid_quantities()

    def uncertainty(self) -> Uncertainty:
        uo = max(1.0 - self.direct_support, self.uo_context)
        return Uncertainty(
            aleatoric=min(1.0, self.ua + self.ua_context),
            epistemic=min(1.0, self.ue),
            contradiction=min(1.0, self.uc),
            observational=min(1.0, max(0.0, uo)),
        )


def _prior_estimate(q: str, cfg: Model2TConfig) -> Estimate:
    d = cfg.dynamics
    rate, rate_sd = {
        CORROSION_DEPTH: (d.corrosion_rate_m_per_yr, d.corrosion_rate_sd_m_per_yr),
        CRACK_LENGTH: (d.crack_rate_m_per_yr, d.crack_rate_sd_m_per_yr),
    }.get(q, (0.0, 0.0))
    return Estimate(level=cfg.prior.mean[q], level_var=cfg.prior.sd[q] ** 2, rate=rate, rate_var=rate_sd**2)


def new_belief(belief_id: UUID, spec: ComponentSpec, cfg: Model2TConfig, now_ns: int) -> ComponentBelief:
    prior = {q: _prior_estimate(q, cfg) for q in QUANTITIES}
    for est in prior.values():
        est.updated_ns = now_ns
    ue = cfg.prior.base_epistemic + (0.0 if spec.material_known else cfg.prior.unknown_material_epistemic)
    return ComponentBelief(
        belief_id=belief_id,
        spec=spec,
        estimates={q: e.copy() for q, e in prior.items()},
        prior=prior,
        ue=ue,
        head_time_ns=now_ns,
    )
