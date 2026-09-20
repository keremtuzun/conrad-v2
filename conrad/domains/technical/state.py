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

from conrad.domains.technical.config import ConditionConfig, Model2TConfig
from conrad.domains.technical.coverage import OcclusionTest, SurfaceGeometry
from conrad.domains.technical.crack_filter import CrackGrid, moments, population_prior, region_prior
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


WHOLE_COMPONENT = -1
"""Region key of a reading that speaks about the whole component (no design geometry, or no measured point)."""


@dataclass
class Region:
    """Belief about the WORST CASE over one region of the component surface (iteration 4, locus refinement).

    A structural reading reports the worst case over the surface it looked at, not over the component. With
    surveyed design geometry each reading therefore updates the region anchored on the cell its measured point
    fell in, and the component's worst case is the worst region (plus the population prior of what was never
    read). Without geometry there is exactly one region, ``WHOLE_COMPONENT``, and the behaviour is the
    iteration-3 whole-component one."""

    estimates: dict[str, Estimate]
    crack_grid: CrackGrid | None = None
    cells: set[int] = field(default_factory=set)
    """Coverage cells this region's readings credited (empty for ``WHOLE_COMPONENT``)."""
    bias_counts: dict[tuple[str, str], float] = field(default_factory=dict)
    locus: dict[str, tuple[tuple[float, float, float], float, float]] = field(default_factory=dict)


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
    locus: dict[str, tuple[tuple[float, float, float], float, float]] = field(default_factory=dict)
    """Per quantity: (measured surface point, its sigma, reading size) of the worst indication seen so far."""
    crack_grid: CrackGrid | None = None
    """Regime-mixture crack belief (``crack_filter``); None under the CONSTANT_RATE / legacy models."""
    bias_counts: dict[tuple[str, str], float] = field(default_factory=dict)
    """Readings consumed per (sensor bias group, quantity); distinct groups set the persistent-bias floor."""
    regions: dict[int, Region] = field(default_factory=dict)
    """Per-region worst-case beliefs (iteration 4). ``WHOLE_COMPONENT`` aliases ``estimates``/``crack_grid``
    for a component without design geometry, so that path is byte-for-byte the iteration-3 one."""
    geometry: SurfaceGeometry | None = None
    """Surveyed design surface (mission context) for coverage; None = readings are whole-component views."""
    occlusion: OcclusionTest | None = None
    """Deployment-supplied Model2S occlusion test (``coverage.OcclusionTest``); None = no occlusion known."""
    covered: set[int] = field(default_factory=set)
    """Coverage cells of ``geometry`` a reading credited (its measured cell, and under
    ``coverage.view_credit = FOOTPRINT`` the rest of that reading's declared footprint)."""
    reveal: float = 1.0
    """Declared probability that a look at the worst case would have revealed a defect at the DEGRADED band
    (the smallest defect that changes the reported condition), over the quantities this component can carry.
    It is what turns surface coverage into "the worst case has been seen"."""
    coverage_complete: float = 0.8
    coverage_provenance: UUID | None = None
    surface_id: UUID | None = None
    """Belief id of the component's READ-SURFACE part (only with geometry): the surface the readings actually
    showed. While the component-level condition is open, readings are committed to this part (DIRECT) and the
    component itself gets a coverage revision; registry_entity_id of the part is None (not a registry item)."""
    surface_revision: int = -1
    surface_relationship: UUID | None = None
    condition_cfg: ConditionConfig = field(default_factory=ConditionConfig)
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

    def coverage_fraction(self) -> float | None:
        if self.geometry is None:
            return None
        return len(self.covered) / max(self.geometry.n_cells, 1)

    def condition_open(self) -> bool:
        """The component-level condition is not determined by what was read: coverage is partial AND the read
        part alone does not already put the component in the worst band (a worst case can only be worse)."""
        cov = self.coverage_fraction()
        if cov is None or cov >= self.coverage_complete:
            return False
        from conrad.domains.technical.claims import severity_parts  # local: claims imports this module

        parts = severity_parts(self, self.condition_cfg)
        sev = max((v for v, _ in parts.values()), default=None)
        return sev is None or sev < self.condition_cfg.bands[2]

    def seen_fraction(self) -> float | None:
        """Probability that the component's WORST CASE has already been seen: the fraction of the surface that
        was read, times the declared probability that a look reveals a defect at the DEGRADED band. A look
        that could not have called the smallest defect that matters does not make the worst case observed."""
        cov = self.coverage_fraction()
        return None if cov is None else cov * max(0.0, min(1.0, self.reveal))

    def uncertainty(self, surface: bool = False) -> Uncertainty:
        """Component uncertainty; ``surface=True``: of the read-surface part (no coverage term)."""
        uo = max(1.0 - self.direct_support, self.uo_context)
        seen = self.seen_fraction()
        if not surface and seen is not None and self.condition_open():
            uo = max(uo, 1.0 - seen)
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


def reveal_probability(valid: frozenset[str], cfg: Model2TConfig, aleatoric: float) -> float:
    """Declared probability that one look reveals a defect at the DEGRADED band, over the valid quantities.

    Wall loss is always measured, so its reveal probability is 1. A crack at the band (``bands[0]`` x
    ``crack_critical_m``, 2.5 mm with the default bands) is near the declared detection limit
    (``crack_pod_a50_m`` 8 mm), so a look barely constrains it. The condition is the worst over the
    quantities, so the component's reveal probability is the smallest of them."""
    if CRACK_LENGTH not in valid:
        return 1.0
    sc = cfg.sensor
    band_m = cfg.condition.bands[0] * cfg.condition.crack_critical_m
    a50 = sc.crack_pod_a50_m * (1.0 + sc.pod_ua_gain * max(0.0, aleatoric))
    z = (math.log(max(band_m, 1e-9)) - math.log(max(a50, 1e-9))) / max(sc.crack_pod_log_width, 1e-6)
    pod = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
    return max(cfg.coverage.reveal_pod_floor, min(1.0, pod))


def uses_crack_grid(cfg: Model2TConfig) -> bool:
    return (
        cfg.direct.measurement_model == "SENSOR_CHARACTERISED" and cfg.crack_growth.model == "REGIME_MIXTURE"
    )


def sync_crack(est: Estimate, grid: CrackGrid, cfg: Model2TConfig) -> None:
    """Copy the crack grid's posterior moments into the Gaussian summary (status is left untouched)."""
    m = moments(grid, cfg.crack_growth)
    est.level, est.level_var, est.rate, est.rate_var, est.cov = (
        m.level,
        m.level_var,
        m.rate,
        m.rate_var,
        m.cov,
    )


def new_belief(
    belief_id: UUID,
    spec: ComponentSpec,
    cfg: Model2TConfig,
    now_ns: int,
    geometry: SurfaceGeometry | None = None,
) -> ComponentBelief:
    prior = {q: _prior_estimate(q, cfg) for q in QUANTITIES}
    grid = None
    if uses_crack_grid(cfg) and CRACK_LENGTH in spec.valid_quantities():
        grid = population_prior(cfg.crack_growth, cfg.prior)
        sync_crack(prior[CRACK_LENGTH], grid, cfg)
    for est in prior.values():
        est.updated_ns = now_ns
    ue = cfg.prior.base_epistemic + (0.0 if spec.material_known else cfg.prior.unknown_material_epistemic)
    estimates = {q: e.copy() for q, e in prior.items()}
    belief = ComponentBelief(
        belief_id=belief_id,
        spec=spec,
        estimates=estimates,
        prior=prior,
        ue=ue,
        head_time_ns=now_ns,
        crack_grid=None if grid is None else grid.copy(),
        geometry=geometry if cfg.coverage.enabled else None,
        coverage_complete=cfg.coverage.complete_fraction,
        condition_cfg=cfg.condition,
        reveal=reveal_probability(spec.valid_quantities(), cfg, 0.0),
    )
    if belief.geometry is None:
        # Without design geometry there is one region and it ALIASES the component summary, so this path is
        # exactly the iteration-3 one (every 2T experiment runs here).
        belief.regions[WHOLE_COMPONENT] = Region(
            estimates=belief.estimates,
            crack_grid=belief.crack_grid,
            bias_counts=belief.bias_counts,
            locus=belief.locus,
        )
    return belief


def new_region(belief: ComponentBelief, cfg: Model2TConfig) -> Region:
    """A region first read now. Its crack prior is the per-region population prior, so that the worst of the
    component's regions reproduces the component-level population prior instead of inflating it."""
    grid = None
    if uses_crack_grid(cfg) and CRACK_LENGTH in belief.spec.valid_quantities():
        n = belief.geometry.n_cells if belief.geometry is not None else 1
        grid = region_prior(cfg.crack_growth, cfg.prior, n)
    return Region(estimates={q: e.copy() for q, e in belief.prior.items()}, crack_grid=grid)


def summarise(belief: ComponentBelief, cfg: Model2TConfig) -> None:
    """Component worst case = the worst region that has been read (locus refinement, iteration 4).

    Only for a component with design geometry: without it there is one aliased region and nothing to do.
    Regions that were never read keep the population prior, which is UNKNOWN and never reported, so the
    summary is the worst READ region."""
    if belief.geometry is None:
        return
    for q in QUANTITIES:
        best_key, best = None, None
        for key, r in belief.regions.items():
            est = r.estimates.get(q)
            if est is None or not est.direct_lineage:
                continue
            if best is None or est.level > best.level:
                best_key, best = key, est
        if best is None or best_key is None:
            continue
        belief.estimates[q] = best.copy()
        if q == CRACK_LENGTH:
            belief.crack_grid = belief.regions[best_key].crack_grid
        loc = belief.regions[best_key].locus.get(q)
        if loc is not None:
            belief.locus[q] = loc
