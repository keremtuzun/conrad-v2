"""Interpretable claims and TechnicalPayload from a component belief (ch10 Model 2T state heads).

Each claim carries its own status. Derived claims (condition, severity, ...) take the status of the
quantity that dominates them, so a severity driven by an INFERRED corrosion value is INFERRED. With surveyed
design geometry, a partially covered component keeps its component-level condition UNKNOWN (the read part
is ``observed_region_condition``) unless the read part alone already reaches the worst band.
"""

from __future__ import annotations

from uuid import UUID

from conrad.domains.technical.config import ConditionConfig
from conrad.domains.technical.registry import CORROSION_DEPTH, CRACK_LENGTH, QUANTITY_UNITS, SURFACE_ANOMALY
from conrad.domains.technical.state import ComponentBelief, Estimate
from conrad.schemas.belief import KnowledgeStatus, PropertyClaim, TechnicalPayload
from conrad.schemas.uncertainty import Uncertainty


def _claim(
    name: str,
    value: float | str | None,
    units: str | None,
    status: KnowledgeStatus,
    prov: UUID | None,
    u: Uncertainty,
) -> PropertyClaim:
    if status is KnowledgeStatus.UNKNOWN or value is None:
        unknown_u = u.model_copy(update={"observational": 1.0})
        return PropertyClaim(
            name=name, value=None, units=units, status=KnowledgeStatus.UNKNOWN, uncertainty=unknown_u
        )
    return PropertyClaim(
        name=name, value=value, units=units, status=status, uncertainty=u, provenance_id=prov
    )


def wall_m(belief: ComponentBelief, cfg: ConditionConfig) -> float:
    return belief.spec.wall_thickness_m or cfg.nominal_wall_m


def severity_parts(belief: ComponentBelief, cfg: ConditionConfig) -> dict[str, tuple[float, Estimate]]:
    parts: dict[str, tuple[float, Estimate]] = {}
    for q, scale in ((CORROSION_DEPTH, wall_m(belief, cfg)), (CRACK_LENGTH, cfg.crack_critical_m)):
        est = belief.estimates[q]
        if est.known and (q in belief.valid or est.status is KnowledgeStatus.INFERRED):
            parts[q] = (min(1.0, max(0.0, est.level) / scale), est)
    return parts


def condition_of(severity: float, cfg: ConditionConfig) -> str:
    d, s, f = cfg.bands
    return (
        "FAILED"
        if severity >= f
        else "SEVERE"
        if severity >= s
        else "DEGRADED"
        if severity >= d
        else "INTACT"
    )


def _open_component_claims(
    belief: ComponentBelief, cfg: ConditionConfig, u: Uncertainty
) -> tuple[PropertyClaim, ...]:
    """Component view while its condition is open (partial coverage, worst band not reached): the component-level
    worst case is UNKNOWN. What the readings showed lives in the read-surface part; the component carries only
    the coverage and the read part's condition, INFERRED through PART_OF (relational, rule 6)."""
    unknown = KnowledgeStatus.UNKNOWN
    claims: list[PropertyClaim] = []
    for q in (CORROSION_DEPTH, CRACK_LENGTH, SURFACE_ANOMALY):
        if q not in belief.valid and not belief.estimates[q].known:
            continue
        claims.append(_claim(q, None, QUANTITY_UNITS[q], unknown, None, u))
        if q != SURFACE_ANOMALY:
            claims.append(_claim(f"{q}.variance", None, "m^2", unknown, None, u))
    claims += [_claim(n, None, None, unknown, None, u) for n in ("condition", "severity")]
    parts = severity_parts(belief, cfg)
    inferred, prov = KnowledgeStatus.INFERRED, belief.coverage_provenance
    if parts:
        sev = max(v for v, _ in parts.values())
        claims.append(_claim("observed_region_condition", condition_of(sev, cfg), None, inferred, prov, u))
    cov = belief.coverage_fraction()
    claims.append(_claim("surface_coverage", cov, "1", inferred if belief.covered else unknown, prov, u))
    claims += [_claim(n, None, None, unknown, None, u) for n in ("geometry_change", "change_state")]
    return tuple(claims)


def build_claims(
    belief: ComponentBelief, cfg: ConditionConfig, *, surface: bool = False
) -> tuple[PropertyClaim, ...]:
    """Claims of the component (default) or of its read-surface part (``surface=True``)."""
    u = belief.uncertainty(surface=surface)
    if not surface and belief.condition_open():
        return _open_component_claims(belief, cfg, u)
    claims: list[PropertyClaim] = []
    for q in (CORROSION_DEPTH, CRACK_LENGTH, SURFACE_ANOMALY):
        est = belief.estimates[q]
        if q not in belief.valid and not est.known:
            continue
        units = QUANTITY_UNITS[q]
        claims.append(_claim(q, max(0.0, est.level), units, est.status, est.provenance_id, u))
        if q != SURFACE_ANOMALY:
            var_units = "m^2"
            claims.append(_claim(f"{q}.variance", est.level_var, var_units, est.status, est.provenance_id, u))
    parts = severity_parts(belief, cfg)
    if parts:
        dom_q = max(parts, key=lambda k: parts[k][0])
        sev, dom = parts[dom_q][0], parts[dom_q][1]
        st, prov = dom.status, dom.provenance_id
        kinds = [k for k, (v, _) in parts.items() if v >= cfg.bands[0]]
        dtype = (
            "COMBINED"
            if len(kinds) == 2
            else "CORROSION"
            if kinds == [CORROSION_DEPTH]
            else "FATIGUE_CRACK"
            if kinds == [CRACK_LENGTH]
            else "NONE"
        )
        func = "FUNCTIONAL" if sev < cfg.bands[1] else "IMPAIRED" if sev < cfg.bands[2] else "AT_RISK"
        claims += [
            _claim("condition", condition_of(sev, cfg), None, st, prov, u),
            _claim("degradation_type", dtype, None, st, prov, u),
            _claim("severity", sev, "1", st, prov, u),
            _claim("functional_state", func, None, st, prov, u),
        ]
    else:
        claims += [_claim(n, None, None, KnowledgeStatus.UNKNOWN, None, u) for n in ("condition", "severity")]
    corr = belief.estimates[CORROSION_DEPTH]
    cov = belief.coverage_fraction()
    if cov is not None:
        read = KnowledgeStatus.OBSERVED if belief.covered else KnowledgeStatus.UNKNOWN
        claims.append(_claim("surface_coverage", cov, "1", read, belief.coverage_provenance, u))
    geo = max(0.0, corr.level) / wall_m(belief, cfg) if corr.known else None
    claims.append(_claim("geometry_change", geo, "1", corr.status, corr.provenance_id, u))
    claims.append(
        _claim("change_state", belief.change_state, None, belief.change_status, belief.change_provenance, u)
    )
    return tuple(claims)


def technical_payload(
    belief: ComponentBelief, cfg: ConditionConfig, claims: tuple[PropertyClaim, ...], *, surface: bool = False
) -> TechnicalPayload:
    parts = {} if not surface and belief.condition_open() else severity_parts(belief, cfg)
    corr, crack = belief.estimates[CORROSION_DEPTH], belief.estimates[CRACK_LENGTH]
    sev = max((v for v, _ in parts.values()), default=None)
    dtype = next((c.value for c in claims if c.name == "degradation_type"), None)
    return TechnicalPayload(
        condition=None if sev is None else condition_of(sev, cfg),
        degradation_type=None if dtype is None else str(dtype),
        severity=sev,
        corrosion_depth_m=max(0.0, corr.level) if CORROSION_DEPTH in parts else None,
        crack_length_m=max(0.0, crack.level) if CRACK_LENGTH in parts else None,
        direct_support=min(1.0, belief.direct_support),
        propagated_support=min(1.0, belief.propagated_support),
    )


def state_embedding(belief: ComponentBelief) -> tuple[float, ...]:
    """Compact interpretable belief embedding (never a copy of an evidence embedding)."""
    vals: list[float] = []
    for q in (CORROSION_DEPTH, CRACK_LENGTH, SURFACE_ANOMALY):
        e = belief.estimates[q]
        scale = 1e3 if q != SURFACE_ANOMALY else 1.0
        vals += [e.level * scale if e.known else 0.0, e.sd * scale if e.known else 0.0, float(e.known)]
    return (*vals, belief.direct_support, belief.propagated_support)
