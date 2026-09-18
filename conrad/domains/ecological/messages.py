"""Claim / cell / message builders for 2E. ENTITY and FIELD beliefs stay separate state types.

Status rules: a directly measured property is OBSERVED; a field value in a region with no sensor
coverage is INFERRED (interpolated); stress is INFERRED; predictions are PREDICTED; anything never
measured is UNKNOWN with no value. ``ecological_damage`` has no analytic decoder from E0/E1
evidence, so it is always UNKNOWN here: stress can never become an observed damage claim.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from uuid import UUID

import numpy as np

from conrad.domains.ecological.cefd_analytic import DAMAGE_CLAIM, STRESS_CLAIM
from conrad.domains.ecological.entity_belief import MOBILE, EntityBelief
from conrad.domains.ecological.field_belief import FieldBeliefGrid
from conrad.schemas.belief import KnowledgeStatus, PropertyClaim
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.uncertainty import Uncertainty

K = KnowledgeStatus


def unc(ua: float, ue: float, uc: float, uo: float) -> Uncertainty:
    return Uncertainty(
        aleatoric=max(0.0, ua), epistemic=max(0.0, ue), contradiction=max(0.0, uc), observational=max(0.0, uo)
    )


def claim(
    name: str,
    value: float | bool | None,
    units: str | None,
    status: KnowledgeStatus,
    u: Uncertainty,
    prov: UUID | None,
) -> PropertyClaim:
    if status is K.UNKNOWN:
        return PropertyClaim(name=name, value=None, units=units, status=status, uncertainty=u)
    return PropertyClaim(
        name=name, value=value, units=units, status=status, uncertainty=u, provenance_id=prov
    )


# ---------------------------------------------------------------------- entity
def entity_uncertainty(b: EntityBelief, cover_var: float, prior_var: float) -> Uncertainty:
    ua = min(1.0, b.meas_var_ema / prior_var) if b.n_hits else 0.0
    ue = 0.5 if b.asset is None else float(b.ood_ema)
    excess = max(0.0, b.nis_ema - 1.0)
    uc = excess / (1.0 + excess) if b.n_hits else 0.0
    uo = 1.0 if b.n_hits == 0 else float(min(1.0, cover_var / prior_var))
    return unc(ua, ue, uc, uo)


def entity_claims(
    b: EntityBelief,
    cover: tuple[float, float],
    presence: float,
    measured_status: KnowledgeStatus,
    direct_prov: UUID | None,
    stress_prov: UUID | None,
    prior_var: float,
    stress_status: KnowledgeStatus = K.INFERRED,
) -> tuple[PropertyClaim, ...]:
    u = entity_uncertainty(b, cover[1], prior_var)
    seen = b.n_hits > 0
    out: list[PropertyClaim] = []
    if b.eco_class == MOBILE:
        st = measured_status if seen else K.UNKNOWN
        out.append(claim("presence_probability", presence, "1", st, u, direct_prov))
    else:
        st = measured_status if seen else K.UNKNOWN
        out.append(claim("cover_fraction", cover[0], "1", st, u, direct_prov))
        out.append(claim("cover_fraction.sd", math.sqrt(cover[1]), "1", st, u, direct_prov))
        if b.stress_p is not None and stress_prov is not None:
            su = unc(0.0, 0.2 if b.asset and b.asset.stress_threshold_c else 0.5, 0.0, u.observational)
            out.append(claim(STRESS_CLAIM, b.stress_p, "1", stress_status, su, stress_prov))
        # No decoder for condition/damage from E0/E1 evidence: UNKNOWN, whatever the stress says.
        out.append(claim(DAMAGE_CLAIM, None, "1", K.UNKNOWN, unc(0.0, 1.0, 0.0, 1.0), None))
    return tuple(out)


def entity_quantities(
    b: EntityBelief, cover: tuple[float, float], presence: float
) -> tuple[dict[str, float], dict[str, str]]:
    q = (
        {"presence_probability": presence}
        if b.eco_class == MOBILE
        else {"cover_fraction": cover[0], "cover_fraction_sd": math.sqrt(cover[1])}
    )
    if b.stress_p is not None:
        q[STRESS_CLAIM] = b.stress_p
    q["survey_hits"] = float(b.n_hits)
    return q, dict.fromkeys(q, "1")


def entity_embedding(b: EntityBelief, cover: tuple[float, float], presence: float) -> tuple[float, ...]:
    """Belief-derived summary vector (never an evidence embedding)."""
    return (cover[0], math.sqrt(cover[1]), presence, b.stress_p or 0.0, math.log1p(b.n_hits))


def entity_support(b: EntityBelief) -> SpatialSupport:
    r = b.asset.radius_m if b.asset is not None else 1.0
    return SpatialSupport(
        frame_id=b.frame_id,
        center_m=(float(b.position_m[0]), float(b.position_m[1]), float(b.position_m[2])),
        half_extent_m=(r, r, r),
    )


# ---------------------------------------------------------------------- field
def field_summary(
    fields: FieldBeliefGrid, name: str, mean: np.ndarray, var: np.ndarray, cells: np.ndarray | None
) -> tuple[dict[str, float], dict[str, str]]:
    fb = fields.fields[name]
    units = fb.spec.units
    m = mean if cells is None else mean[:, cells]
    v = var if cells is None else var[:, cells]
    q: dict[str, float] = {}
    u: dict[str, str] = {}
    for c in range(m.shape[0]):
        sfx = "" if m.shape[0] == 1 else f".{c}"
        q[f"mean{sfx}"], u[f"mean{sfx}"] = float(np.mean(m[c])), units
        q[f"min{sfx}"], u[f"min{sfx}"] = float(np.min(m[c])), units
        q[f"max{sfx}"], u[f"max{sfx}"] = float(np.max(m[c])), units
        q[f"sd{sfx}"], u[f"sd{sfx}"] = float(np.sqrt(np.mean(v[c]))), units
    q["coverage"], u["coverage"] = fields.coverage(name, cells), "1"
    q["cells"], u["cells"] = float(m.shape[1]), "1"
    q["observations"], u["observations"] = float(fb.n_obs), "1"
    return q, u


def field_claims(
    fields: FieldBeliefGrid,
    name: str,
    q: dict[str, float],
    status: KnowledgeStatus,
    u: Uncertainty,
    prov: UUID | None,
) -> tuple[PropertyClaim, ...]:
    fb = fields.fields[name]
    if fb.n_obs == 0:
        status = K.UNKNOWN
    elif status is K.OBSERVED and q["coverage"] == 0.0:
        status = K.INFERRED  # interpolated from sensors outside the region
    units = fb.spec.units
    out = [
        claim(f"{name}.{k}", v, units, status, u, prov) for k, v in q.items() if k.startswith(("mean", "sd"))
    ]
    out.append(claim(f"{name}.coverage", q["coverage"], "1", status, u, prov))
    return tuple(out)


def field_support(fields: FieldBeliefGrid, cells: np.ndarray | None = None) -> SpatialSupport:
    g = fields.grid
    c = g.centers if cells is None or not cells.any() else g.centers[cells]
    lo, hi = c.min(axis=0) - 0.5 * g.spacing, c.max(axis=0) + 0.5 * g.spacing
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return SpatialSupport(
        frame_id=g.cfg.frame_id,
        center_m=(float(mid[0]), float(mid[1]), float(mid[2])),
        half_extent_m=(float(half[0]), float(half[1]), float(half[2])),
    )


def field_embedding(q: dict[str, float]) -> tuple[float, ...]:
    return tuple(float(q[k]) for k in sorted(q))
