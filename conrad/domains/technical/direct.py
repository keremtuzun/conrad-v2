"""Direct evidence update: per-quantity Kalman update weighted by reliability and measurement noise.

* Duplicate evidence IDs are ignored.
* Evidence sharing an independence group inside one batch is fused as ONE measurement (mean value,
  mean variance: no 1/n shrinkage). A group already consumed earlier may move the mean but can never
  shrink the variance (no artificial certainty).
* A surprising reading (normalised innovation > conflict_sigma) inflates the prior variance by
  innov^2 - S, so a genuine change (run-away crack, repair) is followed instead of smoothed away.
* If that reading is also reliable, a conflict is recorded, U_C rises, and the variance is floored at the
  equal-weight two-hypothesis mixture: the disagreement is carried forward, not averaged into certainty.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

from conrad.domains.technical.config import Model2TConfig
from conrad.domains.technical.dynamics import predict_estimate
from conrad.domains.technical.registry import CORROSION_DEPTH, CRACK_LENGTH, QUANTITY_UNITS, SURFACE_ANOMALY
from conrad.domains.technical.state import ComponentBelief, Estimate
from conrad.schemas.belief import KnowledgeStatus, Lifecycle
from conrad.schemas.observation import Evidence, EvidenceValidity

MEASUREMENT_ALIASES: dict[str, str] = {
    "apparent_wall_loss": CORROSION_DEPTH,
    "corrosion_depth_m": CORROSION_DEPTH,
    "crack_indication_length": CRACK_LENGTH,
    "crack_length_m": CRACK_LENGTH,
    "surface_anomaly_score": SURFACE_ANOMALY,
    "surface_anomaly": SURFACE_ANOMALY,
}


@dataclass
class DirectOutcome:
    accepted: list[UUID] = field(default_factory=list)
    rejected: list[UUID] = field(default_factory=list)
    conflicts: list[UUID] = field(default_factory=list)
    changed: bool = False
    max_time_ns: int = 0


def measurements_of(ev: Evidence) -> dict[str, float]:
    """Map evidence measurements onto belief quantities; unit mismatches are dropped, never coerced."""
    out: dict[str, float] = {}
    for name, value in ev.measurements.items():
        q = MEASUREMENT_ALIASES.get(name)
        if q is None or ev.measurement_units.get(name) != QUANTITY_UNITS[q] or not math.isfinite(value):
            continue
        out[q] = float(value)
    return out


def measurement_variance(q: str, ev: Evidence, belief: ComponentBelief, cfg: Model2TConfig) -> float:
    sigma = cfg.direct.sigma_m[q] * (1.0 + ev.aleatoric_uncertainty)
    var = sigma * sigma / max(ev.reliability, cfg.direct.min_reliability)
    return var * (belief.surface_var_gain if q == SURFACE_ANOMALY else 1.0)


def _kalman(est: Estimate, z: float, r: float) -> tuple[float, float]:
    """Returns (normalised innovation, level gain); updates est in place."""
    s = est.level_var + r
    innov = z - est.level
    k0, k1 = est.level_var / s, est.cov / s
    p00, p01, p11 = est.level_var, est.cov, est.rate_var
    est.level += k0 * innov
    est.rate += k1 * innov
    est.level_var = (1.0 - k0) * p00
    est.cov = (1.0 - k0) * p01
    est.rate_var = max(p11 - k1 * p01, 0.0)
    return innov / math.sqrt(s), k0


def _groups(evidence: Sequence[Evidence]) -> list[tuple[str, list[Evidence]]]:
    grouped: dict[str, list[Evidence]] = {}
    for ev in sorted(evidence, key=lambda e: (e.timestamp.time_ns, str(e.evidence_id))):
        grouped.setdefault(ev.independence_group or str(ev.evidence_id), []).append(ev)
    return list(grouped.items())


def apply_direct(
    belief: ComponentBelief, evidence: Sequence[Evidence], provenance_id: UUID, cfg: Model2TConfig
) -> DirectOutcome:
    out = DirectOutcome()
    fresh: list[Evidence] = []
    for ev in evidence:
        if ev.evidence_id in belief.consumed or any(ev.evidence_id == f.evidence_id for f in fresh):
            continue
        if ev.validity is EvidenceValidity.INVALID or ev.reliability < cfg.direct.min_reliability:
            out.rejected.append(ev.evidence_id)
            continue
        fresh.append(ev)
    reliable_consistent = False
    for group, members in _groups(fresh):
        seen_before = group in belief.groups
        t_ns = max(m.timestamp.time_ns for m in members)
        rel = sum(m.reliability for m in members) / len(members)
        per_q: dict[str, list[tuple[float, float]]] = {}
        for m in members:
            for q, z in measurements_of(m).items():
                per_q.setdefault(q, []).append((z, measurement_variance(q, m, belief, cfg)))
        group_conflict = False
        for q, vals in per_q.items():
            if q not in belief.valid:
                continue
            z = sum(v for v, _ in vals) / len(vals)
            r = sum(v for _, v in vals) / len(vals)
            est = belief.estimates[q]
            if not est.direct_lineage:
                est = belief.prior[q].copy()
            if t_ns > est.updated_ns:
                predict_estimate(est, q, (t_ns - est.updated_ns) / 1e9, cfg)
            before = (est.level_var, est.rate_var)
            had_direct = est.direct_lineage and est.known
            innov = z - est.level
            surprise = abs(innov) / (est.level_var + r) ** 0.5
            reliable = rel >= cfg.direct.conflict_min_reliability
            if had_direct and surprise > cfg.direct.conflict_sigma:
                # Adaptive (innovation-matched) process noise: a genuine change is not smoothed away.
                est.level_var += innov * innov - (est.level_var + r)
            nis, _ = _kalman(est, z, r)
            if had_direct and reliable and surprise > cfg.direct.conflict_sigma:
                # Reliable contradiction: keep both hypotheses' spread and raise U_C (not averaged away).
                est.level_var = max(est.level_var, 0.5 * before[0] + 0.5 * r + 0.25 * innov * innov)
                group_conflict = True
            elif had_direct and reliable:
                reliable_consistent = True
            nis = innov / (before[0] + r) ** 0.5
            if seen_before:
                est.level_var = max(est.level_var, before[0])
                est.rate_var = max(est.rate_var, before[1])
            if q in (CORROSION_DEPTH, CRACK_LENGTH) and had_direct:
                big = abs(nis) > cfg.condition.change_sigma
                belief.change_state = ("PROGRESSING" if nis > 0 else "IMPROVED") if big else "STABLE"
                belief.change_status, belief.change_provenance = KnowledgeStatus.OBSERVED, provenance_id
            est.status, est.provenance_id = KnowledgeStatus.OBSERVED, provenance_id
            est.direct_lineage, est.updated_ns = True, max(t_ns, est.updated_ns)
            belief.estimates[q] = est
            out.changed = True
        if group_conflict:
            belief.uc = min(1.0, belief.uc + cfg.direct.uc_gain)
            out.conflicts.extend(m.evidence_id for m in members)
        if not seen_before:
            belief.direct_support = 1.0 - (1.0 - belief.direct_support) * (1.0 - rel)
        belief.groups.add(group)
        out.accepted.extend(m.evidence_id for m in members)
        out.max_time_ns = max(out.max_time_ns, t_ns)
    if out.accepted:
        if reliable_consistent and not out.conflicts:
            belief.uc *= cfg.direct.uc_resolve_factor
        a = cfg.direct.ua_smoothing
        ua_new = sum(e.aleatoric_uncertainty for e in fresh) / len(fresh)
        belief.ua = (1.0 - a) * belief.ua + a * min(1.0, ua_new)
        base_ue = cfg.prior.base_epistemic + (
            0.0 if belief.spec.material_known else cfg.prior.unknown_material_epistemic
        )
        ood = max((e.sensor_context.ood_score or 0.0) for e in fresh)
        belief.ue = min(1.0, max(base_ue, ood))
        belief.consumed.update(out.accepted)
        belief.conflicts.extend(out.conflicts)
        belief.last_evidence = tuple(out.accepted)
        if belief.lifecycle is Lifecycle.CONFIRMED:
            belief.lifecycle = Lifecycle.ACTIVE
        out.changed = True
    return out
