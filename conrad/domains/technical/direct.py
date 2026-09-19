"""Direct evidence update: per-quantity Kalman update weighted by reliability and measurement noise.

* Duplicate evidence IDs are ignored.
* Evidence sharing an independence group inside one batch is fused as ONE measurement (mean value,
  mean variance: no 1/n shrinkage). A group already consumed earlier may move the mean but can never
  shrink the variance (no artificial certainty).
* A surprising reading (normalised innovation > conflict_sigma) inflates the prior variance by
  innov^2 - S, so a genuine change (run-away crack, repair) is followed instead of smoothed away.
* If that reading is also reliable, a conflict is recorded, U_C rises, and the variance is floored at the
  equal-weight two-hypothesis mixture: the disagreement is carried forward, not averaged into certainty.
* Wall loss and crack length (default ``SENSOR_CHARACTERISED``) use the declared sensor characteristics
  (``measurement.py``): relative error, a persistent per-sensor bias (variance floor per distinct sensor),
  censored non-detections, partial views as lower bounds, and "surprise" defined as the belief putting
  almost no mass on the reading's plausible set. A crack becomes OBSERVED only from a detection; readings
  that never detected it update the latent estimate but leave the claim UNKNOWN.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

from conrad.domains.technical import crack_filter
from conrad.domains.technical.config import Model2TConfig
from conrad.domains.technical.dynamics import predict_estimate
from conrad.domains.technical.measurement import (
    MODELLED,
    Reading,
    grid_update,
    scatter_sigmas,
)
from conrad.domains.technical.registry import CORROSION_DEPTH, CRACK_LENGTH, QUANTITY_UNITS, SURFACE_ANOMALY
from conrad.domains.technical.state import ComponentBelief, Estimate, sync_crack
from conrad.schemas.belief import KnowledgeStatus, Lifecycle
from conrad.schemas.frames import WORLD
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


UNATTRIBUTED = "unattributed-sensor"
"""Bias group of evidence that names no sensor: conservatively treated as one shared sensor."""


def bias_group(ev: Evidence) -> str:
    """Persistent-bias group: the ``sensor:<id>`` prefix of a ``sensor:<id>|<reading group>`` group."""
    g = ev.independence_group or ""
    return g.split("|", 1)[0] if "|" in g else UNATTRIBUTED


def _elsewhere(belief: ComponentBelief, q: str, ev: Evidence, locality_m: float) -> bool:
    known = belief.locus.get(q)
    sup = ev.spatial_support
    if known is None or sup is None:
        return False
    (x, y, zc), sigma, _ = known
    d = math.dist(sup.center_m, (x, y, zc))
    return d > locality_m + 2.0 * math.hypot(sigma, sup.position_sigma_m or 0.0)


def _characterised(
    belief: ComponentBelief,
    est: Estimate,
    q: str,
    members: Sequence[Evidence],
    z: float,
    cfg: Model2TConfig,
) -> tuple[bool, float, float, bool]:
    """Sensor-characterised update of est in place. Returns (surprise, innovation, reading var, detected).

    Each reading's likelihood carries only its independent scatter. The persistent per-sensor bias cannot
    be averaged away, so afterwards the level variance is floored at (bias * level)^2 / (number of distinct
    sensors that have read this quantity): repeated same-sensor looks never buy certainty below the bias,
    while a changing state is still followed at full weight (the bias cancels in differences)."""
    sc = cfg.sensor
    rel = sum(m.reliability for m in members) / len(members)
    ua = sum(m.aleatoric_uncertainty for m in members) / len(members)
    key = (bias_group(members[0]), q)
    ind, sys_ = scatter_sigmas(q, z, ua, sc)
    reading = Reading(q, z, rel, ua, elsewhere=_elsewhere(belief, q, members[0], sc.defect_locality_m))
    if q == CRACK_LENGTH and belief.crack_grid is not None:
        return _crack_grid_update(belief, belief.crack_grid, est, reading, key, members[0], (ind, sys_), cfg)
    mean, var = est.level, est.level_var
    first = not est.direct_lineage
    tail = (cfg.prior.tail_weight.get(q, 0.0), cfg.prior.tail_scale_m.get(q, 1.0)) if first else None
    post = grid_update(mean, var, reading, sc, inflate_on_surprise=not first, prior_tail=tail)
    innov = post.mode - mean
    if post.surprise and not first:
        est.level_var += innov * innov  # mirror the grid's inflation so the rate can follow a real change
    prior_var = est.level_var
    if post.var < prior_var * (1.0 - 1e-9):
        r_eq = 1.0 / (1.0 / post.var - 1.0 / prior_var)
        y = mean + (post.mean - mean) * (prior_var + r_eq) / prior_var
        _kalman(est, y, r_eq)
        est.level_var = post.var
    else:
        est.level, est.level_var = post.mean, post.var
    belief.bias_counts[key] = belief.bias_counts.get(key, 0.0) + 1.0
    sup = members[0].spatial_support
    if sup is not None and reading.detected(sc) and z >= belief.locus.get(q, ((0.0, 0.0, 0.0), 0.0, -1.0))[2]:
        belief.locus[q] = (sup.center_m, sup.position_sigma_m or 0.0, z)
    if reading.detected(sc):
        sensors = sum(1 for g, qq in belief.bias_counts if qq == q)
        est.level_var = max(est.level_var, (sys_ * max(est.level, 0.0)) ** 2 / sensors)
    r_read = (math.hypot(ind, sys_) * max(abs(z), abs(mean))) ** 2
    return post.surprise, innov, r_read, reading.detected(sc)


def _crack_grid_update(
    belief: ComponentBelief,
    grid: crack_filter.CrackGrid,
    est: Estimate,
    reading: Reading,
    key: tuple[str, str],
    first: Evidence,
    sigmas: tuple[float, float],
    cfg: Model2TConfig,
) -> tuple[bool, float, float, bool]:
    """Regime-mixture crack update (the grid already holds the prediction to the reading time)."""
    sc = cfg.sensor
    mean = est.level
    res = crack_filter.update(grid, reading, sc, cfg.crack_growth)
    belief.bias_counts[key] = belief.bias_counts.get(key, 0.0) + 1.0
    detected = reading.detected(sc)
    sup = first.spatial_support
    q = reading.quantity
    if sup is not None and detected and reading.value >= belief.locus.get(q, ((0.0, 0.0, 0.0), 0.0, -1.0))[2]:
        belief.locus[q] = (sup.center_m, sup.position_sigma_m or 0.0, reading.value)
    if detected:
        sensors = sum(1 for _, qq in belief.bias_counts if qq == q)
        crack_filter.floor_log_sd(grid, sigmas[1] / math.sqrt(sensors), cfg.crack_growth)
    sync_crack(est, grid, cfg)
    r_read = (math.hypot(*sigmas) * max(abs(reading.value), abs(mean))) ** 2
    return res.surprise, res.mode - mean, r_read, detected


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
    characterised = cfg.direct.measurement_model == "SENSOR_CHARACTERISED"
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
            grid = belief.crack_grid if characterised and q == CRACK_LENGTH else None
            if grid is not None:
                # the grid IS the working crack state (population prior until the first reading)
                crack_filter.propagate(
                    grid, max(t_ns - est.updated_ns, 0) / 1e9, cfg.crack_growth, force=True
                )
                sync_crack(est, grid, cfg)
            elif not est.direct_lineage:
                est = belief.prior[q].copy()
            if grid is None and t_ns > est.updated_ns:
                predict_estimate(est, q, (t_ns - est.updated_ns) / 1e9, cfg)
            before = (est.level_var, est.rate_var)
            had_direct = est.direct_lineage and est.known
            reliable = rel >= cfg.direct.conflict_min_reliability
            detected = True
            if characterised and q in MODELLED:
                surprise_flag, innov, r, detected = _characterised(belief, est, q, members, z, cfg)
                nis = innov / (before[0] + r) ** 0.5
                surprise = cfg.direct.conflict_sigma + 1.0 if surprise_flag else 0.0
            else:
                innov = z - est.level
                surprise = abs(innov) / (est.level_var + r) ** 0.5
                if had_direct and surprise > cfg.direct.conflict_sigma:
                    # Adaptive (innovation-matched) process noise: a genuine change is not smoothed away.
                    est.level_var += innov * innov - (est.level_var + r)
                _kalman(est, z, r)
                nis = innov / (before[0] + r) ** 0.5
            if had_direct and reliable and surprise > cfg.direct.conflict_sigma:
                # Reliable contradiction: keep both hypotheses' spread and raise U_C (not averaged away).
                # (The crack grid carries both hypotheses itself through its surprise mixing.)
                if grid is None:
                    est.level_var = max(est.level_var, 0.5 * before[0] + 0.5 * r + 0.25 * innov * innov)
                group_conflict = True
            elif had_direct and reliable:
                reliable_consistent = True
            if seen_before and grid is None:
                est.level_var = max(est.level_var, before[0])
                est.rate_var = max(est.rate_var, before[1])
            if q in (CORROSION_DEPTH, CRACK_LENGTH) and had_direct:
                big = abs(nis) > cfg.condition.change_sigma
                belief.change_state = ("PROGRESSING" if nis > 0 else "IMPROVED") if big else "STABLE"
                belief.change_status, belief.change_provenance = KnowledgeStatus.OBSERVED, provenance_id
            if detected or est.known:
                est.status, est.provenance_id = KnowledgeStatus.OBSERVED, provenance_id
            # else: a crack never detected stays UNKNOWN; the censored evidence lives in the latent estimate
            est.direct_lineage, est.updated_ns = True, max(t_ns, est.updated_ns)
            belief.estimates[q] = est
            out.changed = True
        if group_conflict:
            belief.uc = min(1.0, belief.uc + cfg.direct.uc_gain)
            out.conflicts.extend(m.evidence_id for m in members)
        if belief.geometry is not None:
            for m in members:
                sup = m.spatial_support
                if sup is not None and sup.frame_id == WORLD:
                    belief.covered.add(belief.geometry.cell_of(sup.center_m))
                    belief.coverage_provenance = provenance_id
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
