"""Analytic TCDP (Topology-Constrained Degradation Propagation). EXPERIMENTAL_CANDIDATE.

Constraints (ch10 TCDP, ch33 RBP rules):
* a message travels only along relation types valid for the quantity's mechanism
  (corrosion: shared environment CONNECTED_TO / ATTACHED_TO / CONTACTS / EXPOSED_TO; fatigue: load path);
* the source must carry direct lineage for that quantity, and the message is gated by the source's
  direct support (reliability gate) and attenuated by an assumed edge correlation rho
  (Gaussian conditional: mean moves by rho * scaled deviation, variance keeps (1 - rho^2) of the prior);
* only targets WITHOUT direct lineage are written: an observed neighbour, healthy or not, is never
  overridden; results are INFERRED and raise propagated_support, never direct_support or U_O.

``GENERIC`` mode is the mechanism-agnostic baseline: every edge, every quantity, no reliability gate,
inferred sources re-broadcast (multi-hop). It exists to measure relational contamination.

Two message models are available (``TCDPConfig.message_model``):

* ``GAUSSIAN_CONDITIONAL`` - the iteration-3 arm described above, with an ASSUMED per-relation correlation.
* ``MEASURED_EXPOSURE`` - shared-exposure inference. A neighbour's posterior is evidence that the receiver
  shares a defect-driving exposure, not a linear predictor of its level. The edge correlation of the
  "carries a defect" indicator is MEASURED on the pairs whose two ends both carry direct evidence and is
  shrunk toward the assumed correlation (empirical Bayes), so an asset whose observed neighbours show no
  correlation sends no information. The message is the resulting two-component prior (background core vs
  shared exposure); when it carries no information it reproduces the population prior exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from conrad.domains.technical.config import MessageModel, PropagationMode, TCDPConfig
from conrad.domains.technical.registry import (
    PROPAGATED_QUANTITIES,
    QUANTITY_MECHANISM,
    AssetRegistry,
)
from conrad.domains.technical.state import ComponentBelief, Estimate
from conrad.schemas.belief import KnowledgeStatus

EdgeKey = tuple[str, str, str]


def edge_key(a: UUID, b: UUID, relation_type: str) -> EdgeKey:
    x, y = sorted((str(a), str(b)))
    return (x, y, relation_type)


@dataclass(frozen=True)
class MessageSource:
    belief_id: UUID
    relationship_id: UUID | None
    relation_type: str
    parent_provenance: UUID | None
    gate: float


@dataclass(frozen=True)
class Inference:
    target: UUID
    quantity: str
    mean: float
    var: float
    rate: float
    rate_var: float
    support: float
    sources: tuple[MessageSource, ...]


def _message(
    target_prior: Estimate, src_prior: Estimate, src: Estimate, rho: float, max_shift_sd: float
) -> tuple[float, float]:
    """Gaussian conditional of the target given the source estimate under correlation rho.

    The shift is bounded to ``max_shift_sd`` target-prior sd: a neighbour is evidence about shared
    exposure, not about a local failure (an unbounded conditional turns one failed crack into many)."""
    st, ss = math.sqrt(target_prior.level_var), math.sqrt(max(src_prior.level_var, 1e-30))
    beta = rho * st / ss
    shift = beta * (src.level - src_prior.level)
    if max_shift_sd > 0:
        shift = max(-max_shift_sd * st, min(max_shift_sd * st, shift))
    mean = target_prior.level + shift
    var = target_prior.level_var * (1.0 - rho * rho) + beta * beta * src.level_var
    return mean, max(var, 1e-30)


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _logit(p: float) -> float:
    q = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(q / (1.0 - q))


def _elevated_p(est: Estimate, core: Estimate, k: float) -> float:
    """P(the component carries a defect) = P(level > core mean + k core sd) under its posterior."""
    thr = core.level + k * math.sqrt(max(core.level_var, 0.0))
    return _normal_sf((thr - est.level) / math.sqrt(max(est.level_var, 1e-30)))


@dataclass(frozen=True)
class ExposureStats:
    """Empirical-Bayes description of the asset's own shared-exposure structure, for one quantity.

    Everything here is measured on the belief plane: only components whose estimate has direct lineage
    contribute, and only edges whose two ends are both directly observed contribute to the correlation."""

    base_rate: float
    rho: dict[str, float]
    n_observed: int
    n_pairs: int


def measure_exposure(
    beliefs: Mapping[UUID, ComponentBelief],
    registry: AssetRegistry,
    cfg: TCDPConfig,
    q: str,
    allowed: frozenset[str],
    prior_core: Callable[[UUID], Estimate],
) -> ExposureStats:
    """Defect base rate and edge correlation of the defect indicator, shrunk toward the configured priors."""
    obs: dict[UUID, float] = {}
    for bid, b in beliefs.items():
        est = b.estimates[q]
        if q in b.valid and est.direct_lineage and est.known:
            obs[bid] = _elevated_p(est, prior_core(bid), cfg.elevated_sd)
    w0 = float(cfg.elevated_prior.get(q, 0.1))
    n = len(obs)
    n0 = max(cfg.base_rate_pseudo_n, 0.0)
    base = (sum(obs.values()) + n0 * w0) / (n + n0) if n + n0 > 0 else w0
    seen: set[EdgeKey] = set()
    pairs: list[tuple[float, float]] = []
    for bid in sorted(obs, key=str):
        for nid, rtype in registry.neighbours(bid, allowed):
            if nid not in obs:
                continue
            key = edge_key(bid, nid, rtype)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((obs[bid], obs[nid]))
    rho_hat = 0.0
    if pairs:
        marg = sum(a + b for a, b in pairs) / (2.0 * len(pairs))
        denom = marg * (1.0 - marg)
        if denom > 1e-6:
            joint = sum(a * b for a, b in pairs) / len(pairs)
            rho_hat = min(max((joint - marg * marg) / denom, 0.0), cfg.max_measured_correlation)
    m = float(len(pairs))
    p0 = max(cfg.correlation_pseudo_pairs, 0.0)
    rho = {
        rt: (m * rho_hat + p0 * cfg.correlation.get(rt, 0.0)) / (m + p0) if m + p0 > 0 else 0.0
        for rt in allowed
    }
    return ExposureStats(base_rate=base, rho=rho, n_observed=n, n_pairs=len(pairs))


def _exposure_message(
    core: Estimate,
    population: Estimate,
    w0: float,
    stats: ExposureStats,
    sources: Sequence[tuple[float, float, float, float, str]],
    max_shift_sd: float,
    pseudo_weight: float,
) -> tuple[float, float]:
    """Two-component posterior: background core vs shared exposure.

    ``sources`` are (elevated probability, level, level variance, gate, relation type) of the neighbours.
    With no information (every conditional equal to the base rate) the result is the population prior."""
    w = min(max(stats.base_rate, 1e-6), 1.0 - 1e-6)
    odds = _logit(w)
    num, den, spread = 0.0, 0.0, 0.0
    for p_s, level, var, gate, rtype in sources:
        q_s = w + gate * stats.rho.get(rtype, 0.0) * (p_s - w)
        odds += _logit(min(max(q_s, 1e-6), 1.0 - 1e-6)) - _logit(w)
        weight = gate * p_s
        num += weight * level
        den += weight
        spread += weight * var
    big = 1.0 / (1.0 + math.exp(-max(min(odds, 50.0), -50.0)))
    tail = core.level + (population.level - core.level) / w0 if population.level > core.level else None
    mu_hi = tail if tail is not None else core.level + 2.0 * core.sd
    c0 = max(pseudo_weight, 0.0)
    level_hi = (num + c0 * mu_hi) / (den + c0) if den + c0 > 0 else mu_hi
    var_hi = max((spread + c0 * core.level_var) / (den + c0) if den + c0 > 0 else core.level_var, 1e-30)
    if den > 0:
        var_hi += sum(g * p * (lv - level_hi) ** 2 for p, lv, _, g, _ in sources) / den
    mean = (1.0 - big) * core.level + big * level_hi
    second = (1.0 - big) * (core.level_var + core.level**2) + big * (var_hi + level_hi**2)
    var = max(second - mean * mean, 1e-30)
    if max_shift_sd > 0:
        lo, hi = core.level - max_shift_sd * core.sd, core.level + max_shift_sd * core.sd
        mean = max(lo, min(hi, mean))
    return mean, var


def _fuse(prior: Estimate, messages: Sequence[tuple[float, float, float]]) -> tuple[float, float]:
    """Gated product of message likelihoods relative to the prior (each message's excess information)."""
    p0 = 1.0 / prior.level_var
    precision, weighted = p0, prior.level * p0
    for mean, var, g in messages:
        extra = max(1.0 / var - p0, 0.0)
        precision += g * extra
        weighted += g * extra * mean
    return weighted / precision, 1.0 / precision


def infer(
    beliefs: Mapping[UUID, ComponentBelief],
    registry: AssetRegistry,
    edge_ids: Mapping[EdgeKey, UUID],
    cfg: TCDPConfig,
    prior_of: Callable[[str, Estimate], Estimate] | None = None,
) -> dict[tuple[UUID, str], Inference | None]:
    """Jacobi pass(es) over all non-direct targets; ``None`` means "no admissible support".

    ``prior_of`` maps a component's Gaussian-core prior to the population prior the estimates are actually
    formed under (the moment-matched heavy-tailed prior of the sensor-characterised model), so the Gaussian
    conditional compares posteriors with a consistent prior on both ends of the edge."""

    def pri(q: str, core: Estimate) -> Estimate:
        return core if prior_of is None else prior_of(q, core)

    if cfg.mode is PropagationMode.NONE:
        return {}
    generic = cfg.mode is PropagationMode.GENERIC
    exposure = not generic and cfg.message_model is MessageModel.MEASURED_EXPOSURE
    iterations = cfg.generic_iterations if generic else 1
    current: dict[tuple[UUID, str], Estimate] = {
        (bid, q): b.estimates[q] for bid, b in beliefs.items() for q in PROPAGATED_QUANTITIES
    }
    stats: dict[str, ExposureStats] = {}
    if exposure:
        for q in PROPAGATED_QUANTITIES:
            allow = frozenset(cfg.mechanism_relations[QUANTITY_MECHANISM[q].value])
            stats[q] = measure_exposure(
                beliefs,
                registry,
                cfg,
                q,
                allow,
                lambda bid, q=q: beliefs[bid].prior[q],  # type: ignore[misc]
            )
    result: dict[tuple[UUID, str], Inference | None] = {}
    for _ in range(iterations):
        nxt: dict[tuple[UUID, str], Inference | None] = {}
        for tid, target in beliefs.items():
            for q in PROPAGATED_QUANTITIES:
                if target.estimates[q].direct_lineage or (not generic and q not in target.valid):
                    continue
                allowed = None if generic else frozenset(cfg.mechanism_relations[QUANTITY_MECHANISM[q].value])
                msgs: list[tuple[float, float, float]] = []
                exp_msgs: list[tuple[float, float, float, float, str]] = []
                sources: list[MessageSource] = []
                for sid, rtype in registry.neighbours(tid, allowed):
                    src_b = beliefs.get(sid)
                    if src_b is None:
                        continue
                    src = current[(sid, q)]
                    if generic:
                        if not src.known:
                            continue
                        gate, rho = 1.0, cfg.generic_correlation
                    else:
                        if q not in src_b.valid or not (src.direct_lineage and src.known):
                            continue
                        gate = src_b.direct_support
                        rho = stats[q].rho.get(rtype, 0.0) if exposure else cfg.correlation.get(rtype, 0.0)
                        if gate < cfg.min_source_support or gate * rho < cfg.min_gate:
                            continue
                    if exposure:
                        exp_msgs.append(
                            (
                                _elevated_p(src, src_b.prior[q], cfg.elevated_sd),
                                src.level,
                                src.level_var,
                                gate,
                                rtype,
                            )
                        )
                        msgs.append((src.level, src.level_var, gate))
                    else:
                        bound = 0.0 if generic else cfg.max_shift_sd
                        mean, var = _message(pri(q, target.prior[q]), pri(q, src_b.prior[q]), src, rho, bound)
                        msgs.append((mean, var, gate))
                    sources.append(
                        MessageSource(
                            src_b.belief_id,
                            edge_ids.get(edge_key(tid, sid, rtype)),
                            rtype,
                            src.provenance_id,
                            gate,
                        )
                    )
                if not msgs:
                    nxt[(tid, q)] = None
                    continue
                if exposure:
                    mean, var = _exposure_message(
                        target.prior[q],
                        pri(q, target.prior[q]),
                        float(cfg.elevated_prior.get(q, 0.1)),
                        stats[q],
                        exp_msgs,
                        cfg.max_shift_sd,
                        cfg.severity_pseudo_weight,
                    )
                else:
                    mean, var = _fuse(pri(q, target.prior[q]), msgs)
                support = 1.0 - math.prod(1.0 - min(g, 1.0) for _, _, g in msgs)
                p = target.prior[q]
                nxt[(tid, q)] = Inference(tid, q, mean, var, p.rate, p.rate_var, support, tuple(sources))
        result = nxt
        if generic:
            for key, inf in nxt.items():
                if inf is not None:
                    current[key] = Estimate(
                        level=inf.mean, level_var=inf.var, status=KnowledgeStatus.INFERRED
                    )
    return result


def relational_contamination(
    claims_degraded: Sequence[bool], truly_healthy: Sequence[bool], has_degraded_neighbour: Sequence[bool]
) -> float | None:
    """RC = P(estimate claims degradation | truly healthy, hidden, with a degraded neighbour) (ch10).

    Inputs are per hidden component. Returns None when no component qualifies (not evaluable)."""
    if not len(claims_degraded) == len(truly_healthy) == len(has_degraded_neighbour):
        raise ValueError("length mismatch")
    pool = [
        c for c, h, n in zip(claims_degraded, truly_healthy, has_degraded_neighbour, strict=True) if h and n
    ]
    return None if not pool else sum(pool) / len(pool)
