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
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from conrad.domains.technical.config import PropagationMode, TCDPConfig
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
) -> dict[tuple[UUID, str], Inference | None]:
    """Jacobi pass(es) over all non-direct targets; ``None`` means "no admissible support"."""
    if cfg.mode is PropagationMode.NONE:
        return {}
    generic = cfg.mode is PropagationMode.GENERIC
    iterations = cfg.generic_iterations if generic else 1
    current: dict[tuple[UUID, str], Estimate] = {
        (bid, q): b.estimates[q] for bid, b in beliefs.items() for q in PROPAGATED_QUANTITIES
    }
    result: dict[tuple[UUID, str], Inference | None] = {}
    for _ in range(iterations):
        nxt: dict[tuple[UUID, str], Inference | None] = {}
        for tid, target in beliefs.items():
            for q in PROPAGATED_QUANTITIES:
                if target.estimates[q].direct_lineage or (not generic and q not in target.valid):
                    continue
                allowed = None if generic else frozenset(cfg.mechanism_relations[QUANTITY_MECHANISM[q].value])
                msgs: list[tuple[float, float, float]] = []
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
                        gate, rho = src_b.direct_support, cfg.correlation.get(rtype, 0.0)
                        if gate < cfg.min_source_support or gate * rho < cfg.min_gate:
                            continue
                    bound = 0.0 if generic else cfg.max_shift_sd
                    mean, var = _message(target.prior[q], src_b.prior[q], src, rho, bound)
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
                mean, var = _fuse(target.prior[q], msgs)
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
