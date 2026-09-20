"""Temporal prediction over PHYSICAL delta_t (TBD-T analytic default, ch10 Model 2T temporal dynamics).

Constant-rate [level, rate] Kalman prediction with the model's own engineering-prior rates (wall loss, surface
anomaly, and crack length under the CONSTANT_RATE ablation). Crack length under the default REGIME_MIXTURE
model is predicted by ``crack_filter.propagate`` (stable / run-away regimes, uncertainty growing with
delta_t). Results
become PREDICTED (never OBSERVED); direct support decays, epistemic uncertainty grows with horizon.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from uuid import UUID

from conrad.domains.technical.config import YEAR_S, Model2TConfig
from conrad.domains.technical.crack_filter import propagate
from conrad.domains.technical.registry import CRACK_LENGTH
from conrad.domains.technical.state import ComponentBelief, Estimate, summarise, sync_crack
from conrad.schemas.belief import KnowledgeStatus


def predict_estimate(est: Estimate, q: str, dt_s: float, cfg: Model2TConfig) -> None:
    """In place: x' = F x, P' = F P F^T + Q with F = [[1, dt], [0, 1]] (dt in years)."""
    if dt_s <= 0.0:
        return
    dt = dt_s / YEAR_S
    d = cfg.dynamics
    ql = d.level_process_noise.get(q, 0.0) ** 2 * dt
    ql += (d.relative_level_noise.get(q, 0.0) * max(est.level, 0.0)) ** 2 * dt
    qr = d.rate_process_noise.get(q, 0.0) ** 2 * dt
    p00, p01, p11 = est.level_var, est.cov, est.rate_var
    est.level = est.level + est.rate * dt
    est.level_var = p00 + 2 * dt * p01 + dt * dt * p11 + ql
    est.cov = p01 + dt * p11
    est.rate_var = p11 + qr


def predict_belief(
    belief: ComponentBelief, dt_s: float, now_ns: int, provenance_id: UUID, cfg: Model2TConfig
) -> bool:
    """Advance every estimate and the latent prior. Returns True if a reported value changed."""
    if dt_s < 0.0:
        raise ValueError("delta_t must be non-negative physical seconds")
    changed = False
    for q, prior in belief.prior.items():
        predict_estimate(prior, q, dt_s, cfg)
        prior.updated_ns = now_ns
    if dt_s == 0.0:
        return False
    for region in belief.regions.values():
        for q, est in region.estimates.items():
            grid = region.crack_grid if q == CRACK_LENGTH else None
            if grid is None:
                predict_estimate(est, q, dt_s, cfg)
            elif propagate(grid, dt_s, cfg.crack_growth):
                sync_crack(est, grid, cfg)
            est.updated_ns = now_ns
            if est.known:
                est.status = KnowledgeStatus.PREDICTED
                est.provenance_id = provenance_id
                changed = True
    summarise(belief, cfg)
    for est in belief.estimates.values():
        est.updated_ns = now_ns
        if est.known:
            est.status = KnowledgeStatus.PREDICTED
            est.provenance_id = provenance_id
            changed = True
    belief.direct_support *= math.exp(-dt_s / cfg.dynamics.support_timescale_s)
    belief.propagated_support *= math.exp(-dt_s / cfg.dynamics.support_timescale_s)
    belief.ue = min(1.0, belief.ue + cfg.dynamics.epistemic_growth_per_yr * dt_s / YEAR_S)
    if belief.change_status is not KnowledgeStatus.UNKNOWN:
        belief.change_status = KnowledgeStatus.PREDICTED
        belief.change_provenance = provenance_id
    return changed
