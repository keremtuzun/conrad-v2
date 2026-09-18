"""Temporal Belief Dynamics on latent tensors: one interface, baselines and the GRUCell candidate (ch33).

``delta_t_s`` is physical elapsed seconds. No module here receives a sequence index.
Predicted tensors are returned as new objects; the corrected inputs are never modified.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from conrad.core.config import CoreConfig
from conrad.core.primitives import mlp
from conrad.core.tbd_analytic import AnalyticPrediction, AnalyticTBD, PropertyDynamics, normalised_surprise

__all__ = [
    "AnalyticPrediction",
    "AnalyticTBD",
    "GruTBD",
    "HoldLastTBD",
    "LinearExtrapolationTBD",
    "MlpTBD",
    "PredictedBelief",
    "PropertyDynamics",
    "TemporalBeliefDynamics",
    "build_tbd",
    "latent_temporal_surprise",
    "normalised_surprise",
]


@dataclass(frozen=True)
class PredictedBelief:
    """B^- : kept distinct from the corrected B^+ it was computed from."""

    z: Tensor
    u: Tensor
    temporal: Tensor
    delta_t_s: Tensor

    def as_tuple(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.z, self.u, self.temporal


def _check_dt(delta_t_s: Tensor) -> Tensor:
    if bool((delta_t_s < 0).any()) or not bool(torch.isfinite(delta_t_s).all()):
        raise ValueError("delta_t_s must be finite, non-negative physical seconds")
    return torch.log1p(delta_t_s.to(torch.float32)).unsqueeze(-1)


class TemporalBeliefDynamics(nn.Module, ABC):
    """forward(z [N,Dz], u [N,Du], temporal [N,Dt], delta_t_s [N], context [N,Dc]) -> PredictedBelief."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self, z: Tensor, u: Tensor, temporal: Tensor, delta_t_s: Tensor, context: Tensor | None = None
    ) -> PredictedBelief: ...

    def _context(self, z: Tensor, context: Tensor | None) -> Tensor:
        return z.new_zeros(z.shape[0], self.cfg.tbd.context_dim) if context is None else context


class HoldLastTBD(TemporalBeliefDynamics):
    """Baseline: last-state persistence."""

    def forward(
        self, z: Tensor, u: Tensor, temporal: Tensor, delta_t_s: Tensor, context: Tensor | None = None
    ) -> PredictedBelief:
        _check_dt(delta_t_s)
        return PredictedBelief(z.clone(), u.clone(), temporal.clone(), delta_t_s)


class LinearExtrapolationTBD(TemporalBeliefDynamics):
    """Baseline: z + (z - z_prev) / dt_prev * dt. Falls back to hold-last without a previous state."""

    def forward(
        self,
        z: Tensor,
        u: Tensor,
        temporal: Tensor,
        delta_t_s: Tensor,
        context: Tensor | None = None,
        z_prev: Tensor | None = None,
        delta_t_prev_s: Tensor | None = None,
    ) -> PredictedBelief:
        _check_dt(delta_t_s)
        if z_prev is None or delta_t_prev_s is None:
            return PredictedBelief(z.clone(), u.clone(), temporal.clone(), delta_t_s)
        rate = (z - z_prev) / delta_t_prev_s.clamp_min(1e-9).unsqueeze(-1)
        return PredictedBelief(z + rate * delta_t_s.unsqueeze(-1), u.clone(), temporal.clone(), delta_t_s)


class _LearnedTBD(TemporalBeliefDynamics):
    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__(cfg)
        self.in_dim = cfg.belief_dim + cfg.uncertainty_dim + cfg.temporal_dim + 1 + cfg.tbd.context_dim
        self.temporal_update = nn.Linear(cfg.temporal_dim + 1, cfg.temporal_dim)
        self.temporal_norm = nn.LayerNorm(cfg.temporal_dim)

    def _inputs(
        self, z: Tensor, u: Tensor, temporal: Tensor, delta_t_s: Tensor, context: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        log_dt = _check_dt(delta_t_s)
        return torch.cat([z, u, temporal, log_dt, self._context(z, context)], dim=-1), log_dt

    def _temporal(self, temporal: Tensor, log_dt: Tensor) -> Tensor:
        out: Tensor = self.temporal_norm(temporal + self.temporal_update(torch.cat([temporal, log_dt], -1)))
        return out


class MlpTBD(_LearnedTBD):
    """Sanity baseline: [z, u, t, c, dt] -> MLP -> [dz, du] (ch3 TBD baseline implementation)."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__(cfg)
        out = cfg.belief_dim + cfg.uncertainty_dim
        self.net = mlp([self.in_dim, cfg.tbd.mlp_hidden, cfg.tbd.mlp_hidden, out], cfg.dropout)

    def forward(
        self, z: Tensor, u: Tensor, temporal: Tensor, delta_t_s: Tensor, context: Tensor | None = None
    ) -> PredictedBelief:
        x, log_dt = self._inputs(z, u, temporal, delta_t_s, context)
        dz, du = self.net(x).split([self.cfg.belief_dim, self.cfg.uncertainty_dim], dim=-1)
        return PredictedBelief(z + dz, u + du, self._temporal(temporal, log_dt), delta_t_s)


class GruTBD(_LearnedTBD):
    """Candidate: stem (-> 512 -> 256) + GRUCell(256, 256) + 256 -> 128 -> 64 uncertainty drift head."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__(cfg)
        dz = cfg.belief_dim
        self.stem = mlp([self.in_dim, cfg.tbd.stem_hidden, dz], cfg.dropout)
        self.cell = nn.GRUCell(dz, dz)
        self.z_proj = nn.Linear(dz, dz)
        self.drift = mlp([dz, cfg.tbd.drift_hidden, cfg.uncertainty_dim], cfg.dropout)

    def forward(
        self, z: Tensor, u: Tensor, temporal: Tensor, delta_t_s: Tensor, context: Tensor | None = None
    ) -> PredictedBelief:
        x, log_dt = self._inputs(z, u, temporal, delta_t_s, context)
        hidden = self.cell(self.stem(x), z)
        return PredictedBelief(
            self.z_proj(hidden), u + self.drift(hidden), self._temporal(temporal, log_dt), delta_t_s
        )


def build_tbd(cfg: CoreConfig, kind: str) -> TemporalBeliefDynamics:
    kinds: dict[str, type[TemporalBeliefDynamics]] = {
        "hold_last": HoldLastTBD,
        "linear": LinearExtrapolationTBD,
        "mlp": MlpTBD,
        "gru": GruTBD,
    }
    if kind not in kinds:
        raise ValueError(f"unknown TBD kind {kind!r}; expected one of {sorted(kinds)}")
    return kinds[kind](cfg)


def latent_temporal_surprise(z_predicted: Tensor, z_corrected: Tensor) -> Tensor:
    """delta_t = d(B_hat^-, B^+): RMS latent displacement caused by the correction, per belief."""
    return torch.sqrt(((z_corrected - z_predicted) ** 2).mean(dim=-1))
