"""Model2T baselines behind one interface (ch10 required 2T baselines). BELIEF PLANE.

B1 LATEST_OBSERVATION, B0 SINGLE_FRAME, B5/B8 INDEPENDENT_COMPONENT (analytic core, no propagation),
B9 GENERIC_RELATIONAL (mechanism-agnostic), B2 GRU_TEMPORAL (small torch GRU), and the candidates
TCDP (analytic engine) and MODEL2T (the full Model2Child path, same operators).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import numpy as np
import torch
from torch import nn

from conrad.domains.technical.config import YEAR_S, Model2TConfig, PropagationMode
from conrad.domains.technical.direct import measurements_of
from conrad.domains.technical.engine import StructuralBeliefEngine
from conrad.domains.technical.model import Model2T
from conrad.domains.technical.registry import CORROSION_DEPTH, CRACK_LENGTH, SURFACE_ANOMALY, AssetRegistry
from conrad.persistence.repository import Repository
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence
from conrad.schemas.timebase import TimeStamp

Estimate2 = tuple[float, float] | None


class StructuralEstimator(ABC):
    name: str

    @abstractmethod
    def reset(self, registry: AssetRegistry, now: TimeStamp) -> None: ...

    @abstractmethod
    def step(self, now: TimeStamp, evidence: Sequence[Evidence]) -> None:
        """Advance to ``now`` (physical time) and consume the evidence measured up to ``now``."""

    @abstractmethod
    def estimate(self, registry_id: UUID, quantity: str) -> Estimate2:
        """(mean, variance) or None when the estimator makes no claim (UNKNOWN)."""


def _associate(ev: Evidence, registry: AssetRegistry) -> UUID | None:
    best = max(
        (c for c in ev.entity_candidates if c.registry_entity_id in registry.components),
        key=lambda c: c.score,
        default=None,
    )
    return None if best is None else best.registry_entity_id


class EngineEstimator(StructuralEstimator):
    """Analytic core with a given propagation mode (NONE = independent component, GENERIC, TCDP)."""

    def __init__(self, mode: PropagationMode, config: Model2TConfig | None = None, seed: int = 0) -> None:
        self.mode, self.config, self.seed = mode, config or Model2TConfig(), seed
        self.name = {
            PropagationMode.NONE: "INDEPENDENT_COMPONENT",
            PropagationMode.GENERIC: "GENERIC_RELATIONAL",
            PropagationMode.TCDP: "TCDP",
        }[mode]
        self.engine: StructuralBeliefEngine | None = None
        self._t: TimeStamp | None = None

    def reset(self, registry: AssetRegistry, now: TimeStamp) -> None:
        ids = IdFactory(seed=self.seed, namespace=f"estimator/{self.name}")
        self.engine, self._t = StructuralBeliefEngine(registry, self.config, ids, now, self.mode), now

    def step(self, now: TimeStamp, evidence: Sequence[Evidence]) -> None:
        assert self.engine is not None and self._t is not None
        self.engine.predict(now.delta_s(self._t), now)
        self._t = now
        self.engine.apply_evidence(evidence, now)
        self.engine.propagate(now)
        self.engine.pending.clear()

    def estimate(self, registry_id: UUID, quantity: str) -> Estimate2:
        assert self.engine is not None
        b = self.engine.beliefs.get(registry_id)
        if b is None or not b.estimates[quantity].known:
            return None
        e = b.estimates[quantity]
        return e.level, e.level_var

    def latent(self, registry_id: UUID, quantity: str) -> Estimate2:
        """The internal estimate whether or not it is claimed (e.g. a crack only ever seen as non-detected
        is UNKNOWN as a claim but carries censored direct evidence). Evaluation use only."""
        assert self.engine is not None
        b = self.engine.beliefs.get(registry_id)
        if b is None or quantity not in b.valid:
            return None
        e = b.estimates[quantity]
        return e.level, e.level_var


class Model2TEstimator(StructuralEstimator):
    """The full Model2Child path (ingest / predict / update_beliefs); optional Repository persistence."""

    name = "MODEL2T"

    def __init__(
        self,
        config: Model2TConfig | None = None,
        seed: int = 0,
        *,
        repository: Repository | None = None,
        run_id: UUID | None = None,
    ) -> None:
        self.config, self.seed = config or Model2TConfig(), seed
        self.repository, self.run_id = repository, run_id
        self.model: Model2T | None = None
        self._t: TimeStamp | None = None

    def reset(self, registry: AssetRegistry, now: TimeStamp) -> None:
        ids = IdFactory(seed=self.seed, namespace="model2t")
        self.model = Model2T(ids, self.config, repository=self.repository, run_id=self.run_id)
        ctx: dict[str, Any] = {"asset_registry": _registry_dict(registry), "timestamp": now}
        self.model.initialize(ctx)
        self._t = now

    def step(self, now: TimeStamp, evidence: Sequence[Evidence]) -> None:
        assert self.model is not None and self._t is not None
        self.model.predict(now.delta_s(self._t), now)
        self._t = now
        self.model.ingest(evidence)
        self.model.update_beliefs(now)

    def estimate(self, registry_id: UUID, quantity: str) -> Estimate2:
        assert self.model is not None
        b = self.model.beliefs.get(registry_id)
        if b is None or not b.estimates[quantity].known:
            return None
        return b.estimates[quantity].level, b.estimates[quantity].level_var

    def latent(self, registry_id: UUID, quantity: str) -> Estimate2:
        """Internal estimate whether or not it is claimed (see EngineEstimator.latent). Evaluation use only."""
        assert self.model is not None
        b = self.model.beliefs.get(registry_id)
        if b is None or quantity not in b.valid:
            return None
        return b.estimates[quantity].level, b.estimates[quantity].level_var


def _registry_dict(registry: AssetRegistry) -> dict[str, Any]:
    comps = [
        {
            "registry_id": c.registry_id,
            "component_type": c.component_type,
            "material": c.material,
            "coating": c.coating,
            "wall_thickness_m": c.wall_thickness_m,
            "parent_id": c.parent_id,
        }
        for c in registry.components.values()
    ]
    rels = [{"source": r.source, "target": r.target, "type": r.relation_type} for r in registry.relations]
    return {"components": comps, "relationships": rels}


class LatestObservation(StructuralEstimator):
    """B1: hold the last measurement forever, with the model's nominal measurement variance."""

    name = "LATEST_OBSERVATION"

    def __init__(self, config: Model2TConfig | None = None, clear_each_step: bool = False) -> None:
        self.config, self.clear = config or Model2TConfig(), clear_each_step
        self.last: dict[tuple[UUID, str], tuple[float, float]] = {}
        self.registry: AssetRegistry | None = None

    def reset(self, registry: AssetRegistry, now: TimeStamp) -> None:
        self.registry, self.last = registry, {}

    def step(self, now: TimeStamp, evidence: Sequence[Evidence]) -> None:
        assert self.registry is not None
        if self.clear:
            self.last = {}
        frame: dict[tuple[UUID, str], list[tuple[float, float]]] = {}
        for ev in evidence:
            rid = _associate(ev, self.registry)
            if rid is None or ev.reliability < self.config.direct.min_reliability:
                continue
            for q, z in measurements_of(ev).items():
                var = self.config.direct.sigma_m[q] ** 2 / max(
                    ev.reliability, self.config.direct.min_reliability
                )
                frame.setdefault((rid, q), []).append((z, var))
        for key, vals in frame.items():
            self.last[key] = (float(np.mean([v for v, _ in vals])), float(np.mean([v for _, v in vals])))

    def estimate(self, registry_id: UUID, quantity: str) -> Estimate2:
        return self.last.get((registry_id, quantity))


class LatestObservationDebiased(LatestObservation):
    """B1 variant: the latest reading divided by the sensor's DECLARED median sizing factor (Model2T's own
    datasheet value). Separates "knows the datasheet" from "does inference"."""

    name = "LATEST_OBSERVATION_DEBIASED"

    def estimate(self, registry_id: UUID, quantity: str) -> Estimate2:
        e = super().estimate(registry_id, quantity)
        if e is None:
            return None
        sc = self.config.sensor
        f = {CORROSION_DEPTH: sc.wall_sizing_median_factor, CRACK_LENGTH: sc.crack_sizing_median_factor}
        k = f.get(quantity, 1.0)
        return e[0] / k, e[1] / (k * k)


class SingleFrame(LatestObservation):
    """B0: only the current frame; anything not seen now is UNKNOWN."""

    name = "SINGLE_FRAME"

    def __init__(self, config: Model2TConfig | None = None) -> None:
        super().__init__(config, clear_each_step=True)


GRU_FEATURES = 6
"""[wall_mm, crack_mm, anomaly, observed, reliability, dt_yr]"""


class _GRUNet(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(GRU_FEATURES, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return torch.as_tensor(self.out(h))


class GRUTemporal(StructuralEstimator):
    """B2: per-component GRU over the inspection history -> (corrosion mm, crack mm) mean + log-var."""

    name = "GRU_TEMPORAL"

    def __init__(self, hidden: int = 16, epochs: int = 60, lr: float = 1e-2, seed: int = 0) -> None:
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed
        torch.manual_seed(seed)
        self.net = _GRUNet(hidden)
        self.fitted = False
        self.history: dict[UUID, list[list[float]]] = {}
        self.registry: AssetRegistry | None = None
        self._t: TimeStamp | None = None

    def fit(self, sequences: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> list[float]:
        """Each item: features [T, 6], targets [T, 2] in mm, mask [T, 2] (True = supervised)."""
        x = torch.tensor(np.stack([s[0] for s in sequences]), dtype=torch.float32)
        y = torch.tensor(np.stack([s[1] for s in sequences]), dtype=torch.float32)
        m = torch.tensor(np.stack([s[2] for s in sequences]), dtype=torch.float32)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        losses = []
        for _ in range(self.epochs):
            opt.zero_grad()
            out = self.net(x)
            mean, logv = out[..., :2], out[..., 2:].clamp(-8.0, 6.0)
            nll = 0.5 * (logv + (mean - y) ** 2 * torch.exp(-logv))
            loss = (nll * m).sum() / m.sum().clamp_min(1.0)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        self.fitted = True
        return losses

    def reset(self, registry: AssetRegistry, now: TimeStamp) -> None:
        self.registry, self._t = registry, now
        self.history = {rid: [] for rid in registry.stateful_ids}

    def step(self, now: TimeStamp, evidence: Sequence[Evidence]) -> None:
        assert self.registry is not None and self._t is not None
        dt = now.delta_s(self._t) / YEAR_S
        self._t = now
        rows = {rid: [0.0, 0.0, 0.0, 0.0, 0.0, dt] for rid in self.history}
        for ev in evidence:
            rid = _associate(ev, self.registry)
            if rid is None or rid not in rows:
                continue
            meas = measurements_of(ev)
            rows[rid] = [
                1e3 * meas.get(CORROSION_DEPTH, 0.0),
                1e3 * meas.get(CRACK_LENGTH, 0.0),
                meas.get(SURFACE_ANOMALY, 0.0),
                1.0,
                ev.reliability,
                dt,
            ]
        for rid, row in rows.items():
            self.history[rid].append(row)

    def estimate(self, registry_id: UUID, quantity: str) -> Estimate2:
        if not self.fitted:
            raise RuntimeError("GRU baseline must be fitted before use")
        hist = self.history.get(registry_id)
        if not hist or not any(r[3] for r in hist) or quantity not in (CORROSION_DEPTH, CRACK_LENGTH):
            return None
        with torch.no_grad():
            out = self.net(torch.tensor([hist], dtype=torch.float32))[0, -1]
        k = 0 if quantity == CORROSION_DEPTH else 1
        return float(out[k]) * 1e-3, math.exp(float(out[2 + k].clamp(-8.0, 6.0))) * 1e-6
