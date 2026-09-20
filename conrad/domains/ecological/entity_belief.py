"""Entity beliefs: biofouling cover per structure, benthic patch cover, mobile-group presence (ch12).

Entity evidence associates to a persistent belief by registry asset + position gating (the asset
registry is permitted mission context). Evidence outside every gate starts an unregistered
CANDIDATE belief; evidence inside two gates at once is ambiguous and is not applied.
Cover is a bounded random walk over PHYSICAL time (no invented trend); presence decays to the prior.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from conrad.domains.ecological.config import EntityConfig
from conrad.schemas.belief import Lifecycle
from conrad.schemas.timebase import NS_PER_S

BIOFOULING = "BIOFOULING_ON_STRUCTURE"
BENTHIC = "BENTHIC_PATCH"
MOBILE = "MOBILE_GROUP"
UNKNOWN_CLASS = "UNKNOWN_ECOLOGICAL"
SESSILE_CLASSES = (BIOFOULING, BENTHIC)
_KEYWORDS = (
    (BIOFOULING, ("pipeline", "pipe", "structure", "pile", "jacket", "riser", "hull")),
    (BENTHIC, ("coral", "benthic", "patch", "seagrass", "reef")),
    (MOBILE, ("fish", "school", "shoal", "mobile")),
)
_MAX_COVER_VAR = 0.25  # variance bound of any quantity confined to [0, 1]


def classify_entity_type(entity_type: str) -> str:
    t = entity_type.lower()
    for cls, words in _KEYWORDS:
        if any(w in t for w in words):
            return cls
    return UNKNOWN_CLASS


@dataclass(frozen=True)
class RegistryAsset:
    registry_id: UUID
    entity_type: str
    eco_class: str
    position_m: np.ndarray
    frame_id: str
    radius_m: float
    stress_threshold_c: float | None = None


def parse_registry(entries: Sequence[Mapping[str, Any]], cfg: EntityConfig) -> list[RegistryAsset]:
    out = []
    for e in entries:
        pos = e.get("position_m")
        if pos is None or len(pos) != 3:
            raise ValueError(f"registry asset {e.get('registry_entity_id')} lacks a 3D position_m")
        out.append(
            RegistryAsset(
                registry_id=UUID(str(e["registry_entity_id"])),
                entity_type=str(e["entity_type"]),
                eco_class=str(e.get("eco_class") or classify_entity_type(str(e["entity_type"]))),
                position_m=np.asarray(pos, dtype=np.float64),
                frame_id=str(e.get("frame_id", "WORLD")),
                radius_m=float(e.get("radius_m", cfg.default_radius_m)),
                stress_threshold_c=None
                if e.get("stress_threshold_c") is None
                else float(e["stress_threshold_c"]),
            )
        )
    return out


@dataclass
class EntityBelief:
    belief_id: UUID
    eco_class: str
    position_m: np.ndarray
    frame_id: str
    asset: RegistryAsset | None
    cover_mean: float
    cover_var: float
    presence_p: float
    lifecycle: Lifecycle = Lifecycle.CANDIDATE
    time_ns: int | None = None
    n_hits: int = 0
    nis_ema: float = 1.0
    meas_var_ema: float = 0.0
    ood_ema: float = 0.0
    ambiguous_hits: int = 0
    stress_p: float | None = None
    stress_time_ns: int | None = None
    decline_stat: float = 0.0
    """Sufficient statistic of the entity's OWN cover innovations for a decline of unit rate:
    sum over readings of (-innovation) * elapsed_days / innovation_variance (CEFD confounder control)."""
    decline_norm: float = 0.0
    """Matching normaliser: sum of elapsed_days^2 / innovation variance."""
    stress_drift_per_day: float = 0.0
    """Cover loss per day currently attributed to thermal stress (0 unless ecological coupling is ON)."""

    @property
    def sessile(self) -> bool:
        return self.eco_class in SESSILE_CLASSES


class EntityBeliefStore:
    def __init__(self, cfg: EntityConfig) -> None:
        self.cfg = cfg
        self.beliefs: dict[UUID, EntityBelief] = {}

    def add(
        self,
        belief_id: UUID,
        eco_class: str,
        position: np.ndarray,
        frame_id: str,
        asset: RegistryAsset | None,
    ) -> EntityBelief:
        b = EntityBelief(
            belief_id=belief_id,
            eco_class=eco_class,
            position_m=np.asarray(position, dtype=np.float64),
            frame_id=frame_id,
            asset=asset,
            cover_mean=self.cfg.cover_prior_mean,
            cover_var=min(_MAX_COVER_VAR, self.cfg.cover_prior_sd**2),
            presence_p=self.cfg.presence_prior,
        )
        self.beliefs[belief_id] = b
        return b

    def by_registry(self, registry_id: UUID) -> EntityBelief | None:
        for b in self.beliefs.values():
            if b.asset is not None and b.asset.registry_id == registry_id:
                return b
        return None

    # ------------------------------------------------------------------ association
    def associate(
        self, position: np.ndarray, sigma_m: float, frame_id: str, kind: str, gate_inflation: float = 1.0
    ) -> tuple[EntityBelief | None, bool]:
        """Nearest compatible belief inside its gate. Returns (belief, ambiguous)."""
        scored: list[tuple[float, EntityBelief]] = []
        for b in self.beliefs.values():
            if b.frame_id != frame_id or not _compatible(b.eco_class, kind):
                continue
            d = float(np.linalg.norm(b.position_m - position))
            radius = b.asset.radius_m if b.asset is not None else self.cfg.default_radius_m
            gate = (
                self.cfg.gate_radius_m + radius + self.cfg.gate_sigma_multiplier * sigma_m
            ) * gate_inflation
            if d <= gate:
                scored.append((d, b))
        if not scored:
            return None, False
        scored.sort(key=lambda s: (s[0], str(s[1].belief_id)))
        if len(scored) > 1 and scored[1][0] - scored[0][0] < self.cfg.ambiguity_margin_m:
            return None, True
        return scored[0][1], False

    # ------------------------------------------------------------------ dynamics
    def cover_moments_at(self, b: EntityBelief, to_ns: int, inflation: float = 1.0) -> tuple[float, float]:
        if b.time_ns is None or to_ns <= b.time_ns:
            return b.cover_mean, b.cover_var
        days = (to_ns - b.time_ns) / NS_PER_S / 86400.0
        q = inflation * self.cfg.cover_process_sd_per_sqrt_day**2 * days
        mean = b.cover_mean * math.exp(-b.stress_drift_per_day * days)
        return float(np.clip(mean, 0.0, 1.0)), min(_MAX_COVER_VAR, b.cover_var + q)

    def presence_at(self, b: EntityBelief, to_ns: int) -> float:
        if b.time_ns is None or to_ns <= b.time_ns:
            return b.presence_p
        a = math.exp(-(to_ns - b.time_ns) / NS_PER_S / self.cfg.presence_decay_time_s)
        return self.cfg.presence_prior + a * (b.presence_p - self.cfg.presence_prior)

    def advance(self, b: EntityBelief, to_ns: int, inflation: float = 1.0) -> None:
        if b.time_ns is not None and to_ns <= b.time_ns:
            return
        b.cover_mean, b.cover_var = self.cover_moments_at(b, to_ns, inflation)
        b.presence_p = self.presence_at(b, to_ns)
        b.time_ns = to_ns

    # ------------------------------------------------------------------ measurement
    def update_cover(
        self, b: EntityBelief, y: float, meas_var: float, t_ns: int, inflation: float = 1.0
    ) -> bool:
        late = b.time_ns is not None and t_ns < b.time_ns
        gap_days = 0.0 if b.time_ns is None else max((t_ns - b.time_ns) / NS_PER_S / 86400.0, 0.0)
        if late:
            assert b.time_ns is not None
            lag_days = (b.time_ns - t_ns) / NS_PER_S / 86400.0
            meas_var += inflation * self.cfg.cover_process_sd_per_sqrt_day**2 * lag_days
        else:
            self.advance(b, t_ns, inflation)
        s = b.cover_var + meas_var
        innov = y - b.cover_mean
        drift_applied = b.cover_mean * math.expm1(b.stress_drift_per_day * gap_days)
        self._accumulate_decline(b, innov + drift_applied, b.cover_mean * gap_days, gap_days, s)
        k = b.cover_var / s
        b.cover_mean = float(np.clip(b.cover_mean + k * innov, 0.0, 1.0))
        b.cover_var = max(1e-6, (1.0 - k) * b.cover_var)
        b.nis_ema = 0.8 * b.nis_ema + 0.2 * innov * innov / s
        b.meas_var_ema = 0.8 * b.meas_var_ema + 0.2 * meas_var if b.n_hits else meas_var
        b.n_hits += 1
        return late

    def _accumulate_decline(
        self, b: EntityBelief, innov0: float, scale: float, gap_days: float, s: float
    ) -> None:
        """Sufficient statistics of a sequential likelihood ratio for "the cover is declining".

        ``innov0`` is the innovation the entity would have shown WITHOUT any drift already attributed to
        stress, so the statistic never confirms the drift it produced itself. ``scale`` is the expected
        decline per unit fractional loss rate (cover x elapsed days). For a fractional loss rate d the
        log-likelihood ratio against no decline is ``d * decline_stat - 0.5 * d^2 * decline_norm``."""
        if gap_days <= 0.0 or s <= 0.0 or scale <= 0.0:
            return
        keep = math.exp(-gap_days / max(self.cfg.decline_memory_days, 1e-6))
        b.decline_stat = keep * b.decline_stat + (-innov0) * scale / s
        b.decline_norm = keep * b.decline_norm + scale * scale / s

    def update_detection(self, b: EntityBelief, t_ns: int) -> bool:
        late = b.time_ns is not None and t_ns < b.time_ns
        if not late:
            self.advance(b, t_ns)
        h, fa = self.cfg.presence_hit_rate, self.cfg.presence_false_alarm
        p = b.presence_p
        b.presence_p = h * p / (h * p + fa * (1.0 - p))
        b.n_hits += 1
        return late


def _compatible(eco_class: str, kind: str) -> bool:
    if kind == "cover_fraction":
        return eco_class in (*SESSILE_CLASSES, UNKNOWN_CLASS)
    if kind == "detection":
        return eco_class in (MOBILE, UNKNOWN_CLASS)
    return True
