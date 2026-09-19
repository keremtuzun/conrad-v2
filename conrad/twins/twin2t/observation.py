"""Observation layer O_t^m = H_m(S_t, V_t, eta_m) (ch10 Three observation fidelity levels; ch11 Observation
layer, Observation masking, Contradictory evidence generation, Sensor degradation curriculum).

Fidelity levels: T0 abstract (interpretable structural measurements), T1 feature-level (truth-conditioned
feature vectors), T2 sensor-level (a registered renderer hook paints :func:`surface_appearance` into an
RGB/sonar array; Twin2T itself does not render).

Sensing-quality inputs come from ``SensingContext.degradation``:
  ``visibility:<entity uuid>`` (absent = 0 = not observable), ``turbidity``, ``biofouling_cover``,
  ``occlusion``, ``corruption`` (curriculum level), ``sensor_bias_m``, ``contradiction`` in [-1, 1],
  ``missing_modality`` (>= 0.5 drops the modality).

T0 measurement model (``ObservationConfig.measurement_model``, see docs/audits/STRUCTURAL_LINEAGE_AUDIT.md):
REALISTIC sizes a crack only over its in-view part, detects it with a log-logistic POD in length and viewing
conditions, and applies multiplicative error with a persistent per (component, sensor) bias that repeated
looks cannot average away. IDEALISED_ADDITIVE (truth + ~1 mm additive noise) is kept to reproduce old runs.

TRUTH PLANE (reads truth, emits sensor-shaped data). implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.observation import SensorHealth
from conrad.twins.twin2t.config import ObservationConfig
from conrad.twins.twin2t.state import ComponentRuntime, status_of

PIT_DIAMETER_TO_DEPTH = 5.0
"""ENGINEERING_ESTIMATE: pit mouth diameter / depth ratio used only for texture scale."""


class FidelityLevel(str, Enum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"


T0_MEASUREMENTS: tuple[str, ...] = ("apparent_wall_loss", "surface_anomaly_score", "crack_indication_length")
T0_UNITS: tuple[str, ...] = ("m", "1", "m")

SensorLevelRenderer = Callable[[dict[str, Any], np.random.Generator], np.ndarray]
"""(surface appearance, rng) -> sensor array. Supplied by Twin2S / Unity integration."""


class SensorLevelHookMissing(RuntimeError):
    """T2 was requested but no renderer is registered for the modality (never silently downgraded)."""


@dataclass(frozen=True)
class SensingQuality:
    turbidity: float
    biofouling_cover: float
    occlusion: float
    corruption: float
    sensor_bias_m: float
    contradiction: float
    missing_modality: bool

    @staticmethod
    def from_degradation(deg: dict[str, float]) -> SensingQuality:
        def unit(key: str) -> float:
            v = float(deg.get(key, 0.0))
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"degradation[{key!r}] must be in [0, 1], got {v}")
            return v

        contra = float(deg.get("contradiction", 0.0))
        if not -1.0 <= contra <= 1.0:
            raise ValueError("degradation['contradiction'] must be in [-1, 1]")
        return SensingQuality(
            unit("turbidity"),
            unit("biofouling_cover"),
            unit("occlusion"),
            unit("corruption"),
            float(deg.get("sensor_bias_m", 0.0)),
            contra,
            float(deg.get("missing_modality", 0.0)) >= 0.5,
        )


def visibility_of(deg: dict[str, float], entity_id: UUID, occlusion: float) -> float:
    v = float(deg.get(f"visibility:{entity_id}", 0.0))
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"visibility for {entity_id} must be in [0, 1]")
    return v * (1.0 - occlusion)


def surface_appearance_of(rt: ComponentRuntime) -> dict[str, Any]:
    """Renderer-facing appearance parameters (TRUTH PLANE: consumed by Twin2S/Unity, never by Model 2)."""
    s, v = rt.state, rt.state.validity
    return {
        "rust_coverage_fraction": s.corrosion_area_fraction if v.get("corrosion_area_fraction") else 0.0,
        "pitting_texture_scale_m": PIT_DIAMETER_TO_DEPTH * s.corrosion_depth_m
        if v.get("corrosion_depth_m")
        else 0.0,
        "crack_visible_length_m": s.crack_length_m if v.get("crack_length_m") else 0.0,
        "coating_breakdown_fraction": s.coating_breakdown_fraction
        if v.get("coating_breakdown_fraction")
        else 0.0,
        "material": None if rt.component.material is None else rt.component.material.name,
        "component_type": rt.component.component_type.name,
        "condition": status_of(s, rt.effective_wall_m).value,
    }


@dataclass(frozen=True)
class MeasurementDraft:
    values: tuple[float, ...]
    names: tuple[str, ...]
    units: tuple[str, ...]
    noise_gain: float
    health: SensorHealth
    contradiction_injected: bool


class ObservationModel:
    def __init__(self, config: ObservationConfig, seed: int) -> None:
        self.config = config
        proj_rng = np.random.default_rng(np.random.SeedSequence([seed & 0xFFFFFFFF, 0x7A2]))
        self._projection = proj_rng.normal(0.0, 1.0, size=(config.feature_dim, 6))
        self._bias_rng = np.random.default_rng(np.random.SeedSequence([seed & 0xFFFFFFFF, 0xB1A5]))
        self._bias: dict[tuple[UUID, UUID | None], tuple[float, float]] = {}

    def noise_gain(self, q: SensingQuality) -> float:
        c = self.config
        return (
            1.0
            + c.turbidity_noise_gain * q.turbidity
            + c.biofouling_noise_gain * q.biofouling_cover
            + (c.corruption_noise_gain * q.corruption)
        )

    @staticmethod
    def health(q: SensingQuality, gain: float) -> SensorHealth:
        if q.corruption >= 0.9:
            return SensorHealth.FAULT
        return SensorHealth.DEGRADED if gain > 2.0 or q.corruption >= 0.5 else SensorHealth.OK

    def _surface_signal(self, rt: ComponentRuntime, q: SensingQuality) -> float:
        app = surface_appearance_of(rt)
        return (1.0 - q.biofouling_cover) * (
            0.7 * app["rust_coverage_fraction"] + 0.3 * app["coating_breakdown_fraction"]
        )

    def persistent_bias(self, entity_id: UUID, sensor_id: UUID | None) -> tuple[float, float]:
        """(ln wall-loss bias, ln crack-sizing bias) for one (component, sensor) pair: drawn once, then fixed.

        Drawn lazily from a dedicated stream in first-observation order, so the value depends on the sensing
        history and never on the world-entity UUID itself.
        """
        key = (entity_id, sensor_id)
        if key not in self._bias:
            c = self.config
            self._bias[key] = (
                float(self._bias_rng.normal(0.0, c.wall_systematic_rel_sigma)),
                float(self._bias_rng.normal(0.0, c.crack_systematic_rel_sigma)),
            )
        return self._bias[key]

    def crack_pod(self, visible_length_m: float, gain: float) -> float:
        """REALISTIC probability of detecting a crack whose in-view length is ``visible_length_m``."""
        if visible_length_m <= 0.0:
            return 0.0
        c = self.config
        z = (math.log(visible_length_m) - math.log(c.crack_pod_a50_m * gain)) / c.crack_pod_log_width
        return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))

    def t0(
        self,
        rt: ComponentRuntime,
        q: SensingQuality,
        rng: np.random.Generator,
        *,
        visibility: float = 1.0,
        sensor_id: UUID | None = None,
    ) -> MeasurementDraft:
        c, g = self.config, self.noise_gain(q)
        s = rt.state
        contra = q.contradiction != 0.0
        if c.measurement_model == "IDEALISED_ADDITIVE":  # pre-audit model, same RNG order as before
            wall = s.corrosion_depth_m + q.sensor_bias_m + float(rng.normal(0.0, c.wall_loss_sigma_m * g))
            surf = self._surface_signal(rt, q) + q.contradiction * c.contradiction_magnitude
            surf = float(np.clip(surf + rng.normal(0.0, c.anomaly_sigma * g), 0.0, 1.0))
            detect = s.crack_length_m >= c.crack_detection_limit_m * g
            crack = s.crack_length_m if detect else 0.0
            crack = max(0.0, crack + float(rng.normal(0.0, c.crack_sigma_m * g)))
            return MeasurementDraft(
                (wall, surf, crack), T0_MEASUREMENTS, T0_UNITS, g, self.health(q, g), contra
            )
        b_wall, b_crack = self.persistent_bias(rt.component.entity_id, sensor_id)
        z = rng.normal(0.0, 1.0, size=5)  # fixed draw count: branch choices never shift the RNG stream
        u = float(rng.random())
        wall = s.corrosion_depth_m * math.exp(b_wall + c.wall_rel_sigma * g * float(z[0]))
        wall += q.sensor_bias_m + c.wall_loss_sigma_m * g * float(z[1])
        surf = self._surface_signal(rt, q) + q.contradiction * c.contradiction_magnitude
        surf = float(np.clip(surf + c.anomaly_sigma * g * float(z[2]), 0.0, 1.0))
        vis = float(np.clip(visibility, 0.0, 1.0))
        in_view = s.crack_length_m * vis**c.crack_visibility_exponent
        crack = 0.0
        if u < self.crack_pod(in_view, g):
            crack = (
                in_view
                * c.crack_sizing_median_factor
                * math.exp(b_crack + c.crack_rel_sigma * g * float(z[3]))
            )
        crack = max(0.0, crack + c.crack_sigma_m * g * float(z[4]))
        return MeasurementDraft((wall, surf, crack), T0_MEASUREMENTS, T0_UNITS, g, self.health(q, g), contra)

    def t1(self, rt: ComponentRuntime, q: SensingQuality, rng: np.random.Generator) -> MeasurementDraft:
        c, g = self.config, self.noise_gain(q)
        s, wall = rt.state, max(rt.effective_wall_m, 1e-9)
        x = np.array(
            [
                s.corrosion_depth_m / wall,
                self._surface_signal(rt, q) + q.contradiction * c.contradiction_magnitude,
                s.coating_breakdown_fraction,
                s.crack_length_m / 0.05,
                s.crack_depth_m / wall,
                1.0,
            ]
        )
        feat = np.tanh(self._projection @ x) + rng.normal(0.0, c.anomaly_sigma * g, size=c.feature_dim)
        names = tuple(f"t1_feature_{i}" for i in range(c.feature_dim))
        return MeasurementDraft(
            tuple(float(v) for v in feat),
            names,
            ("1",) * c.feature_dim,
            g,
            self.health(q, g),
            q.contradiction != 0.0,
        )


def coverage_visibility(
    entity_ids: list[UUID] | tuple[UUID, ...], fraction: float, rng: np.random.Generator, level: float = 1.0
) -> dict[str, float]:
    """Observation coverage control C_t (ch10: 100/50/20/5 %): visibility entries for a random subset."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("coverage fraction must be in [0, 1]")
    ids = sorted(entity_ids, key=str)
    k = round(fraction * len(ids))
    chosen = rng.choice(len(ids), size=k, replace=False) if k else np.array([], dtype=int)
    return {f"visibility:{ids[int(i)]}": float(level) for i in sorted(chosen)}
