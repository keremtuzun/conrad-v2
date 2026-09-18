"""Degradation state V1 and per-component hidden parameters (ch10 Degradation state V1, State masks).

Units are explicit in field names (SI: m, Pa, s). ``validity`` is M_i^state: a masked dimension keeps
the value 0 and is excluded from losses; it is never "healthy" by fabrication.

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from conrad.twins.twin2t.priors import SampledParameter
from conrad.twins.twin2t.registry import Mechanism
from conrad.twins.twin2t.topology import StructuralComponent

STATE_DIMENSIONS: tuple[str, ...] = (
    "corrosion_depth_m",
    "corrosion_area_fraction",
    "coating_breakdown_fraction",
    "crack_length_m",
    "crack_depth_m",
)
"""Order of the D_T axis of the S^true tensor."""

DIMENSION_MECHANISM: dict[str, str] = {
    "corrosion_depth_m": "CORROSION",
    "corrosion_area_fraction": "CORROSION",
    "coating_breakdown_fraction": "COATING_FAILURE",
    "crack_length_m": "FATIGUE",
    "crack_depth_m": "FATIGUE",
}


class DegradationStatus(str, Enum):
    INTACT = "INTACT"
    DEGRADED = "DEGRADED"
    SEVERE = "SEVERE"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class DegradationState:
    corrosion_depth_m: float = 0.0
    corrosion_area_fraction: float = 0.0
    coating_breakdown_fraction: float = 0.0
    crack_length_m: float = 0.0
    crack_depth_m: float = 0.0
    corrosion_exposure_yr: float = 0.0
    """Effective exposure time since last reset (drives the non-linear corrosion law)."""
    coating_age_yr: float = 0.0
    validity: dict[str, bool] = field(default_factory=dict)

    def vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, d)) for d in STATE_DIMENSIONS)

    def mask(self) -> tuple[bool, ...]:
        return tuple(bool(self.validity.get(d, False)) for d in STATE_DIMENSIONS)

    def with_(self, **changes: Any) -> DegradationState:
        return replace(self, **changes)


@dataclass(frozen=True)
class Environment:
    """Environmental modifiers for corrosion (ch10 Environmental / Loading Context)."""

    temperature_c: float = 10.0
    dissolved_oxygen_mg_l: float = 7.0
    cp_active: bool = False
    buried: bool = False


@dataclass(frozen=True)
class Loading:
    load_cycles_per_s: float
    stress_range_pa: float


@dataclass
class ComponentParameters:
    """Hidden mechanism parameters theta for one component. TRUTH: never exported to a belief model."""

    wall_thickness_m: float
    coated: bool
    params: dict[str, SampledParameter]

    def value(self, name: str) -> float:
        return self.params[name].value

    def log(self) -> list[dict[str, Any]]:
        return [self.params[k].as_dict() for k in sorted(self.params)]


@dataclass
class ComponentRuntime:
    """Mutable per-component truth held by the MCDE (static identity + dynamic state + hidden theta)."""

    component: StructuralComponent
    params: ComponentParameters
    state: DegradationState
    environment: Environment
    loading: Loading
    initial_state: DegradationState
    reinforcement_m: float = 0.0
    load_multiplier: float = 1.0
    """Transient loading multiplier set by the topology/load-path coupling (1 = nominal)."""
    stress_concentration: float = 1.0
    """Net-section stress amplification set by the mechanism coupling (1 = nominal)."""

    @property
    def effective_wall_m(self) -> float:
        return self.params.wall_thickness_m + self.reinforcement_m


def validity_for(component: StructuralComponent) -> dict[str, bool]:
    return {d: component.mechanism_valid(Mechanism(m)) for d, m in DIMENSION_MECHANISM.items()}


def status_of(
    state: DegradationState, wall_thickness_m: float, failed_fraction: float = 0.8
) -> DegradationStatus:
    """Wall-loss / crack-depth fraction thresholds (ENGINEERING_ESTIMATE bands)."""
    if not any(state.mask()):
        return DegradationStatus.NOT_APPLICABLE
    loss = max(state.corrosion_depth_m, state.crack_depth_m) / max(wall_thickness_m, 1e-9)
    if loss >= failed_fraction:
        return DegradationStatus.FAILED
    if loss >= 0.3:
        return DegradationStatus.SEVERE
    if loss >= 0.02 or state.crack_length_m > 0.0 or state.corrosion_area_fraction > 0.05:
        return DegradationStatus.DEGRADED
    return DegradationStatus.INTACT
