"""Structural events and interventions: S_t+ = Event(S_t-, e) (ch10 Events, ch11 Maintenance).

INSPECTION never alters truth. Interventions (repair, replacement, cleaning, coating renewal,
reinforcement, maintenance) are what make degradation non-monotone.

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import Any

from conrad.twins.twin2t.mechanisms import paris_integrate, stress_intensity_range
from conrad.twins.twin2t.state import ComponentRuntime, DegradationState, Environment


class StructuralEventType(str, Enum):
    LOAD_SPIKE = "LOAD_SPIKE"
    IMPACT = "IMPACT"
    COATING_FAILURE = "COATING_FAILURE"
    ENVIRONMENT_CHANGE = "ENVIRONMENT_CHANGE"
    INSPECTION = "INSPECTION"
    MAINTENANCE = "MAINTENANCE"
    CLEANING = "CLEANING"
    COATING_RENEWAL = "COATING_RENEWAL"
    REPAIR = "REPAIR"
    REPLACEMENT = "REPLACEMENT"
    REINFORCEMENT = "REINFORCEMENT"


INTERVENTIONS = frozenset(
    {
        StructuralEventType.MAINTENANCE,
        StructuralEventType.CLEANING,
        StructuralEventType.COATING_RENEWAL,
        StructuralEventType.REPAIR,
        StructuralEventType.REPLACEMENT,
        StructuralEventType.REINFORCEMENT,
    }
)
GLOBAL_OK = frozenset({StructuralEventType.ENVIRONMENT_CHANGE, StructuralEventType.INSPECTION})
"""Event types that may omit a target (they then apply to every component)."""


class StructuralEventError(ValueError):
    pass


def parse_event_type(raw: str) -> StructuralEventType | None:
    """Returns None for event types owned by other twins (they are ignored, not fabricated)."""
    try:
        return StructuralEventType(raw.upper())
    except ValueError:
        return None


def _fraction(params: dict[str, Any], key: str, default: float) -> float:
    v = float(params.get(key, default))
    if not 0.0 <= v <= 1.0:
        raise StructuralEventError(f"{key} must be in [0, 1], got {v}")
    return v


def _renew_coating(s: DegradationState) -> DegradationState:
    if not s.validity.get("coating_breakdown_fraction", False):
        return s
    return s.with_(coating_breakdown_fraction=0.0, coating_age_yr=0.0)


def apply_event(rt: ComponentRuntime, etype: StructuralEventType, params: dict[str, Any]) -> dict[str, Any]:
    """Mutates ``rt`` and returns a log record. Deterministic (no randomness in event effects)."""
    s = rt.state
    before = s.vector()
    v = s.validity
    if etype is StructuralEventType.INSPECTION:
        pass
    elif etype is StructuralEventType.LOAD_SPIKE:
        mult = float(params.get("stress_multiplier", 2.0))
        cycles = float(params.get("n_cycles", 1000.0))
        if mult <= 0 or cycles < 0:
            raise StructuralEventError("LOAD_SPIKE needs stress_multiplier > 0 and n_cycles >= 0")
        dsig = rt.loading.stress_range_pa * mult
        y = rt.params.value("geometry_factor_Y")
        above = v.get("crack_length_m") and s.crack_length_m > 0
        if above and stress_intensity_range(s.crack_length_m, dsig, y) >= rt.params.value(
            "delta_k_threshold"
        ):
            a1 = paris_integrate(
                s.crack_length_m,
                cycles,
                rt.params.value("paris_C"),
                rt.params.value("paris_m"),
                y,
                dsig / 1e6,
            )
            a1 = min(a1, 1.0)
            depth = min(rt.effective_wall_m, max(s.crack_depth_m, a1 / rt.params.value("crack_aspect_ratio")))
            s = s.with_(crack_length_m=a1, crack_depth_m=depth)
    elif etype is StructuralEventType.IMPACT:
        if v.get("crack_length_m"):
            seed = float(params.get("crack_length_m", 5.0e-3))
            a = max(s.crack_length_m, seed)
            s = s.with_(
                crack_length_m=a,
                crack_depth_m=min(
                    rt.effective_wall_m, max(s.crack_depth_m, a / rt.params.value("crack_aspect_ratio"))
                ),
            )
        if v.get("coating_breakdown_fraction"):
            dmg = _fraction(params, "coating_damage_fraction", 0.2)
            s = s.with_(coating_breakdown_fraction=min(1.0, s.coating_breakdown_fraction + dmg))
    elif etype is StructuralEventType.COATING_FAILURE:
        if v.get("coating_breakdown_fraction"):
            frac = _fraction(params, "fraction", 1.0)
            s = s.with_(coating_breakdown_fraction=max(s.coating_breakdown_fraction, frac))
    elif etype is StructuralEventType.ENVIRONMENT_CHANGE:
        env = rt.environment
        allowed = {"temperature_c", "dissolved_oxygen_mg_l", "cp_active", "buried"}
        unknown = set(params) - allowed - {"scope"}
        if unknown:
            raise StructuralEventError(f"unknown ENVIRONMENT_CHANGE keys {sorted(unknown)}")
        rt.environment = Environment(
            temperature_c=float(params.get("temperature_c", env.temperature_c)),
            dissolved_oxygen_mg_l=float(params.get("dissolved_oxygen_mg_l", env.dissolved_oxygen_mg_l)),
            cp_active=bool(params.get("cp_active", env.cp_active)),
            buried=bool(params.get("buried", env.buried)),
        )
    elif etype in (StructuralEventType.CLEANING, StructuralEventType.COATING_RENEWAL):
        s = _renew_coating(s)
    elif etype is StructuralEventType.MAINTENANCE:
        s = _renew_coating(s)
        if "cp_active" in params:
            rt.environment = replace(rt.environment, cp_active=bool(params["cp_active"]))
    elif etype is StructuralEventType.REPAIR:
        frac = _fraction(params, "restore_fraction", 1.0)
        s = s.with_(
            corrosion_depth_m=s.corrosion_depth_m * (1.0 - frac),
            corrosion_area_fraction=s.corrosion_area_fraction * (1.0 - frac),
            corrosion_exposure_yr=s.corrosion_exposure_yr * (1.0 - frac),
            crack_length_m=0.0 if params.get("remove_cracks", True) else s.crack_length_m,
            crack_depth_m=0.0 if params.get("remove_cracks", True) else s.crack_depth_m,
        )
        s = _renew_coating(s) if params.get("recoat", True) else s
    elif etype is StructuralEventType.REPLACEMENT:
        s = DegradationState(validity=dict(v))
        rt.reinforcement_m = 0.0
    elif etype is StructuralEventType.REINFORCEMENT:
        add = float(params.get("added_thickness_m", 5.0e-3))
        if add <= 0:
            raise StructuralEventError("REINFORCEMENT needs added_thickness_m > 0")
        rt.reinforcement_m += add
    rt.state = s
    return {
        "event_type": etype.value,
        "intervention": etype in INTERVENTIONS,
        "parameters": dict(params),
        "state_before": list(before),
        "state_after": list(s.vector()),
    }
