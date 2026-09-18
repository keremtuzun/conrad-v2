"""Disturbance and recovery events (ch12 Disturbances, ch13 Recovery). TRUTH PLANE.

Every event is logged with its outcome; an event is never dropped silently. Events owned by
other twins are logged as NOT_ECOLOGICAL and left to them.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np

from conrad.schemas.world import ScenarioEvent
from conrad.twins.twin2e.config import MeifeSwitches
from conrad.twins.twin2e.ecology import SESSILE, EntityState
from conrad.twins.twin2e.fields import FieldEngine, region_mask
from conrad.twins.twin2e.priors import KIND_HABITAT, canonical_units

FIELD_EVENTS = ("TEMPERATURE_ANOMALY", "TURBIDITY_SPIKE", "FLOW_CHANGE", "LIGHT_REDUCTION")
ENTITY_EVENTS = (
    "PHYSICAL_DISTURBANCE",
    "HABITAT_DAMAGE",
    "COLONIZATION_EVENT",
    "MORTALITY_EVENT",
    "RECOVERY_EVENT",
)
ECOLOGICAL_EVENTS = FIELD_EVENTS + ENTITY_EVENTS


class Twin2EEventError(ValueError):
    """An ecological event is malformed (missing parameter, wrong units, bad fraction)."""


def _frac(params: dict[str, Any], key: str, default: float) -> float:
    v = float(params.get(key, default))
    if not 0.0 <= v <= 1.0:
        raise Twin2EEventError(f"{key} must be in [0, 1], got {v}")
    return v


def _targets(ev: ScenarioEvent, entities: dict[UUID, EntityState]) -> list[EntityState]:
    if ev.target_entity_id is None:
        raise Twin2EEventError(f"{ev.event_type} needs target_entity_id")
    target = entities[ev.target_entity_id]
    if target.kind == KIND_HABITAT:
        return [e for e in entities.values() if e.parent_id == target.entity_id and e.kind in SESSILE]
    return [target]


def apply_event(
    ev: ScenarioEvent,
    t_s: float,
    fields: FieldEngine,
    entities: dict[UUID, EntityState],
    sw: MeifeSwitches,
) -> dict[str, Any]:
    rec: dict[str, Any] = {"event_id": str(ev.event_id), "event_type": ev.event_type, "applied_at_s": t_s}
    p = ev.parameters
    if ev.event_type not in ECOLOGICAL_EVENTS:
        return {**rec, "status": "NOT_ECOLOGICAL"}
    if not sw.disturbances:
        return {**rec, "status": "IGNORED_ABLATED_DISTURBANCES"}
    if ev.target_entity_id is not None and ev.target_entity_id not in entities:
        return {**rec, "status": "TARGET_NOT_ECOLOGICAL"}
    if ev.event_type in FIELD_EVENTS:
        if not sw.field_dynamics:
            return {**rec, "status": "IGNORED_STATIC_FIELDS"}
        _field_event(ev, t_s, fields, entities)
    elif ev.event_type == "RECOVERY_EVENT":
        if not sw.recovery:
            return {**rec, "status": "IGNORED_ABLATED_RECOVERY"}
        amount = _frac(p, "condition_gain", 0.3)
        for e in _targets(ev, entities):
            e.condition = min(1.0, e.condition + amount)
            e.thermal_stress = 0.0
    else:
        for e in _targets(ev, entities):
            if ev.event_type in ("PHYSICAL_DISTURBANCE", "HABITAT_DAMAGE"):
                f = _frac(p, "cover_loss_fraction", 0.5)
                e.cover *= 1.0 - f
                e.condition *= 1.0 - 0.5 * f
            elif ev.event_type == "MORTALITY_EVENT":
                e.cover *= 1.0 - _frac(p, "mortality_fraction", 0.5)
            elif ev.event_type == "COLONIZATION_EVENT":
                e.cover = min(1.0, e.cover + _frac(p, "cover_gain", 0.05))
    return {**rec, "status": "APPLIED", "parameters": dict(p)}


def _center(ev: ScenarioEvent, entities: dict[UUID, EntityState]) -> Any:
    if "center_m" in ev.parameters:
        return ev.parameters["center_m"]
    if ev.target_entity_id is not None:
        return [float(v) for v in entities[ev.target_entity_id].position_m]
    return None


def _field_event(
    ev: ScenarioEvent, t_s: float, fields: FieldEngine, entities: dict[UUID, EntityState]
) -> None:
    p = ev.parameters
    duration = float(p.get("duration_s", 86400.0))
    if duration <= 0:
        raise Twin2EEventError("duration_s must be positive")
    center = _center(ev, entities)
    radius = p.get("radius_m")
    if center is not None and radius is None:
        raise Twin2EEventError(f"{ev.event_type} with a centre needs radius_m")
    if ev.event_type == "TEMPERATURE_ANOMALY":
        canonical_units(str(p.get("units", "degC")), "degC")
        fields.forcing.temperature_anomalies.append(
            {
                "start_s": t_s,
                "end_s": t_s + duration,
                "delta_c": float(p["delta_c"]),
                "center_m": center,
                "radius_m": radius,
            }
        )
    elif ev.event_type == "TURBIDITY_SPIKE":
        canonical_units(str(p.get("units", "NTU")), "NTU")
        delta = float(p["delta_ntu"])
        if delta < 0:
            raise Twin2EEventError("delta_ntu must be non-negative")
        for st in (fields.regional, fields.local):
            st.turbidity[region_mask(st.grid, center, radius)] += delta
    elif ev.event_type == "FLOW_CHANGE":
        canonical_units(str(p.get("units", "")), "m s-1")
        vec = np.asarray(p["current_m_s"], dtype=np.float64)
        if vec.shape != (3,):
            raise Twin2EEventError("current_m_s must be a 3-vector")
        fields.p.current_base_m_s = vec
    elif ev.event_type == "LIGHT_REDUCTION":
        factor = _frac(p, "factor", 0.5)
        fields.forcing.light_reductions.append({"end_s": t_s + duration, "factor": factor})
