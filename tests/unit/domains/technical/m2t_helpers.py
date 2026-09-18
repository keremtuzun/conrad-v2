"""Hand-built asset registry + evidence for Model2T unit tests (no Twin2T involvement)."""

from __future__ import annotations

from uuid import UUID

from conrad.domains.technical import Model2T, Model2TConfig
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import EntityCandidate, Evidence, Modality, QualityContext
from conrad.schemas.timebase import stamp

DAY = 86400.0
NAMES = ("asset", "seg_a", "seg_b", "weld_a", "support_steel", "support_concrete", "seg_adj")


def registry_ids(seed: int = 11) -> dict[str, UUID]:
    f = IdFactory(seed=seed, namespace="registry")
    return {n: f.new() for n in NAMES}


def context(r: dict[str, UUID]) -> dict:
    comps = [
        {"registry_id": r["asset"], "component_type": "ASSET"},
        {
            "registry_id": r["seg_a"],
            "component_type": "SEGMENT",
            "material": "carbon_steel",
            "wall_thickness_m": 0.015,
        },
        {
            "registry_id": r["seg_b"],
            "component_type": "SEGMENT",
            "material": "carbon_steel",
            "wall_thickness_m": 0.015,
        },
        {"registry_id": r["weld_a"], "component_type": "WELD", "material": "carbon_steel"},
        {"registry_id": r["support_steel"], "component_type": "SUPPORT", "material": "carbon_steel"},
        {"registry_id": r["support_concrete"], "component_type": "SUPPORT", "material": "concrete"},
        {"registry_id": r["seg_adj"], "component_type": "SEGMENT", "material": "carbon_steel"},
    ]
    rels = [
        {"source": r["seg_a"], "target": r["seg_b"], "type": "CONNECTED_TO"},
        {"source": r["weld_a"], "target": r["seg_a"], "type": "ATTACHED_TO"},
        {"source": r["seg_a"], "target": r["support_steel"], "type": "SUPPORTED_BY"},
        {"source": r["seg_b"], "target": r["support_concrete"], "type": "SUPPORTED_BY"},
        {"source": r["seg_a"], "target": r["seg_adj"], "type": "ADJACENT_TO"},
        {"source": r["seg_a"], "target": r["asset"], "type": "PART_OF"},
    ]
    return {"asset_registry": {"components": comps, "relationships": rels}}


def model(
    seed: int = 3, config: Model2TConfig | None = None, **kw
) -> tuple[Model2T, dict[str, UUID], IdFactory]:
    ids = IdFactory(seed=seed, namespace="model2t")
    r = registry_ids()
    m = Model2T(ids, config, **kw)
    m.initialize(context(r))
    return m, r, IdFactory(seed=seed, namespace="evidence")


def evidence(
    ids: IdFactory,
    target: UUID,
    t_s: float,
    *,
    wall: float | None = None,
    crack: float | None = None,
    anomaly: float | None = None,
    rel: float = 0.9,
    ua: float = 0.1,
    group: str | None = None,
    run_id: UUID | None = None,
) -> Evidence:
    meas, units = {}, {}
    for name, val, unit in (
        ("apparent_wall_loss", wall, "m"),
        ("crack_indication_length", crack, "m"),
        ("surface_anomaly_score", anomaly, "1"),
    ):
        if val is not None:
            meas[name], units[name] = val, unit
    ts = stamp(t_s, "sim")
    return Evidence(
        evidence_id=ids.new(),
        source_observation_id=ids.new(),
        mission_id=ids.new(),
        run_id=run_id or ids.new(),
        trace_id=ids.new(),
        modality=Modality.STRUCTURED,
        timestamp=ts,
        created_time_ns=ts.time_ns,
        embedding=(1.0,),
        entity_candidates=(EntityCandidate(registry_entity_id=target, score=1.0),),
        reliability=rel,
        aleatoric_uncertainty=ua,
        sensor_context=QualityContext(),
        measurements=meas,
        measurement_units=units,
        independence_group=group,
        provenance_id=ids.new(),
        encoder_version="test",
    )


def claim(msg_or_cell, name: str):
    claims = getattr(msg_or_cell, "state_summary", None) or msg_or_cell.claims
    return next(c for c in claims if c.name == name)
