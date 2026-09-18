"""Hand-built Twin2T runtimes with fixed (non-sampled) hidden parameters, for exact mechanism tests."""

from __future__ import annotations

from uuid import UUID

from conrad.schemas.ids import IdFactory
from conrad.twins.twin2t.config import MCDEConfig
from conrad.twins.twin2t.mcde import MCDE
from conrad.twins.twin2t.priors import ENGINEERING_ESTIMATE, SampledParameter
from conrad.twins.twin2t.registry import RelationshipType, default_component_types, default_materials
from conrad.twins.twin2t.state import (
    ComponentParameters,
    ComponentRuntime,
    DegradationState,
    Environment,
    Loading,
    validity_for,
)
from conrad.twins.twin2t.topology import StructuralComponent, StructuralGraph, StructuralRelationship

FIXED = {
    "corrosion_A": 1.0e-4,
    "corrosion_n": 0.6,
    "corrosion_r0": 1.0e-4,
    "corrosion_rs": 5.0e-5,
    "corrosion_t_transition_yr": 2.0,
    "temperature_q10": 1.5,
    "coating_a": 0.05,
    "coating_b_per_yr": 0.02,
    "cp_residual_factor": 0.05,
    "burial_factor": 0.5,
    "paris_C": 1.0e-11,
    "paris_m": 3.0,
    "delta_k_threshold": 3.0,
    "geometry_factor_Y": 1.12,
    "crack_aspect_ratio": 4.0,
    "sigma_corrosion": 2.0e-5,
    "sigma_fatigue": 5.0e-5,
}
YEAR = 365.25 * 86400.0


def params(wall: float = 0.012, coated: bool = False, **over: float) -> ComponentParameters:
    vals = {**FIXED, **over}
    return ComponentParameters(
        wall, coated, {k: SampledParameter(k, v, "-", ENGINEERING_ESTIMATE, "test") for k, v in vals.items()}
    )


def runtime(
    eid: UUID,
    ctype: str = "SEGMENT",
    material: str = "carbon_steel",
    *,
    coated: bool = False,
    corrosion: float = 0.0,
    crack: float = 0.0,
    coating: float = 0.0,
    stress: float = 60e6,
    cycles: float = 0.1,
    env: Environment | None = None,
    **over: float,
) -> ComponentRuntime:
    comp = StructuralComponent(
        eid, default_component_types().get(ctype), None, None, default_materials().get(material)
    )
    v = validity_for(comp)
    p = params(coated=coated, **over)
    st = DegradationState(
        corrosion_depth_m=corrosion if v["corrosion_depth_m"] else 0.0,
        coating_breakdown_fraction=coating if v["coating_breakdown_fraction"] else 0.0,
        crack_length_m=crack if v["crack_length_m"] else 0.0,
        crack_depth_m=(crack / 4.0) if v["crack_length_m"] else 0.0,
        validity=v,
    )
    return ComponentRuntime(comp, p, st, env or Environment(), Loading(cycles, stress), initial_state=st)


def engine(
    rts: list[ComponentRuntime],
    rels: list[tuple[UUID, UUID, RelationshipType]] = (),  # type: ignore[assignment]
    cfg: MCDEConfig | None = None,
    seed: int = 1,
) -> MCDE:
    graph = StructuralGraph(
        [rt.component for rt in rts], [StructuralRelationship(a, b, t) for a, b, t in rels]
    )
    return MCDE(graph, {rt.component.entity_id: rt for rt in rts}, cfg or MCDEConfig(), seed)


def new_ids(n: int, seed: int = 5) -> list[UUID]:
    f = IdFactory(seed=seed)
    return [f.new() for _ in range(n)]
