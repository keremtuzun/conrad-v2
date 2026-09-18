"""MCDE mechanism coupling graph G^mechanism (ch10 Mechanism coupling, Coupling graph, Cross-component).

Only couplings with a stated mechanistic justification exist. Cross-component couplings are restricted
to named structural relationship types; there is NO generic graph contagion (ADJACENT_TO carries nothing).

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from conrad.twins.twin2t.config import MCDEConfig
from conrad.twins.twin2t.registry import CouplingEdgeType, Mechanism, RelationshipType
from conrad.twins.twin2t.state import ComponentRuntime
from conrad.twins.twin2t.topology import LOAD_PATH_TYPES, SHARED_ENVIRONMENT_TYPES, StructuralGraph


@dataclass(frozen=True)
class CouplingEdge:
    source: Mechanism
    target: Mechanism
    edge_type: CouplingEdgeType
    scope: str
    """'INTRA' (same component) or 'CROSS' (via ``relationship_types``)."""
    relationship_types: frozenset[RelationshipType]
    justification: str


COUPLING_GRAPH: tuple[CouplingEdge, ...] = (
    CouplingEdge(
        Mechanism.CORROSION,
        Mechanism.FATIGUE,
        CouplingEdgeType.MODIFIES_GEOMETRY,
        "INTRA",
        frozenset(),
        "wall loss -> net section: membrane stress scales as t_nominal/(t_effective - d) "
        "(thin-wall hoop stress sigma = P*D/(2t)) -> higher dK in the Paris law",
    ),
    CouplingEdge(
        Mechanism.COATING_FAILURE,
        Mechanism.CORROSION,
        CouplingEdgeType.INCREASES_SUSCEPTIBILITY,
        "INTRA",
        frozenset(),
        "coating breakdown exposes bare metal: exposed fraction = coating breakdown factor (DNV-RP-B401)",
    ),
    CouplingEdge(
        Mechanism.FATIGUE,
        Mechanism.COATING_FAILURE,
        CouplingEdgeType.INCREASES_SUSCEPTIBILITY,
        "INTRA",
        frozenset(),
        "a through-coating crack breaches the coating over the cracked region, exposing metal to "
        "local corrosion (breach fraction = crack_length / length scale, ENGINEERING_ESTIMATE)",
    ),
    CouplingEdge(
        Mechanism.CORROSION,
        Mechanism.FATIGUE,
        CouplingEdgeType.MODIFIES_LOADING,
        "CROSS",
        LOAD_PATH_TYPES,
        "support wall loss reduces support stiffness -> load redistributes to the supported component "
        "(linear redistribution slope, ENGINEERING_ESTIMATE)",
    ),
    CouplingEdge(
        Mechanism.CORROSION,
        Mechanism.CORROSION,
        CouplingEdgeType.ACCELERATES,
        "CROSS",
        SHARED_ENVIRONMENT_TYPES,
        "shared environment: an environment change on a component applies to physically connected "
        "components in the same medium (no transfer of damage itself)",
    ),
)


def intra_edge_active(edge: CouplingEdge, rt: ComponentRuntime) -> bool:
    """Mechanism validity mask: an edge only acts when both mechanisms are valid for the component."""
    comp = rt.component
    return comp.mechanism_valid(edge.source) and comp.mechanism_valid(edge.target)


def export_coupling_graph() -> list[dict[str, object]]:
    return [
        {
            "source": e.source.value,
            "target": e.target.value,
            "type": e.edge_type.value,
            "scope": e.scope,
            "relationship_types": sorted(r.value for r in e.relationship_types),
            "justification": e.justification,
        }
        for e in COUPLING_GRAPH
    ]


def wall_loss_fraction(rt: ComponentRuntime) -> float:
    s = rt.state
    return min(1.0, max(s.corrosion_depth_m, s.crack_depth_m) / max(rt.effective_wall_m, 1e-9))


def update_mechanism_coupling(rt: ComponentRuntime, cfg: MCDEConfig) -> dict[str, float]:
    """Intra-component coupling. Sets the conditions the next tick's mechanisms see."""
    geo, coat_corr, crack_coat = COUPLING_GRAPH[0], COUPLING_GRAPH[1], COUPLING_GRAPH[2]
    del coat_corr  # consumed by the corrosion exposure rule in the engine
    rt.stress_concentration = 1.0
    if intra_edge_active(geo, rt):
        remaining = rt.effective_wall_m - rt.state.corrosion_depth_m
        scf = rt.params.wall_thickness_m / max(remaining, 1e-9)
        rt.stress_concentration = float(min(cfg.max_stress_concentration, max(scf, 0.0)))
    breach = 0.0
    if intra_edge_active(crack_coat, rt) and rt.params.coated:
        breach = min(1.0, rt.state.crack_length_m / cfg.crack_coating_length_scale_m)
        if breach > rt.state.coating_breakdown_fraction:
            rt.state = rt.state.with_(coating_breakdown_fraction=breach)
    return {"stress_concentration": rt.stress_concentration, "crack_coating_breach": breach}


def update_topology_coupling(
    graph: StructuralGraph, runtimes: dict[UUID, ComponentRuntime], cfg: MCDEConfig
) -> dict[UUID, float]:
    """Load-path coupling only along SUPPORTED_BY / LOAD_TRANSFER edges. Returns load multipliers."""
    edge = COUPLING_GRAPH[3]
    out: dict[UUID, float] = {}
    for eid in graph.component_ids:
        rt = runtimes.get(eid)
        if rt is None:
            continue
        loss = 0.0
        if rt.component.mechanism_valid(edge.target):
            for sup in graph.load_supporters(eid):
                srt = runtimes.get(sup)
                if srt is not None and srt.component.mechanism_valid(edge.source):
                    loss = max(loss, wall_loss_fraction(srt))
        rt.load_multiplier = 1.0 + cfg.support_load_redistribution * loss
        out[eid] = rt.load_multiplier
    return out


def shared_environment_targets(graph: StructuralGraph, eid: UUID) -> tuple[UUID, ...]:
    return graph.neighbours(eid, COUPLING_GRAPH[4].relationship_types)
