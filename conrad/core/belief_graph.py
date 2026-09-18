"""Working belief graph: the small fast-access tensor set Model 2 computes over (ch3, ch9).

Working memory is a cache. Clearing it never touches the persistent store (CC-09).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import torch
from torch import Tensor

from conrad.core.config import CoreConfig
from conrad.core.state import AnalyticBeliefState
from conrad.schemas.belief import BeliefCell, Relationship


@dataclass(frozen=True)
class BeliefGraphBatch:
    """Z [N,Dz], U [N,Du], T [N,Dt], H [N,Dh], edge_index [2,M] (row 0 = source j, row 1 = target i)."""

    z: Tensor
    u: Tensor
    temporal: Tensor
    history: Tensor
    edge_index: Tensor
    edge_type: Tensor
    edge_numeric: Tensor
    node_mask: Tensor
    belief_ids: tuple[UUID, ...]
    relationship_ids: tuple[UUID, ...]


@dataclass
class WorkingNode:
    cell: BeliefCell
    z: Tensor
    u: Tensor
    temporal: Tensor
    history: Tensor
    corrected: AnalyticBeliefState
    predicted: AnalyticBeliefState | None = None  # B^- kept apart from the corrected B^+
    last_surprise: float = 0.0


def _fit(values: tuple[float, ...], dim: int) -> Tensor:
    if len(values) == dim:
        return torch.tensor(values, dtype=torch.float32)
    if len(values) == 0:
        return torch.zeros(dim)
    raise ValueError(f"persisted tensor has width {len(values)}, config expects {dim}")


@dataclass
class WorkingBeliefGraph:
    cfg: CoreConfig
    nodes: dict[UUID, WorkingNode] = field(default_factory=dict)
    relationships: dict[UUID, Relationship] = field(default_factory=dict)
    relation_type_ids: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, belief_id: UUID) -> bool:
        return belief_id in self.nodes

    def load(self, cell: BeliefCell, state: AnalyticBeliefState, u: Tensor | None = None) -> WorkingNode:
        """Bring a persisted belief into working memory. Tensors come from the BELIEF, never from evidence."""
        node = WorkingNode(
            cell=cell,
            z=_fit(cell.state_embedding, self.cfg.belief_dim),
            u=torch.zeros(self.cfg.uncertainty_dim) if u is None else u,
            temporal=_fit(cell.temporal_state, self.cfg.temporal_dim),
            history=_fit(cell.history_state, self.cfg.history_dim),
            corrected=state,
        )
        self.nodes[cell.belief_id] = node
        return node

    def evict(self, belief_id: UUID) -> None:
        self.nodes.pop(belief_id, None)
        for rid in [
            r
            for r, rel in self.relationships.items()
            if belief_id in (rel.source_belief_id, rel.target_belief_id)
        ]:
            del self.relationships[rid]

    def add_relationship(self, rel: Relationship) -> None:
        self.relationships[rel.relationship_id] = rel

    def relation_type_id(self, relation_type: str) -> int:
        """Stable id per type inside this graph; ids wrap into the configured embedding table."""
        if relation_type not in self.relation_type_ids:
            self.relation_type_ids[relation_type] = len(self.relation_type_ids) % self.cfg.rbp.relation_types
        return self.relation_type_ids[relation_type]

    def clear(self) -> None:
        self.nodes.clear()
        self.relationships.clear()

    def batch(self) -> BeliefGraphBatch:
        ids = tuple(sorted(self.nodes, key=lambda b: b.int))
        index = {b: i for i, b in enumerate(ids)}
        c = self.cfg

        def stack(attr: str, dim: int) -> Tensor:
            return torch.stack([getattr(self.nodes[b], attr) for b in ids]) if ids else torch.zeros(0, dim)

        rels = [
            r
            for r in sorted(self.relationships.values(), key=lambda r: r.relationship_id.int)
            if r.source_belief_id in index and r.target_belief_id in index
        ]
        edge_index = torch.tensor(
            [[index[r.source_belief_id] for r in rels], [index[r.target_belief_id] for r in rels]],
            dtype=torch.long,
        ).reshape(2, len(rels))
        numeric = torch.zeros(len(rels), c.rbp.relation_numeric_dim)
        for m, r in enumerate(rels):
            attrs = r.attributes
            row = [
                attrs.get("dx_m", 0.0),
                attrs.get("dy_m", 0.0),
                attrs.get("dz_m", 0.0),
                attrs.get("distance_m", 0.0),
                attrs.get("age_s", 0.0),
                r.confidence,
            ]
            numeric[m] = torch.tensor(
                (row + [0.0] * c.rbp.relation_numeric_dim)[: c.rbp.relation_numeric_dim]
            )
        return BeliefGraphBatch(
            z=stack("z", c.belief_dim),
            u=stack("u", c.uncertainty_dim),
            temporal=stack("temporal", c.temporal_dim),
            history=stack("history", c.history_dim),
            edge_index=edge_index,
            edge_type=torch.tensor([self.relation_type_id(r.relation_type) for r in rels], dtype=torch.long),
            edge_numeric=numeric,
            node_mask=torch.ones(len(ids), dtype=torch.bool),
            belief_ids=ids,
            relationship_ids=tuple(r.relationship_id for r in rels),
        )
