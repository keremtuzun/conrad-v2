"""Explicit cross-modal pairing graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PairRelation(str, Enum):
    SAME_WINDOW = "SAME_WINDOW"
    TEMPORAL_NEIGHBOR = "TEMPORAL_NEIGHBOR"
    UNPAIRED = "UNPAIRED"


@dataclass(frozen=True)
class PairEdge:
    left: str
    right: str
    relation: PairRelation
    dt_ms: int


@dataclass(frozen=True)
class PairGraph:
    nodes: tuple[str, ...]
    edges: tuple[PairEdge, ...]

    def coverage(self) -> dict[str, float]:
        possible = max(1, len(self.nodes) * (len(self.nodes) - 1) // 2)
        paired = sum(1 for e in self.edges if e.relation is not PairRelation.UNPAIRED)
        return {"pair_edges": float(paired), "pair_coverage": float(paired / possible)}

    def relation(self, left: str, right: str) -> PairRelation:
        wanted = {left, right}
        for edge in self.edges:
            if {edge.left, edge.right} == wanted:
                return edge.relation
        return PairRelation.UNPAIRED

