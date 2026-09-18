"""Array export of the sparse hierarchical map (for navigation, MCBR, evaluation and replay). BELIEF PLANE.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from conrad.domains.spatial.grid import STATUS_NAMES, SparseLevel
from conrad.domains.spatial.hierarchy import HierarchicalMap
from conrad.domains.spatial.keys import FloatArr, IntArr, decode, index_to_center
from conrad.domains.spatial.temporal import age_s


@dataclass(frozen=True)
class LevelExport:
    level: int
    resolution_m: float
    indices: IntArr  # (N, 3)
    centers_m: FloatArr  # (N, 3) map frame
    logodds: FloatArr
    probability: FloatArr
    variance: FloatArr
    status: NDArray[np.str_]
    coverage: FloatArr
    uncertainty: FloatArr  # (N, 4) UA, UE, UC, UO
    observation_count: IntArr
    change_score: FloatArr
    dynamic: NDArray[np.bool_]
    last_update_ns: IntArr
    age_s: FloatArr
    semantic_class: NDArray[np.object_]
    refined: NDArray[np.bool_]


@dataclass(frozen=True)
class GridExport:
    frame_id: str
    levels: tuple[LevelExport, ...]
    global_block_cells: int

    def level(self, i: int) -> LevelExport:
        return self.levels[i]


def export_level(hmap: HierarchicalMap, li: int, now_ns: int) -> LevelExport:
    lvl: SparseLevel = hmap.levels[li]
    rows = lvl.all_rows()
    codes = lvl.codes[rows]
    idx = decode(codes)
    refined = np.isin(codes, np.fromiter(hmap.refined[li], dtype=np.int64))
    return LevelExport(
        level=li,
        resolution_m=lvl.res,
        indices=idx,
        centers_m=index_to_center(idx, lvl.res),
        logodds=lvl.f["lo"][rows].copy(),
        probability=lvl.probability(rows),
        variance=lvl.variance(rows),
        status=np.array([STATUS_NAMES[s] for s in lvl.status(rows).tolist()], dtype=str),
        coverage=lvl.coverage(rows),
        uncertainty=lvl.uncertainty(rows),
        observation_count=lvl.n_obs[rows].copy(),
        change_score=lvl.f["change"][rows].copy(),
        dynamic=lvl.b["dynamic"][rows].copy(),
        last_update_ns=lvl.last_ns[rows].copy(),
        age_s=age_s(lvl, rows, now_ns),
        semantic_class=np.array([lvl.semantic.get(r) for r in rows.tolist()], dtype=object),
        refined=refined,
    )


def export_map(hmap: HierarchicalMap, now_ns: int) -> GridExport:
    return GridExport(
        frame_id=hmap.cfg.grid.frame_id,
        levels=tuple(export_level(hmap, li, now_ns) for li in range(len(hmap.levels))),
        global_block_cells=hmap.cfg.grid.global_block_cells,
    )
