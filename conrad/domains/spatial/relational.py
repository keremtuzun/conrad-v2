"""Relational (neighbour) inference on the regional grid: bridge SMALL gaps between observed occupied cells.

A cell is INFERRED occupied only when it lies on a lattice line strictly BETWEEN two OBSERVED occupied
cells at most ``max_gap_cells`` apart (interpolation, never extrapolation), carries no appreciable
free-space evidence and has no direct OBSERVED status. Inferred cells get a sub-confident occupancy,
raised U_O and RELATIONAL_INFERENCE provenance listing the supporting evidence. One pass only: inferred
cells never support further inference, so large unobserved regions and the hidden side of a structure
(which is not between two observed cells) stay UNKNOWN.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

import numpy as np

from conrad.domains.spatial.config import SpatialConfig
from conrad.domains.spatial.grid import OBSERVED, SRC_DIRECT, SRC_NONE, SRC_RELATIONAL, SparseLevel
from conrad.domains.spatial.keys import LINE_DIRECTIONS, IntArr, decode, encode


def clear_inferred(level: SparseLevel) -> IntArr:
    """Remove every previous inference (it is recomputed from current evidence). Returns their codes."""
    rows = np.nonzero(level.b["inferred"][: level.n])[0]
    level.b["inferred"][rows] = False
    level.f["inf_p"][rows] = 0.0
    level.f["inf_uo"][rows] = 0.0
    level.f["inf_ue"][rows] = 0.0
    level.src[rows] = np.where(level.f["w"][rows] > 0, SRC_DIRECT, SRC_NONE).astype(np.int8)
    return np.asarray(level.codes[rows], dtype=np.int64)


def gap_candidates(support_codes: IntArr, max_gap: int) -> dict[int, tuple[int, int]]:
    """Gap cell code -> (start, end) support codes, first pair found in a deterministic scan order."""
    out: dict[int, tuple[int, int]] = {}
    if support_codes.size == 0:
        return out
    idx = decode(support_codes)
    for d in LINE_DIRECTIONS:
        for gap in range(1, max_gap + 1):
            end = encode(idx + (gap + 1) * d)
            ok = np.isin(end, support_codes)
            if not ok.any():
                continue
            starts, ends = support_codes[ok], end[ok]
            for s in range(1, gap + 1):
                cells = encode(idx[ok] + s * d)
                for c, a, b in zip(cells.tolist(), starts.tolist(), ends.tolist(), strict=True):
                    out.setdefault(c, (a, b))
    return out


def relational_fill(
    level: SparseLevel, cfg: SpatialConfig, create_rows: Callable[[IntArr], IntArr]
) -> tuple[IntArr, dict[int, list[UUID]]]:
    """Recompute inferred cells. Returns (codes whose inference changed, cell code -> support evidence)."""
    rc = cfg.relational
    old = clear_inferred(level)
    if not rc.enabled or level.n == 0:
        return old, {}
    rows = level.all_rows()
    p, st = level.probability(rows), level.status(rows)
    support = np.sort(level.codes[rows][(st == OBSERVED) & (p >= rc.min_support_probability)])
    cands = gap_candidates(support, rc.max_gap_cells)
    if not cands:
        return old, {}
    codes = np.fromiter(cands.keys(), dtype=np.int64, count=len(cands))
    existing = level.rows(codes)
    ok = np.ones(len(codes), dtype=bool)
    have = existing >= 0
    er = existing[have]
    free_mass = level.f["w"][er] - level.f["w_hit"][er]
    ok[have] = (level.status(er) != OBSERVED) & (free_mass <= rc.max_free_mass) & ~level.b["predicted"][er]
    codes = codes[ok]
    if codes.size == 0:
        return old, {}
    new_rows = create_rows(codes)
    sup_rows = dict(zip(support.tolist(), level.rows(support).tolist(), strict=True))
    ue_all = level.uncertainty(level.rows(support))[:, 1]
    ue_by_code = dict(zip(support.tolist(), ue_all.tolist(), strict=True))
    refs: dict[int, list[UUID]] = {}
    for c, r in zip(codes.tolist(), new_rows.tolist(), strict=True):
        a, b = cands[c]
        ids: list[UUID] = []
        for s in (a, b):
            for e in level.provenance.get(sup_rows[s], []):
                if e not in ids:
                    ids.append(e)
        refs[c] = ids[-rc.max_support_refs :]
        level.f["inf_ue"][r] = 0.5 * (ue_by_code[a] + ue_by_code[b])
        level.add_provenance(np.array([r]), refs[c])
    level.b["inferred"][new_rows] = True
    level.f["inf_p"][new_rows] = rc.inferred_occupancy
    level.f["inf_uo"][new_rows] = rc.inferred_observational_uncertainty
    level.src[new_rows] = (
        SRC_RELATIONAL  # the reported value comes from inference, whatever partial mass exists
    )
    return np.union1d(old, codes), refs
