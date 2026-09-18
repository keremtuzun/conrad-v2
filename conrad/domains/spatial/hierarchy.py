"""Hierarchical sparse map: global blocks / regional base grid / local refined cells. BELIEF PLANE.

- REGIONAL (persistent): the base grid (default 0.25 m). EVERY observation updates it.
- LOCAL (working memory): refined cells (0.125 m, 0.0625 m) inside refined parents near the robot.
  They are extra detail, never the only copy of evidence, so dropping them (coarsening or
  ``reset_working_memory``) preserves coverage, uncertainty and provenance in the regional level.
- GLOBAL: blocks of ``global_block_cells**3`` base cells; summaries are derived on demand.

Refinement is a deterministic policy (ch33): coverage >= min AND (geometry discontinuity OR uncertainty
gradient above threshold), within ``local_radius_m`` of the current focus, under a per-level budget.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from conrad.domains.spatial.config import SpatialConfig
from conrad.domains.spatial.grid import INFERRED, OBSERVED, UNKNOWN, SparseLevel
from conrad.domains.spatial.integrate import CellMasses, MassFn
from conrad.domains.spatial.keys import (
    FACE_OFFSETS,
    FloatArr,
    IntArr,
    children_index,
    decode,
    encode,
    index_to_center,
    points_to_codes,
)


@dataclass(frozen=True)
class PointState:
    probability: FloatArr
    status: NDArray[np.int8]
    uncertainty: FloatArr  # (N, 4) UA, UE, UC, UO
    coverage: FloatArr
    variance: FloatArr
    level: IntArr  # -1 = no cell anywhere (never touched)


class HierarchicalMap:
    def __init__(self, cfg: SpatialConfig) -> None:
        self.cfg = cfg
        self.resolutions = (cfg.grid.base_voxel_m, *cfg.grid.refinement_voxels_m)
        self.levels = [SparseLevel(r, cfg) for r in self.resolutions]
        self.refined: list[set[int]] = [set() for _ in self.resolutions]
        self.blocks: dict[int, list[int]] = {}
        self.focus: FloatArr | None = None

    @property
    def base(self) -> SparseLevel:
        return self.levels[0]

    def block_code(self, base_codes: IntArr) -> IntArr:
        return encode(np.floor_divide(decode(base_codes), self.cfg.grid.global_block_cells))

    def block_bounds(self, block: int) -> tuple[FloatArr, FloatArr]:
        size = self.cfg.grid.global_block_cells * self.resolutions[0]
        lo = decode(np.array([block]))[0].astype(np.float64) * size
        return lo, lo + size

    # ------------------------------------------------------------------ update
    def base_rows(self, codes: IntArr) -> IntArr:
        """Rows in the base level, creating cells (and registering them in their global block)."""
        lvl = self.base
        before = lvl.n
        rows = lvl.rows(codes, create=True)
        new = rows >= before
        if new.any():
            for b, r in zip(self.block_code(codes[new]).tolist(), rows[new].tolist(), strict=True):
                self.blocks.setdefault(b, []).append(r)
        return rows

    def integrate(self, masses_at: MassFn, evidence_id: UUID, t_ns: int) -> IntArr:
        """Apply one observation at every level. Returns the base codes it touched.

        A level whose kernel cannot represent the observation's pose blur receives the coarser level's
        masses on its children (the evidence supports no finer detail than that)."""
        touched = np.zeros(0, dtype=np.int64)
        prev: CellMasses | None = None
        for li, lvl in enumerate(self.levels):
            if li > 0 and not self.refined[li - 1]:
                break
            cm = masses_at(lvl.res, li > 0)
            if cm is None:
                if prev is None:
                    raise ValueError("the base level must always produce masses")
                kids = children_index(decode(prev.codes))
                rep = np.repeat(np.arange(len(prev)), 8)
                cm = CellMasses(
                    encode(kids), prev.hit[rep], prev.miss[rep], prev.ua[rep], prev.ue[rep], prev.sigma[rep]
                )
            prev = cm
            if li > 0 and len(cm):
                parents = encode(np.floor_divide(decode(cm.codes), 2))
                cm = cm.select(np.isin(parents, np.fromiter(self.refined[li - 1], dtype=np.int64)))
            if not len(cm):
                continue
            rows = self.base_rows(cm.codes) if li == 0 else lvl.rows(cm.codes, create=True)
            lvl.apply_direct(rows, cm.hit, cm.miss, cm.ua, cm.ue, cm.sigma, evidence_id, t_ns)
            if li == 0:
                touched = cm.codes
        return touched

    # ------------------------------------------------------------------ queries
    def point_state(self, points: FloatArr) -> PointState:
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        n = len(pts)
        p, cov, var = np.full(n, 0.5), np.zeros(n), np.full(n, 0.25)
        st = np.full(n, UNKNOWN, dtype=np.int8)
        unc = np.tile(np.array([0.0, 1.0, 0.0, 1.0]), (n, 1))
        level = np.full(n, -1, dtype=np.int64)
        for li in range(len(self.levels) - 1, -1, -1):
            lvl = self.levels[li]
            if lvl.n == 0:
                continue
            todo = np.nonzero(level < 0)[0]
            if todo.size == 0:
                break
            rows = lvl.rows(points_to_codes(pts[todo], lvl.res))
            hit = rows >= 0
            sel, r = todo[hit], rows[hit]
            p[sel], st[sel], unc[sel] = lvl.probability(r), lvl.status(r), lvl.uncertainty(r)
            cov[sel], var[sel], level[sel] = lvl.coverage(r), lvl.variance(r), li
        return PointState(p, st, unc, cov, var, level)

    # ------------------------------------------------------------------ refinement
    def _neighbour_rows(self, lvl: SparseLevel, codes: IntArr) -> IntArr:
        idx = decode(codes)
        nb = (idx[:, None, :] + FACE_OFFSETS[None, :, :]).reshape(-1, 3)
        return lvl.rows(encode(nb)).reshape(len(codes), len(FACE_OFFSETS))

    def refinement_scores(self, li: int, rows: IntArr) -> tuple[FloatArr, FloatArr]:
        """(geometry discontinuity, uncertainty gradient) per row, over the 6 face neighbours."""
        lvl = self.levels[li]
        nb = self._neighbour_rows(lvl, lvl.codes[rows])
        p, st = lvl.probability(rows), lvl.status(rows)
        uo = lvl.uncertainty(rows)[:, 3]
        flat = nb.reshape(-1)
        have = flat >= 0
        np_, nst, nuo = np.full(flat.shape, 0.5), np.full(flat.shape, UNKNOWN), np.ones(flat.shape)
        np_[have] = lvl.probability(flat[have])
        nst[have] = lvl.status(flat[have])
        nuo[have] = lvl.uncertainty(flat[have])[:, 3]
        known = (nst != UNKNOWN).reshape(nb.shape) & (st != UNKNOWN)[:, None]
        disc = np.where(known, np.abs(np_.reshape(nb.shape) - p[:, None]), 0.0).max(axis=1)
        grad = np.abs(nuo.reshape(nb.shape) - uo[:, None]).max(axis=1)
        return disc, grad

    def refine(self, focus: FloatArr | None) -> dict[int, int]:
        """Deterministic refinement pass. Returns the number of newly refined cells per level."""
        rc = self.cfg.refinement
        added: dict[int, int] = {}
        if not rc.enabled or focus is None:
            return added
        self.focus = np.asarray(focus, dtype=np.float64)
        for li in range(len(self.levels) - 1):
            lvl = self.levels[li]
            budget = rc.max_refined_cells_per_level - len(self.refined[li])
            if lvl.n == 0 or budget <= 0:
                continue
            rows = lvl.all_rows()
            centers = index_to_center(decode(lvl.codes[rows]), lvl.res)
            near = np.linalg.norm(centers - self.focus, axis=1) <= rc.local_radius_m
            near &= ~np.isin(lvl.codes[rows], np.fromiter(self.refined[li], dtype=np.int64))
            near &= lvl.coverage(rows) >= rc.min_coverage
            near &= np.isin(lvl.status(rows), (OBSERVED, INFERRED))
            child_res = self.resolutions[li + 1]
            near &= lvl.mean_pose_sigma(rows) * self.cfg.pose.splat_sigma_k <= child_res * max(
                self.cfg.pose.max_splat_radius_cells, 1
            )
            cand = rows[near]
            if cand.size == 0:
                continue
            disc, grad = self.refinement_scores(li, cand)
            score = np.maximum(
                disc / max(rc.discontinuity_threshold, 1e-9),
                grad / max(rc.uncertainty_gradient_threshold, 1e-9),
            )
            ok = score >= 1.0
            cand, score = cand[ok], score[ok]
            order = np.lexsort((lvl.codes[cand], -score))[:budget]
            for r in cand[order].tolist():
                self._split(li, r)
            added[li] = len(order)
        return added

    def _split(self, li: int, row: int) -> None:
        """Split a cell into 8 children one level finer; children inherit the parent state."""
        parent, child = self.levels[li], self.levels[li + 1]
        code = int(parent.codes[row])
        kids = child.rows(encode(children_index(decode(np.array([code])))), create=True)
        for k, v in parent.f.items():
            child.f[k][kids] = v[row]
        for k, bv in parent.b.items():
            child.b[k][kids] = bv[row]
        child.n_obs[kids], child.last_ns[kids], child.src[kids] = (
            parent.n_obs[row],
            parent.last_ns[row],
            parent.src[row],
        )
        for kr in kids.tolist():
            child.provenance[kr] = list(parent.provenance.get(row, []))
            if row in parent.semantic:
                child.semantic[kr] = parent.semantic[row]
        self.refined[li].add(code)

    def clear_local(self) -> None:
        """Drop the local (refined) working memory. The regional level is untouched."""
        for li in range(1, len(self.levels)):
            self.levels[li].clear()
        self.refined = [set() for _ in self.resolutions]
        self.focus = None
