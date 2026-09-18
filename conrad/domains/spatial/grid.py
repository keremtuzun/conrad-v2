"""One resolution level of the sparse voxel map: struct-of-arrays cells keyed by int64 codes. BELIEF PLANE.

Per cell: direct-evidence log-odds and mass, observation count, weighted measurement (UA) and pose (UE)
uncertainty, change score (UC), dynamic flag, last-update time, relational/temporal state, optional
semantic class and bounded provenance evidence IDs. Status, probability, coverage, variance and the
four uncertainty channels are DERIVED from that state, so they can never disagree with it.

Channel choice (documented): UA = measurement noise; UE = pose-induced spread (localisation ignorance is
reducible, hence epistemic); UC = contradicting re-observation; UO = 1 - coverage (+ inference/prediction).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from conrad.domains.spatial.config import SpatialConfig
from conrad.domains.spatial.keys import FloatArr, IntArr

UNKNOWN, OBSERVED, INFERRED, PREDICTED = 0, 1, 2, 3
STATUS_NAMES = ("UNKNOWN", "OBSERVED", "INFERRED", "PREDICTED")
SRC_NONE, SRC_DIRECT, SRC_RELATIONAL, SRC_TEMPORAL = 0, 1, 2, 3

_FLOAT_FIELDS = (
    "lo",
    "w",
    "w_hit",
    "ua_acc",
    "ue_acc",
    "sig_acc",
    "change",
    "inf_p",
    "inf_uo",
    "inf_ue",
    "pred_uo",
)
_BOOL_FIELDS = ("dynamic", "inferred", "predicted")


def sigmoid(x: FloatArr) -> FloatArr:
    return np.asarray(1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64))), dtype=np.float64)


class SparseLevel:
    def __init__(self, res: float, cfg: SpatialConfig, capacity: int = 1024) -> None:
        self.res = res
        self.cfg = cfg
        self._reset(capacity)

    def _reset(self, capacity: int) -> None:
        self._index: dict[int, int] = {}
        self.n = 0
        self.codes: IntArr = np.zeros(capacity, dtype=np.int64)
        self.f: dict[str, FloatArr] = {k: np.zeros(capacity) for k in _FLOAT_FIELDS}
        self.b: dict[str, NDArray[np.bool_]] = {k: np.zeros(capacity, dtype=bool) for k in _BOOL_FIELDS}
        self.n_obs = np.zeros(capacity, dtype=np.int64)
        self.last_ns = np.zeros(capacity, dtype=np.int64)
        self.src = np.zeros(capacity, dtype=np.int8)
        self.semantic: dict[int, str] = {}
        self.provenance: dict[int, list[UUID]] = {}

    def __len__(self) -> int:
        return self.n

    # ------------------------------------------------------------------ storage
    def _grow(self, need: int) -> None:
        cap = len(self.codes)
        if need <= cap:
            return
        new = max(need, 2 * cap)
        self.codes = np.resize(self.codes, new)
        for d in (self.f, self.b):
            for k, v in d.items():
                grown = np.zeros(new, dtype=v.dtype)
                grown[:cap] = v
                d[k] = grown  # type: ignore[assignment]
        for name in ("n_obs", "last_ns", "src"):
            old = getattr(self, name)
            grown_i = np.zeros(new, dtype=old.dtype)
            grown_i[:cap] = old
            setattr(self, name, grown_i)

    def rows(self, codes: IntArr, create: bool = False) -> IntArr:
        """Row per code (-1 when absent and ``create`` is False). Codes should be unique when creating."""
        out = np.fromiter((self._index.get(c, -1) for c in codes.tolist()), dtype=np.int64, count=len(codes))
        if create:
            miss = np.nonzero(out < 0)[0]
            if miss.size:
                self._grow(self.n + miss.size)
                new_rows = np.arange(self.n, self.n + miss.size, dtype=np.int64)
                for c, r in zip(codes[miss].tolist(), new_rows.tolist(), strict=True):
                    self._index[c] = r
                self.codes[new_rows] = codes[miss]
                out[miss] = new_rows
                self.n += miss.size
        return out

    def all_rows(self) -> IntArr:
        return np.arange(self.n, dtype=np.int64)

    def clear(self) -> None:
        self._reset(1024)

    def add_provenance(self, rows: IntArr, ids: list[UUID]) -> None:
        cap = self.cfg.messages.max_cell_provenance
        for r in rows.tolist():
            lst = self.provenance.setdefault(r, [])
            for i in ids:
                if i not in lst:
                    lst.append(i)
            del lst[: max(len(lst) - cap, 0)]  # keep the most recent references

    # ------------------------------------------------------------------ derived state
    def status(self, rows: IntArr) -> NDArray[np.int8]:
        oc = self.cfg.occupancy
        w = self.f["w"][rows]
        observed = w >= oc.observed_min_weight if oc.knowledge_status else w > 0
        observed &= w > 0
        st = np.full(len(rows), UNKNOWN, dtype=np.int8)
        st[self.b["inferred"][rows] & ~observed] = INFERRED
        st[observed] = OBSERVED
        st[self.b["predicted"][rows]] = PREDICTED
        return st

    def mean_pose_sigma(self, rows: IntArr) -> FloatArr:
        w = self.f["w"][rows]
        return np.where(w > 0, self.f["sig_acc"][rows] / np.maximum(w, 1e-12), 0.0)

    def logodds_cap(self, rows: IntArr) -> FloatArr:
        """Upper bound on occupied-side log-odds: l_max / (1 + (sigma_pose/res)^2)."""
        oc = self.cfg.occupancy
        if not self.cfg.pose.confidence_cap:
            return np.full(len(rows), max(oc.l_max, -oc.l_min))
        rho = self.mean_pose_sigma(rows) / self.res
        return np.asarray(max(oc.l_max, -oc.l_min) / (1.0 + rho**2), dtype=np.float64)

    def probability(self, rows: IntArr) -> FloatArr:
        st = self.status(rows)
        cap = self.logodds_cap(
            rows
        )  # a return cannot become a confident OCCUPIED cell under large pose sigma
        lo = np.minimum(self.f["lo"][rows], cap)
        p = np.where(self.f["w"][rows] > 0, sigmoid(lo), 0.5)
        p = np.where(st == INFERRED, self.f["inf_p"][rows], p)
        return np.asarray(np.where(st == PREDICTED, sigmoid(self.f["lo"][rows]), p), dtype=np.float64)

    def coverage(self, rows: IntArr) -> FloatArr:
        scale = self.cfg.occupancy.coverage_weight_scale
        return np.asarray(1.0 - np.exp(-self.f["w"][rows] / scale), dtype=np.float64)

    def variance(self, rows: IntArr) -> FloatArr:
        """Beta-style spread p(1-p)/(1+w): shrinks only with direct mass, never with inference."""
        p = self.probability(rows)
        return np.asarray(p * (1.0 - p) / (1.0 + self.f["w"][rows]), dtype=np.float64)

    def uncertainty(self, rows: IntArr) -> FloatArr:
        """(N, 4) channels (UA, UE, UC, UO), each in [0, 1]."""
        w = self.f["w"][rows]
        has = w > 0
        safe = np.maximum(w, 1e-12)
        ua = np.where(has, self.f["ua_acc"][rows] / safe, 0.0)
        ue = np.where(has, self.f["ue_acc"][rows] / safe, 1.0)
        inferred_only = self.b["inferred"][rows] & (self.status(rows) == INFERRED)
        ue = np.where(inferred_only, np.maximum(ue * has, self.f["inf_ue"][rows]), ue)
        uo = 1.0 - self.coverage(rows)
        uo = np.where(inferred_only, np.maximum(uo, self.f["inf_uo"][rows]), uo)
        uo = np.clip(uo + self.f["pred_uo"][rows] * (1.0 - uo), 0.0, 1.0)
        return np.stack([ua, ue, self.f["change"][rows], uo], axis=1)

    # ------------------------------------------------------------------ direct update
    def apply_direct(
        self,
        rows: IntArr,
        hit_mass: FloatArr,
        miss_mass: FloatArr,
        ua: FloatArr,
        ue: FloatArr,
        sigma_pose: FloatArr,
        evidence_id: UUID,
        t_ns: int,
    ) -> None:
        """Fuse one observation's per-cell hit/free mass. Contradictions against confident OBSERVED cells
        raise the change score (UC); the map is not overwritten at once (log-odds clamps, ch14)."""
        oc, tc = self.cfg.occupancy, self.cfg.temporal
        prior_status = self.status(rows)
        prior_p = self.probability(rows)
        net = hit_mass - miss_mass
        conf = np.abs(2.0 * prior_p - 1.0)
        prior_sign = np.sign(prior_p - 0.5)
        credible = (1.0 - ue) * np.minimum(np.abs(net), 1.0)
        contra = (
            (prior_status == OBSERVED)
            & (conf >= tc.contradiction_min_confidence)
            & (np.abs(net) >= tc.contradiction_min_mass)
            & (np.sign(net) != prior_sign)
        )
        agree = (prior_status == OBSERVED) & ~contra & (np.abs(net) >= tc.contradiction_min_mass)
        ch = self.f["change"][rows]
        ch = np.where(contra, (1 - tc.change_alpha) * ch + tc.change_alpha * conf * credible, ch)
        ch = np.where(agree, ch * (1.0 - 0.5 * tc.change_alpha), ch)
        self.f["change"][rows] = np.clip(ch, 0.0, 1.0)
        self.b["dynamic"][rows] |= ch >= tc.dynamic_change_threshold
        dlo = oc.l_hit * hit_mass + oc.l_miss * miss_mass
        self.f["lo"][rows] = np.clip(self.f["lo"][rows] + dlo, oc.l_min, oc.l_max)
        mass = hit_mass + miss_mass
        self.f["w"][rows] += mass
        self.f["w_hit"][rows] += hit_mass
        self.f["ua_acc"][rows] += mass * ua
        self.f["ue_acc"][rows] += mass * ue
        self.f["sig_acc"][rows] += mass * sigma_pose
        self.n_obs[rows] += 1
        self.last_ns[rows] = np.maximum(self.last_ns[rows], t_ns)
        self.src[rows] = SRC_DIRECT
        self.b["predicted"][rows] = False
        self.f["pred_uo"][rows] = 0.0
        self.add_provenance(rows, [evidence_id])
