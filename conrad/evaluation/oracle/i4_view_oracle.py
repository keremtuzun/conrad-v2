"""EVALUATION-ONLY view oracle for the I4 occluded family: the headroom a perfect ranker would have.

This module SEES TRUTH. It exists to answer one question about the world family, not to be a planner:

    if the ranking rule were replaced by one that knows, per candidate, exactly which part of the hidden
    defect that view would reveal, how much better than a systematic coverage sweep would the mission be?

It lives on the evaluation side and is unreachable from the deployment planes: ``tests/leakage``'s static
guard forbids ``conrad.active``, ``conrad.orchestration`` and ``conrad.decision`` from importing
``conrad.evaluation`` at all. An experiment injects the oracle into a run the same way ACTIVE-MCBR-E005
injects a candidate production planner (it rebinds ``conrad.orchestration.deliberation.production_planner``
inside its own worker process), so no runtime module ever names it.

The oracle's objective is the truth-side surface of the defect patch a candidate view would newly see, on the
same surface-cell partition Model2T uses, with the world's real unregistered occluders. In this family that
quantity IS the hidden-state outcome and not a proxy for it: over the 120 spent run-2 missions, a world whose
defect was read scored a mean hidden_state_error_improvement of 0.78 and a world whose defect was not read
scored exactly 0.00, with no intermediate case (docs/audits/MCBR_V3_FAILURE_ANALYSIS.md).

``steps=1`` ranks a candidate by what it alone would reveal. ``steps=2`` ranks it by the best two-view union
that starts with it, which is the standard one-step-lookahead relaxation of the sequential problem.

implementation_status: EXPERIMENTAL_CANDIDATE (evaluation instrument)
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from conrad.active.candidates import ViewpointGenerator
from conrad.active.config import MCBRConfig
from conrad.active.gap import KnowledgeGap
from conrad.active.planner import MCBRPlanner, PlanningRequest, ScoredCandidate
from conrad.schemas.ids import IdFactory

CELL_M = 0.5  # CoverageConfig.cell_m
SECTORS = 8  # CoverageConfig.sectors
PROBE_OFFSET_M = 0.15  # SurfacePredictiveConfig.probe_offset_m
POST_HALF_M = 0.07  # ViewOcclusionOptions.post_half_m
ORACLE_1 = "ORACLE_1STEP"
ORACLE_2 = "ORACLE_2STEP"
ORACLE_1_RATIO = "ORACLE_1STEP_RATIO"
ORACLE_W = "ORACLE_1STEP_WEIGHTED"
ORACLE_W_RATIO = "ORACLE_1STEP_WEIGHTED_RATIO"
COST_FLOOR = 0.05  # RankerConfig.cost_floor of the frozen V-bayes_eig_ratio planner
COST_WEIGHT = 1.0  # RankerConfig.cost_weight of the same


def _unit(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) / max(float(np.linalg.norm(v)), 1e-12)


@dataclass(frozen=True)
class _Obb:
    centre: np.ndarray
    axes: np.ndarray
    half: np.ndarray


@dataclass(frozen=True)
class TruthView:
    """Everything the oracle needs about one world, harvested from a finished run bundle's truth record."""

    seed: int
    cell_points: np.ndarray
    cell_normals: np.ndarray
    patch_mask: np.ndarray
    boxes: tuple[_Obb, ...]
    max_range_m: float

    def visible(self, pose_position: np.ndarray) -> np.ndarray:
        """Cells this pose can see at all: facing, in range, and not hidden by the world's rack panels."""
        ray = pose_position[None, :] - self.cell_points
        dist = np.linalg.norm(ray, axis=1)
        cos = np.einsum("ij,ij->i", self.cell_normals, ray) / np.maximum(dist, 1e-9)
        ok = (cos > 0.0) & (dist <= self.max_range_m)
        for i in np.nonzero(ok)[0]:
            probe = self.cell_points[i] + PROBE_OFFSET_M * self.cell_normals[i]
            if any(_hits(probe, pose_position, box) for box in self.boxes):
                ok[i] = False
        return ok

    def weights(self, pose_position: np.ndarray) -> np.ndarray:
        """Per-cell observation weight, cos(incidence) on the visible cells and 0 elsewhere.

        This is the truth-side twin of the production predictive model's own cell weight
        (``MissionPredictive`` CellWeightFn: clear-ray probability times cos(incidence) ** incidence_power,
        with incidence_power 1.0), with the world's real occluders in place of the believed ones. A binary
        visibility test over-credits a grazing look, which a real reading does not deliver."""
        ray = pose_position[None, :] - self.cell_points
        dist = np.linalg.norm(ray, axis=1)
        cos = np.einsum("ij,ij->i", self.cell_normals, ray) / np.maximum(dist, 1e-9)
        return np.where(self.visible(pose_position), np.clip(cos, 0.0, 1.0), 0.0)

    def patch_fraction(self, mask: np.ndarray) -> float:
        """Share of the defect patch this per-cell weight vector delivers (a boolean mask counts as 1.0)."""
        n = int(self.patch_mask.sum())
        if not n:
            return 0.0
        w = np.asarray(mask, dtype=np.float64)
        return float(w[self.patch_mask].sum() / n)

    def to_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "cell_points": self.cell_points.tolist(),
            "cell_normals": self.cell_normals.tolist(),
            "patch_mask": self.patch_mask.astype(bool).tolist(),
            "max_range_m": self.max_range_m,
            "boxes": [
                {"centre": b.centre.tolist(), "axes": b.axes.tolist(), "half": b.half.tolist()}
                for b in self.boxes
            ],
        }

    @staticmethod
    def from_json(raw: dict[str, Any]) -> TruthView:
        return TruthView(
            seed=int(raw["seed"]),
            cell_points=np.asarray(raw["cell_points"], dtype=np.float64),
            cell_normals=np.asarray(raw["cell_normals"], dtype=np.float64),
            patch_mask=np.asarray(raw["patch_mask"], dtype=bool),
            boxes=tuple(
                _Obb(
                    np.asarray(b["centre"], dtype=np.float64),
                    np.asarray(b["axes"], dtype=np.float64),
                    np.asarray(b["half"], dtype=np.float64),
                )
                for b in raw["boxes"]
            ),
            max_range_m=float(raw["max_range_m"]),
        )


def _hits(a: np.ndarray, b: np.ndarray, box: _Obb) -> bool:
    o = box.axes @ (a - box.centre)
    d = box.axes @ (b - a)
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if abs(o[k]) > box.half[k]:
                return False
            continue
        lo = (-box.half[k] - o[k]) / d[k]
        hi = (box.half[k] - o[k]) / d[k]
        if lo > hi:
            lo, hi = hi, lo
        t0, t1 = max(t0, lo), min(t1, hi)
        if t0 > t1:
            return False
    return True


def harvest(run_dir: Path) -> TruthView:
    """Build the oracle's world model from a finished bundle (truth record + surveyed design geometry)."""
    truth = json.loads((run_dir / "truth" / "truth_record.json").read_text(encoding="utf-8"))
    ctx = json.loads((run_dir / "capture" / "mission_context.json").read_text(encoding="utf-8"))
    meta = truth["meta"]
    target = str(meta["target_registry_id"])
    item = next(d for d in ctx["design"] if str(d["registry_id"]) == target)
    p0 = np.asarray(item["p0_m"], dtype=np.float64)
    p1 = np.asarray(item["p1_m"], dtype=np.float64)
    radius = float(item["radius_m"])
    axis = _unit(p1 - p0)
    length = float(np.linalg.norm(p1 - p0))
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _unit(ref - (ref @ axis) * axis)
    v = np.cross(axis, u)
    n_along = max(1, math.ceil(length / CELL_M))
    pts, nrm, ts = [], [], []
    for i in range(n_along):
        for s in range(SECTORS):
            ang = (s + 0.5) * 2.0 * math.pi / SECTORS
            n = math.cos(ang) * u + math.sin(ang) * v
            t = (i + 0.5) / n_along * length
            pts.append(p0 + t * axis + radius * n)
            nrm.append(n)
            ts.append(t)
    points, normals, taxis = np.asarray(pts), np.asarray(nrm), np.asarray(ts)

    defect = meta["defect"]
    pdir = _unit(np.asarray(meta["patch_direction"], dtype=np.float64))
    half = math.radians(float(defect["patch_half_angle_deg"]))
    axial = float(defect["patch_axial_fraction"]) * length
    t_c = float((np.asarray(meta["patch_centre_m"], dtype=np.float64) - p0) @ axis)
    patch = (normals @ pdir >= math.cos(half)) & (np.abs(taxis - t_c) <= 0.5 * axial)

    occ = meta.get("view_occlusion") or {}
    panel_half = np.asarray(occ.get("panel_half_extent_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    boxes: list[_Obb] = []
    for p in occ.get("panels", []):
        m = _unit(np.asarray(p["normal"], dtype=np.float64))
        x = _unit(axis - (axis @ m) * m)
        boxes.append(
            _Obb(
                np.asarray(p["center_m"], dtype=np.float64),
                np.stack([x, np.cross(m, x), m]),
                panel_half,
            )
        )
    for p in occ.get("posts", []):
        hz = float(p["half_z_m"])
        boxes.append(
            _Obb(
                np.asarray(p["center_m"], dtype=np.float64),
                np.eye(3),
                np.array([POST_HALF_M, POST_HALF_M, hz]),
            )
        )
    ids = set(ctx.get("structural_sensor_ids", []))
    rng_m = 4.0
    for s in ctx.get("sensors", []):
        if str(s.get("sensor_id")) in ids:
            rng_m = float(s.get("parameters", {}).get("max_range_m", 4.0))
            break
    return TruthView(int(meta["seed"]), points, normals, patch, tuple(boxes), rng_m)


class TruthViewOracle:
    """Scorer that ranks candidates by the defect surface they would newly reveal. TRUTH, evaluation only."""

    def __init__(
        self,
        truth: TruthView,
        steps: int = 1,
        cfg: MCBRConfig | None = None,
        cost_mode: str = "none",
        weighted: bool = False,
    ) -> None:
        if steps not in (1, 2):
            raise ValueError("the oracle supports one-step and two-step lookahead")
        if cost_mode not in ("none", "ratio"):
            raise ValueError("cost_mode is 'none' (pure selection) or 'ratio' (the frozen ranker's rule)")
        self.truth = truth
        self.steps = steps
        self.cost_mode = cost_mode
        self.weighted = weighted
        self.cfg = cfg or MCBRConfig()
        self._cache: dict[tuple[float, float, float], np.ndarray] = {}
        self._request: int | None = None
        self._seen: np.ndarray | None = None
        self._all: list[np.ndarray] = []

    def _mask(self, position: Sequence[float]) -> np.ndarray:
        key = (round(position[0], 4), round(position[1], 4), round(position[2], 4))
        hit = self._cache.get(key)
        if hit is None:
            p = np.asarray(key, dtype=np.float64)
            hit = self.truth.weights(p) if self.weighted else self.truth.visible(p).astype(np.float64)
            self._cache[key] = hit
        return hit

    def _prepare(self, gap: KnowledgeGap) -> np.ndarray:
        """Cells the oracle already counts as read: every earlier view of this need. Recomputed per call
        (the per-pose masks are cached), because a PlanningRequest id can be reused after collection."""
        key = tuple(sorted(tuple(float(x) for x in v.position_m) for v in gap.prior_views))
        if self._seen is None or self._request != hash(key):
            seen = np.zeros(len(self.truth.cell_points), dtype=np.float64)
            for view in gap.prior_views:
                seen = np.maximum(seen, self._mask(tuple(float(x) for x in view.position_m)))
            self._seen, self._request = seen, hash(key)
        return self._seen

    def __call__(self, gap: KnowledgeGap, c: ScoredCandidate, req: PlanningRequest) -> float:
        seen = self._prepare(gap)
        mine = self._mask(tuple(float(x) for x in c.raw.pose.position_m))
        first = self.truth.patch_fraction(np.maximum(mine - seen, 0.0))
        if self.steps == 1:
            value = first
        else:
            # Two-step: the best two-view union that starts here. Many starts reach the same union, so the
            # immediate gain breaks the tie; without it the lookahead saturates and ranks by candidate index.
            best = first
            for other in self._peers(gap, req):
                union = np.maximum(np.maximum(mine, other) - seen, 0.0)
                best = max(best, self.truth.patch_fraction(union))
            value = best + 1e-3 * first
        if self.cost_mode == "ratio":  # the frozen planner's own cost handling, truth value in place of EIG
            return value / (COST_FLOOR + COST_WEIGHT * c.cost)
        return value

    def _peers(self, gap: KnowledgeGap, req: PlanningRequest) -> list[np.ndarray]:
        """Masks of every candidate of this plan, for the two-step union. Computed once per request."""
        if not self._all:
            raws = ViewpointGenerator(self.cfg).generate(gap.target_region, req.sensors)
            self._all = [self._mask(tuple(float(x) for x in r.pose.position_m)) for r in raws]
        return self._all


#: arm name -> (lookahead steps, cost handling, incidence weighting)
ARMS: dict[str, tuple[int, str, bool]] = {
    ORACLE_1: (1, "none", False),
    ORACLE_2: (2, "none", False),
    ORACLE_1_RATIO: (1, "ratio", False),
    ORACLE_W: (1, "none", True),
    ORACLE_W_RATIO: (1, "ratio", True),
}


def oracle_planner(
    ids: IdFactory,
    cfg: MCBRConfig,
    truth: TruthView,
    steps: int = 1,
    cost_mode: str = "none",
    weighted: bool = False,
    name: str = ORACLE_1,
) -> MCBRPlanner:
    """The oracle behind the unchanged MCBR pipeline: same candidates, same feasibility filter, new ranking."""
    scorer = TruthViewOracle(truth, steps, cfg, cost_mode, weighted)
    return MCBRPlanner(ids, cfg, scorer, name=name, value_gate=False)


__all__ = [
    "ARMS",
    "ORACLE_1",
    "ORACLE_1_RATIO",
    "ORACLE_2",
    "ORACLE_W",
    "ORACLE_W_RATIO",
    "TruthView",
    "TruthViewOracle",
    "harvest",
    "oracle_planner",
]
