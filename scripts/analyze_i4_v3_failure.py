"""Post-hoc analysis of the SPENT gate I4 traces (run 2, 8001000-8001059; run 1 where useful).

EVALUATION ONLY, POST HOC. It reads finished Unity run bundles and the evaluation-only truth record inside
them. Nothing here is imported by any runtime plane and nothing here may be used to tune a planner: the worlds
are spent, so every number below can motivate a hypothesis and can falsify a stated mechanism, but it cannot
select or calibrate anything. See docs/audits/MCBR_V3_FAILURE_ANALYSIS.md.

Output: artifacts/analysis/MCBR_V3_FAILURE/{per_world.json,per_view.json,candidates.json,summary.json}.

Usage:

    python -m uv run python scripts/analyze_i4_v3_failure.py [--run2-dir ...] [--run1-dir ...]

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from conrad.domains.technical.config import PriorConfig

REPO = Path(__file__).resolve().parents[1]
RUN2_DIR = REPO / "artifacts" / "unity" / "gate_runs" / "I4-OCCLUDED-REP2"
RUN1_DIR = REPO / "artifacts" / "unity" / "gate_runs" / "I4-OCCLUDED"
OUT_DIR = REPO / "artifacts" / "analysis" / "MCBR_V3_FAILURE"

CELL_M = 0.5  # CoverageConfig.cell_m
SECTORS = 8  # CoverageConfig.sectors
PROBE_OFFSET_M = 0.15  # SurfacePredictiveConfig.probe_offset_m
COST_FLOOR = 0.05  # RankerConfig.cost_floor of the frozen V-bayes_eig_ratio planner
COST_WEIGHT = 1.0
QUANTITIES = ("corrosion_depth_m", "crack_length_m")
POST_HALF_M = 0.07  # ViewOcclusionOptions.post_half_m
PRIOR_MEAN = dict(PriorConfig().mean)  # the same prior the run's own scorer normalises against


# ------------------------------------------------------------------------------------------------ geometry
def _unit(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) / max(float(np.linalg.norm(v)), 1e-12)


@dataclass(frozen=True)
class Capsule:
    p0: np.ndarray
    p1: np.ndarray
    radius_m: float

    @property
    def axis(self) -> np.ndarray:
        return _unit(self.p1 - self.p0)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def centre(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)


def cell_grid(cap: Capsule) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell centres, outward normals and axial coordinates, indexed as SurfaceGeometry.cell_of indexes."""
    d = cap.axis
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _unit(ref - (ref @ d) * d)
    v = np.cross(d, u)
    n_along = max(1, math.ceil(cap.length / CELL_M))
    pts, nrm, ts = [], [], []
    for i in range(n_along):
        for s in range(SECTORS):
            ang = (s + 0.5) * 2.0 * math.pi / SECTORS
            n = math.cos(ang) * u + math.sin(ang) * v
            t = (i + 0.5) / n_along * cap.length
            pts.append(cap.p0 + t * d + cap.radius_m * n)
            nrm.append(n)
            ts.append(t)
    return np.asarray(pts), np.asarray(nrm), np.asarray(ts)


@dataclass(frozen=True)
class Obb:
    centre: np.ndarray
    axes: np.ndarray  # rows = unit axes
    half: np.ndarray


def panels_of(meta: dict[str, Any], cap: Capsule) -> list[Obb]:
    occ = meta.get("view_occlusion") or {}
    half = np.asarray(occ.get("panel_half_extent_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    out = []
    d = cap.axis
    for p in occ.get("panels", []):
        m = _unit(np.asarray(p["normal"], dtype=np.float64))
        x = _unit(d - (d @ m) * m)
        y = np.cross(m, x)
        out.append(Obb(np.asarray(p["center_m"], dtype=np.float64), np.stack([x, y, m]), half))
    for p in occ.get("posts", []):
        hz = float(p["half_z_m"])
        out.append(
            Obb(
                np.asarray(p["center_m"], dtype=np.float64),
                np.eye(3),
                np.array([POST_HALF_M, POST_HALF_M, hz]),
            )
        )
    return out


def segment_hits_obb(a: np.ndarray, b: np.ndarray, box: Obb) -> bool:
    """Slab test of the segment a->b against an oriented box."""
    o = box.axes @ (a - box.centre)
    dvec = box.axes @ (b - a)
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(dvec[k]) < 1e-12:
            if abs(o[k]) > box.half[k]:
                return False
            continue
        lo = (-box.half[k] - o[k]) / dvec[k]
        hi = (box.half[k] - o[k]) / dvec[k]
        if lo > hi:
            lo, hi = hi, lo
        t0, t1 = max(t0, lo), min(t1, hi)
        if t0 > t1:
            return False
    return True


def visible_cells(
    view: np.ndarray, pts: np.ndarray, nrm: np.ndarray, boxes: Sequence[Obb], max_range_m: float
) -> np.ndarray:
    """Boolean mask of surface cells this view pose can see (facing, in range, unoccluded by the rack)."""
    ray = view[None, :] - pts
    dist = np.linalg.norm(ray, axis=1)
    cos = np.einsum("ij,ij->i", nrm, ray) / np.maximum(dist, 1e-9)
    ok = (cos > 0.0) & (dist <= max_range_m)
    for i in np.nonzero(ok)[0]:
        probe = pts[i] + PROBE_OFFSET_M * nrm[i]
        if any(segment_hits_obb(probe, view, box) for box in boxes):
            ok[i] = False
    return ok


def patch_cells(meta: dict[str, Any], cap: Capsule, nrm: np.ndarray, ts: np.ndarray) -> np.ndarray:
    defect = meta["defect"]
    pdir = _unit(np.asarray(meta["patch_direction"], dtype=np.float64))
    half = math.radians(float(defect["patch_half_angle_deg"]))
    axial = float(defect["patch_axial_fraction"]) * cap.length
    t_c = float((np.asarray(meta["patch_centre_m"], dtype=np.float64) - cap.p0) @ cap.axis)
    return (nrm @ pdir >= math.cos(half)) & (np.abs(ts - t_c) <= 0.5 * axial)


def azimuth_elevation(view: np.ndarray, centre: np.ndarray) -> tuple[float, float]:
    r = view - centre
    return math.degrees(math.atan2(r[1], r[0])), math.degrees(math.atan2(r[2], math.hypot(r[0], r[1])))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))))


# ------------------------------------------------------------------------------------------------ bundle io
@dataclass
class Bundle:
    path: Path
    seed: int
    arm: str
    truth: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    goals: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    revisions: list[dict[str, Any]] = field(default_factory=list)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def target_revisions(path: Path, target_registry_id: str) -> list[dict[str, Any]]:
    con = sqlite3.connect(path / "conrad.sqlite")
    try:
        ids = [
            r[0]
            for r in con.execute(
                "select belief_id from beliefs where registry_entity_id = ?", (target_registry_id,)
            )
        ]
        if not ids:
            return []
        rows = list(
            con.execute(
                "select revision, update_kind, measurement_time_ns, payload_json from belief_revisions"
                " where belief_id = ? order by revision",
                (ids[0],),
            )
        )
    finally:
        con.close()
    out = []
    for rev, kind, ns, payload in rows:
        cell = json.loads(payload)["cell"]
        claims = {c["name"]: c for c in cell["claims"]}
        rec: dict[str, Any] = {
            "revision": int(rev),
            "update_kind": str(kind),
            "t_s": ns / 1e9,
            "U_A": cell["uncertainty"]["aleatoric"],
            "U_E": cell["uncertainty"]["epistemic"],
            "U_C": cell["uncertainty"]["contradiction"],
            "U_O": cell["uncertainty"]["observational"],
            "knowledge_status": cell["knowledge_status"],
        }
        for name in (*QUANTITIES, "condition"):
            c = claims.get(name)
            rec[name] = None if c is None else c.get("value")
            rec[f"{name}.status"] = "UNKNOWN" if c is None else c["status"]
        out.append(rec)
    return out


def load_bundle(path: Path) -> Bundle:
    name = path.name
    seed = int(name.split("-s")[1].split("-")[0])
    arm = name.split(f"-s{seed}-", 1)[1]
    b = Bundle(path=path, seed=seed, arm=arm)
    b.truth = _json(path / "truth" / "truth_record.json")
    b.context = _json(path / "capture" / "mission_context.json")
    b.goals = _json(path / "mission" / "trajectories.json")["goals"]
    tables_path = path / "mission" / "mcbr_candidate_tables.json"
    b.tables = _json(tables_path) if tables_path.exists() else []
    b.metrics = _json(path / "mission" / "runtime_metrics.json")
    b.revisions = target_revisions(path, b.truth["meta"]["target_registry_id"])
    return b


def structural_range_m(ctx: dict[str, Any]) -> float:
    ids = set(ctx.get("structural_sensor_ids", []))
    for s in ctx.get("sensors", []):
        if str(s.get("sensor_id")) in ids:
            return float(s.get("parameters", {}).get("max_range_m", 4.0))
    return 4.0


def design_capsule(ctx: dict[str, Any], registry_id: str) -> Capsule:
    for item in ctx["design"]:
        if str(item["registry_id"]) == registry_id:
            return Capsule(
                np.asarray(item["p0_m"], dtype=np.float64),
                np.asarray(item["p1_m"], dtype=np.float64),
                float(item["radius_m"]),
            )
    raise KeyError(f"no design geometry for {registry_id}")


# ------------------------------------------------------------------------------------------------ measures
def abs_errors(rec: dict[str, Any] | None, truth_end: dict[str, Any]) -> dict[str, float]:
    out = {}
    for q in QUANTITIES:
        prior = abs(PRIOR_MEAN[q] - float(truth_end[q]))
        value = None if rec is None else rec.get(q)
        err = prior if value is None else abs(float(value) - float(truth_end[q]))
        out[q] = err
        out[f"{q}.norm"] = err / prior if prior > 0 else 1.0
    out["hse_final"] = float(np.mean([out[f"{q}.norm"] for q in QUANTITIES]))
    out["hse_improvement"] = 1.0 - out["hse_final"]
    return out


def state_at(revisions: Sequence[dict[str, Any]], t_s: float) -> dict[str, Any] | None:
    seen = [r for r in revisions if r["t_s"] <= t_s]
    return seen[-1] if seen else None


def _rank(x: Sequence[float]) -> np.ndarray:
    """Average ranks (ties share their mean rank). Plain argsort ranks are wrong here: the oracle label is
    heavily tied, and index order is candidate-generation order, which correlates with the score."""
    a = np.asarray(x, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_paired(diffs: Sequence[float], n: int = 4000, seed: int = 20260920) -> dict[str, float]:
    d = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return {
        "mean": float(d.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
    }


# ------------------------------------------------------------------------------------------- per-run report
def analyse_bundle(b: Bundle) -> dict[str, Any]:
    meta = b.truth["meta"]
    cap = design_capsule(b.context, meta["target_registry_id"])
    pts, nrm, ts = cell_grid(cap)
    boxes = panels_of(meta, cap)
    patch = patch_cells(meta, cap, nrm, ts)
    rng_m = structural_range_m(b.context)
    truth_end = b.truth["target_states"]["end"]
    duration = float(b.truth["trajectory"][-1][0]) if b.truth["trajectory"] else 100.0

    inspects = [g for g in b.goals if g["purpose"] == "INSPECT" and g["accepted"]]
    direct = [r for r in b.revisions if r["update_kind"] == "DIRECT"]
    patch_series = b.truth["series"].get("patch_visibility", [])
    labels = b.truth["series"].get("structural_label", [])
    t_patch_read = next((x["t_s"] for x in labels if x.get("patch")), None)

    views: list[dict[str, Any]] = []
    seen_union = np.zeros(len(pts), dtype=bool)
    dirs: list[np.ndarray] = []
    for i, g in enumerate(inspects):
        pos = np.asarray(g["target"], dtype=np.float64)
        t0 = float(g["t_s"])
        t1 = float(inspects[i + 1]["t_s"]) if i + 1 < len(inspects) else math.inf
        mask = visible_cells(pos, pts, nrm, boxes, rng_m)
        az, el = azimuth_elevation(pos, cap.centre)
        direction = pos - cap.centre
        overlap = float((mask & seen_union).sum() / max(int(mask.sum()), 1))
        table = b.tables[i]["candidate_table"] if i < len(b.tables) else []
        chosen = _chosen_row(table, pos)
        reached, gap_m = _closest_approach(b.truth, pos, t0, t1)
        reached_mask = visible_cells(reached, pts, nrm, boxes, rng_m) if reached is not None else None
        before = state_at(b.revisions, t0)
        after = state_at(b.revisions, t1 if math.isfinite(t1) else duration + 1.0)
        e_before, e_after = abs_errors(before, truth_end), abs_errors(after, truth_end)
        views.append(
            {
                "index": i,
                "t_s": t0,
                "position_m": [float(x) for x in pos],
                "azimuth_deg": az,
                "elevation_deg": el,
                "angular_diversity_min_deg": (
                    min(angle_between(direction, d) for d in dirs) if dirs else None
                ),
                "angular_diversity_mean_deg": (
                    float(np.mean([angle_between(direction, d) for d in dirs])) if dirs else None
                ),
                "cells_visible": int(mask.sum()),
                "cell_overlap_with_earlier_views": overlap,
                "new_cell_fraction": 1.0 - overlap,
                "patch_cells_visible_fraction": float((mask & patch).sum() / max(int(patch.sum()), 1)),
                "predicted_score": None if chosen is None else chosen["score"],
                "predicted_cost": None if chosen is None else chosen["cost"],
                "predicted_value": None if chosen is None else _value_of(b.arm, chosen),
                "analytic_gain_total": None if chosen is None else _gain_total(chosen),
                "closest_approach_m": gap_m,
                "realised_patch_cells_visible_fraction": (
                    None
                    if reached_mask is None
                    else float((reached_mask & patch).sum() / max(int(patch.sum()), 1))
                ),
                "truth_patch_visible_fraction_window": max(
                    (p["visible_fraction"] for p in patch_series if t0 <= p["t_s"] < t1), default=0.0
                ),
                "direct_revisions_in_window": sum(1 for r in direct if t0 < r["t_s"] <= t1),
                "hse_improvement_before": e_before["hse_improvement"],
                "hse_improvement_after": e_after["hse_improvement"],
                "hse_improvement_delta": e_after["hse_improvement"] - e_before["hse_improvement"],
            }
        )
        seen_union |= mask
        dirs.append(direction)

    detect_idx = next(
        (v["index"] for v in views if v["direct_revisions_in_window"] > 0),
        None,
    )
    t_detect = direct[0]["t_s"] if direct else None
    final = b.revisions[-1] if b.revisions else None
    errs = abs_errors(final, truth_end)
    at_detect = abs_errors(state_at(b.revisions, t_detect), truth_end) if t_detect else None
    return {
        "seed": b.seed,
        "arm": b.arm,
        "inspection_views": len(inspects),
        "planner_calls": len(b.metrics.get("plans", [])),
        "plan_statuses": [p["status"] for p in b.metrics.get("plans", [])],
        "views": views,
        "view_pair_angles_deg": [
            angle_between(dirs[i], dirs[j]) for i in range(len(dirs)) for j in range(i + 1, len(dirs))
        ],
        "mean_patch_cells_visible_fraction": float(
            np.mean([v["patch_cells_visible_fraction"] for v in views])
        )
        if views
        else 0.0,
        "mean_new_cell_fraction": float(np.mean([v["new_cell_fraction"] for v in views])) if views else None,
        "patch_max_visible_fraction": max((p["visible_fraction"] for p in patch_series), default=0.0),
        "detection_view_index": detect_idx,
        "detected": float(detect_idx is not None),
        "mean_realised_patch_cells_visible_fraction": _mean(
            v["realised_patch_cells_visible_fraction"] for v in views
        )
        or 0.0,
        "detection_t_s": t_detect,
        "first_patch_reading_t_s": t_patch_read,
        "post_detection_views": (
            sum(1 for v in views if detect_idx is not None and v["index"] > detect_idx)
            if detect_idx is not None
            else 0
        ),
        "hse_improvement_at_detection": None if at_detect is None else at_detect["hse_improvement"],
        "post_detection_hse_gain": (
            None if at_detect is None else errs["hse_improvement"] - at_detect["hse_improvement"]
        ),
        "corrosion_abs_error_m": errs["corrosion_depth_m"],
        "crack_abs_error_m": errs["crack_length_m"],
        "hidden_state_error_improvement": errs["hse_improvement"],
        "read_surface_fraction_proxy": None if final is None else 1.0 - float(final["U_O"]),
        "U_O": None if final is None else float(final["U_O"]),
        "U_C": None if final is None else float(final["U_C"]),
        "U_E": None if final is None else float(final["U_E"]),
        "U_A": None if final is None else float(final["U_A"]),
        "final_condition": None if final is None else final.get("condition"),
        "final_condition_status": None if final is None else final.get("condition.status"),
        "target_observed": float(final is not None and final.get("condition.status") == "OBSERVED"),
        "travel_m": _travel(b.truth),
        "mission_time_s": duration,
        "energy_j": float(b.truth["vehicle"]["energy_used_j"]),
        "collisions": float(b.truth["vehicle"]["collisions"]),
        "direct_revisions": len(direct),
        "defect_truth": {q: float(truth_end[q]) for q in QUANTITIES},
        "n_patch_cells": int(patch.sum()),
        "n_cells": len(pts),
    }


def _closest_approach(
    truth: dict[str, Any], pos: np.ndarray, t0: float, t1: float
) -> tuple[np.ndarray | None, float | None]:
    """Where the vehicle actually got to while this commanded view was the active goal."""
    rows = [r for r in truth["trajectory"] if t0 <= float(r[0]) < t1]
    if not rows:
        return None, None
    pts = np.asarray([r[1:4] for r in rows], dtype=np.float64)
    i = int(np.argmin(np.linalg.norm(pts - pos[None, :], axis=1)))
    return pts[i], float(np.linalg.norm(pts[i] - pos))


def _travel(truth: dict[str, Any]) -> float:
    traj = np.asarray([row[1:4] for row in truth["trajectory"]], dtype=np.float64)
    return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()) if len(traj) > 1 else 0.0


def _chosen_row(table: Sequence[dict[str, Any]], pos: np.ndarray) -> dict[str, Any] | None:
    feasible = [r for r in table if r.get("feasible")]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda r: float(np.linalg.norm(np.asarray(r["position_m"], dtype=np.float64) - pos)),
    )


def _gain_total(row: dict[str, Any]) -> float:
    g = row.get("gain") or {}
    return float(
        g.get("aleatoric", 0.0)
        + g.get("epistemic", 0.0)
        + g.get("contradiction", 0.0)
        + g.get("observational", 0.0)
    )


def _value_of(arm: str, row: dict[str, Any]) -> float:
    """PRODUCTION ranks by value / (cost_floor + w * cost), so the predictive EIG is recoverable."""
    if arm == "PRODUCTION":
        return float(row["score"]) * (COST_FLOOR + COST_WEIGHT * float(row["cost"]))
    return float(row["score"])


# --------------------------------------------------------------------------------- candidate-level calibration
def candidate_report(b: Bundle) -> list[dict[str, Any]]:
    """Every feasible candidate of every plan, labelled with the evaluation-only geometric oracle."""
    meta = b.truth["meta"]
    cap = design_capsule(b.context, meta["target_registry_id"])
    pts, nrm, ts = cell_grid(cap)
    boxes = panels_of(meta, cap)
    patch = patch_cells(meta, cap, nrm, ts)
    rng_m = structural_range_m(b.context)
    inspects = [g for g in b.goals if g["purpose"] == "INSPECT" and g["accepted"]]
    out: list[dict[str, Any]] = []
    for plan_i, table in enumerate(b.tables):
        rows = [r for r in table["candidate_table"] if r.get("feasible")]
        if not rows:
            continue
        taken = np.asarray(inspects[plan_i]["target"], dtype=np.float64) if plan_i < len(inspects) else None
        chosen = None if taken is None else _chosen_row(rows, taken)
        for r in rows:
            pos = np.asarray(r["position_m"], dtype=np.float64)
            mask = visible_cells(pos, pts, nrm, boxes, rng_m)
            out.append(
                {
                    "seed": b.seed,
                    "arm": b.arm,
                    "plan": plan_i,
                    "action_id": r["action_id"],
                    "score": float(r["score"]),
                    "value": _value_of(b.arm, r),
                    "cost": float(r["cost"]),
                    "visibility": float(r["visibility"]),
                    "analytic_gain_total": _gain_total(r),
                    "oracle_patch_fraction": float((mask & patch).sum() / max(int(patch.sum()), 1)),
                    "oracle_cells": int(mask.sum()),
                    "chosen": bool(chosen is not None and r["action_id"] == chosen["action_id"]),
                }
            )
    return out


# ----------------------------------------------------------------------------------------- redundancy probe
def redundancy_probe(b: Bundle) -> list[dict[str, Any]]:
    """Limitation B: after a view is taken, does the predicted value of candidates that OVERLAP it fall
    more than the value of candidates that do not? Candidates are matched across plans by pose."""
    meta = b.truth["meta"]
    cap = design_capsule(b.context, meta["target_registry_id"])
    pts, nrm, _ = cell_grid(cap)
    boxes = panels_of(meta, cap)
    rng_m = structural_range_m(b.context)
    inspects = [g for g in b.goals if g["purpose"] == "INSPECT" and g["accepted"]]
    out: list[dict[str, Any]] = []
    for plan_i in range(len(b.tables) - 1):
        if plan_i >= len(inspects):
            break
        taken = np.asarray(inspects[plan_i]["target"], dtype=np.float64)
        taken_mask = visible_cells(taken, pts, nrm, boxes, rng_m)
        if not taken_mask.any():
            continue
        now = {_key(r): r for r in b.tables[plan_i]["candidate_table"] if r.get("feasible")}
        nxt = {_key(r): r for r in b.tables[plan_i + 1]["candidate_table"] if r.get("feasible")}
        for key, r0 in now.items():
            r1 = nxt.get(key)
            if r1 is None:
                continue
            pos = np.asarray(r0["position_m"], dtype=np.float64)
            mask = visible_cells(pos, pts, nrm, boxes, rng_m)
            if not mask.any():
                continue
            overlap = float((mask & taken_mask).sum() / int(mask.sum()))
            v0, v1 = _value_of(b.arm, r0), _value_of(b.arm, r1)
            out.append(
                {
                    "seed": b.seed,
                    "arm": b.arm,
                    "plan": plan_i,
                    "overlap_with_taken_view": overlap,
                    "value_before": v0,
                    "value_after": v1,
                    "value_ratio": (v1 / v0) if v0 > 0 else None,
                }
            )
    return out


def _key(row: dict[str, Any]) -> tuple[float, float, float]:
    p = row["position_m"]
    return (round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4))


# ------------------------------------------------------------------------------------------------ summaries
def _mean(xs: Iterable[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return float(np.mean(vals)) if vals else None


def summarize(
    per_world: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    redundancy: list[dict[str, Any]],
) -> dict[str, Any]:
    arms = sorted({r["arm"] for r in per_world})
    by_arm = {a: [r for r in per_world if r["arm"] == a] for a in arms}
    seeds = sorted({r["seed"] for r in per_world})

    means: dict[str, dict[str, Any]] = {}
    for a, rows in by_arm.items():
        views = [v for r in rows for v in r["views"]]
        means[a] = {
            "n_worlds": len(rows),
            "inspection_views": _mean(r["inspection_views"] for r in rows),
            "hidden_state_error_improvement": _mean(r["hidden_state_error_improvement"] for r in rows),
            "target_observed": _mean(r["target_observed"] for r in rows),
            "patch_max_visible_fraction": _mean(r["patch_max_visible_fraction"] for r in rows),
            "mean_patch_cells_visible_fraction": _mean(r["mean_patch_cells_visible_fraction"] for r in rows),
            "per_view_patch_cells_visible_fraction": _mean(v["patch_cells_visible_fraction"] for v in views),
            "per_view_realised_patch_cells_visible_fraction": _mean(
                v["realised_patch_cells_visible_fraction"] for v in views
            ),
            "per_view_closest_approach_m": _mean(v["closest_approach_m"] for v in views),
            "per_view_reached_fraction": _mean(
                float(v["closest_approach_m"] <= 0.5) for v in views if v["closest_approach_m"] is not None
            ),
            "per_view_new_cell_fraction": _mean(v["new_cell_fraction"] for v in views),
            "per_view_cell_overlap": _mean(v["cell_overlap_with_earlier_views"] for v in views),
            "view_pair_angle_deg": _mean(x for r in rows for x in r["view_pair_angles_deg"]),
            "read_surface_fraction_proxy": _mean(r["read_surface_fraction_proxy"] for r in rows),
            "U_O": _mean(r["U_O"] for r in rows),
            "U_C": _mean(r["U_C"] for r in rows),
            "U_E": _mean(r["U_E"] for r in rows),
            "corrosion_abs_error_m": _mean(r["corrosion_abs_error_m"] for r in rows),
            "crack_abs_error_m": _mean(r["crack_abs_error_m"] for r in rows),
            "travel_m": _mean(r["travel_m"] for r in rows),
            "energy_j": _mean(r["energy_j"] for r in rows),
            "collisions": _mean(r["collisions"] for r in rows),
            "detected_worlds": sum(1 for r in rows if r["detection_view_index"] is not None),
            "detection_view_index": _mean(r["detection_view_index"] for r in rows),
            "detection_t_s": _mean(r["detection_t_s"] for r in rows),
            "post_detection_views": _mean(
                r["post_detection_views"] for r in rows if r["detection_view_index"] is not None
            ),
            "post_detection_hse_gain": _mean(r["post_detection_hse_gain"] for r in rows),
            "hse_improvement_at_detection": _mean(r["hse_improvement_at_detection"] for r in rows),
            "new_cell_fraction_by_view": [
                _mean(v["new_cell_fraction"] for v in views if v["index"] == k) for k in range(4)
            ],
            "patch_fraction_by_view": [
                _mean(v["patch_cells_visible_fraction"] for v in views if v["index"] == k) for k in range(4)
            ],
        }

    out: dict[str, Any] = {"arms": arms, "n_worlds": len(seeds), "means": means}

    if len(arms) == 2:
        a, c = "PRODUCTION", next(x for x in arms if x != "PRODUCTION")
        lookup = {(r["arm"], r["seed"]): r for r in per_world}
        paired_seeds = [s for s in seeds if (a, s) in lookup and (c, s) in lookup]
        out["paired"] = {
            m: bootstrap_paired([lookup[(a, s)][m] - lookup[(c, s)][m] for s in paired_seeds])
            for m in (
                "hidden_state_error_improvement",
                "mean_patch_cells_visible_fraction",
                "patch_max_visible_fraction",
                "inspection_views",
                "read_surface_fraction_proxy",
                "target_observed",
                "detected",
                "mean_realised_patch_cells_visible_fraction",
            )
            if all(lookup[(a, s)][m] is not None and lookup[(c, s)][m] is not None for s in paired_seeds)
        }
        det = {x: [lookup[(x, s)]["detection_t_s"] for s in paired_seeds] for x in (a, c)}
        both = [
            s
            for s in paired_seeds
            if lookup[(a, s)]["detection_t_s"] is not None and lookup[(c, s)]["detection_t_s"] is not None
        ]
        out["detection"] = {
            "worlds_detected": {x: sum(1 for v in det[x] if v is not None) for x in (a, c)},
            "both_detected": len(both),
            "production_earlier": sum(
                1 for s in both if lookup[(a, s)]["detection_t_s"] < lookup[(c, s)]["detection_t_s"] - 1e-9
            ),
            "coverage_earlier": sum(
                1 for s in both if lookup[(c, s)]["detection_t_s"] < lookup[(a, s)]["detection_t_s"] - 1e-9
            ),
            "mean_detection_t_s": {x: _mean(det[x]) for x in (a, c)},
            "production_only": sum(
                1
                for s in paired_seeds
                if lookup[(a, s)]["detection_t_s"] is not None and lookup[(c, s)]["detection_t_s"] is None
            ),
            "coverage_only": sum(
                1
                for s in paired_seeds
                if lookup[(c, s)]["detection_t_s"] is not None and lookup[(a, s)]["detection_t_s"] is None
            ),
            "neither": sum(
                1
                for s in paired_seeds
                if lookup[(a, s)]["detection_t_s"] is None and lookup[(c, s)]["detection_t_s"] is None
            ),
        }

    out["calibration"] = calibration(candidates, per_world)
    out["redundancy"] = redundancy_summary(redundancy)
    out["mechanism"] = {a: mechanism(by_arm[a]) for a in arms}
    return out


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Where the chain from a chosen view to a belief revision actually breaks."""
    views = [v for r in rows for v in r["views"]]
    good = [v for v in views if v["patch_cells_visible_fraction"] > 0.5]
    realised_good = [v for v in views if (v["realised_patch_cells_visible_fraction"] or 0.0) > 0.5]
    detected = [r for r in rows if r["detection_view_index"] is not None]
    missed = [r for r in rows if r["detection_view_index"] is None]
    per_world_realised = [
        _mean(v["realised_patch_cells_visible_fraction"] for v in r["views"]) or 0.0 for r in rows
    ]
    return {
        "views": len(views),
        "views_with_a_direct_revision": _mean(float(v["direct_revisions_in_window"] > 0) for v in views),
        "commanded_patch_gt_half": len(good),
        "commanded_patch_gt_half_to_revision": _mean(
            float(v["direct_revisions_in_window"] > 0) for v in good
        ),
        "realised_patch_gt_half": len(realised_good),
        "realised_patch_gt_half_to_revision": _mean(
            float(v["direct_revisions_in_window"] > 0) for v in realised_good
        ),
        "views_reached_within_0_5m": _mean(float((v["closest_approach_m"] or 9.9) <= 0.5) for v in views),
        "median_closest_approach_m": float(
            np.median([v["closest_approach_m"] for v in views if v["closest_approach_m"] is not None])
        )
        if views
        else None,
        "worlds_by_view_count": {
            str(k): sum(1 for r in rows if r["inspection_views"] == k) for k in range(5)
        },
        "worlds_detected": len(detected),
        "hse_when_detected": _mean(r["hidden_state_error_improvement"] for r in detected),
        "hse_when_not_detected": _mean(r["hidden_state_error_improvement"] for r in missed),
        "detection_view_index_hist": {
            str(k): sum(1 for r in detected if r["detection_view_index"] == k) for k in range(4)
        },
        "post_detection_view_hist": {
            str(k): sum(1 for r in detected if r["post_detection_views"] == k) for k in range(4)
        },
        "spearman_commanded_patch_vs_hse": spearman(
            [r["mean_patch_cells_visible_fraction"] for r in rows],
            [r["hidden_state_error_improvement"] for r in rows],
        ),
        "spearman_realised_patch_vs_hse": spearman(
            per_world_realised, [r["hidden_state_error_improvement"] for r in rows]
        ),
        "spearman_patch_max_vs_hse": spearman(
            [r["patch_max_visible_fraction"] for r in rows],
            [r["hidden_state_error_improvement"] for r in rows],
        ),
    }


def calibration(candidates: list[dict[str, Any]], per_world: list[dict[str, Any]]) -> dict[str, Any]:
    arms = sorted({c["arm"] for c in candidates})
    out: dict[str, Any] = {}
    for a in arms:
        rows = [c for c in candidates if c["arm"] == a]
        plans = sorted({(c["seed"], c["plan"]) for c in rows})
        rhos: list[float] = []
        rhos_value: list[float] = []
        regrets: list[float] = []
        recall5: list[float] = []
        recall10: list[float] = []
        chosen_f: list[float] = []
        best_f: list[float] = []
        mean_f: list[float] = []
        useful_plans = 0
        for seed, plan in plans:
            rs = [c for c in rows if c["seed"] == seed and c["plan"] == plan]
            rho = spearman([c["score"] for c in rs], [c["oracle_patch_fraction"] for c in rs])
            if rho is not None:
                rhos.append(rho)
            rho_v = spearman([c["value"] for c in rs], [c["oracle_patch_fraction"] for c in rs])
            if rho_v is not None:
                rhos_value.append(rho_v)
            mean_f.append(float(np.mean([c["oracle_patch_fraction"] for c in rs])))
            best_f.append(max(c["oracle_patch_fraction"] for c in rs))
            picked = next((c for c in rs if c["chosen"]), None)
            if picked is not None:
                chosen_f.append(picked["oracle_patch_fraction"])
            if picked is not None:
                regrets.append(best_f[-1] - picked["oracle_patch_fraction"])
            useful = [c for c in rs if c["oracle_patch_fraction"] > 0.0]
            if useful:
                useful_plans += 1
                order = sorted(rs, key=lambda c: -c["score"])
                for k, bucket in ((5, recall5), (10, recall10)):
                    hit = sum(1 for c in order[:k] if c["oracle_patch_fraction"] > 0.0)
                    bucket.append(hit / min(k, len(useful)))
        bins: list[dict[str, Any]] = []
        for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)):
            sel = [c for c in rows if lo <= _pct(c, rows) < hi or (hi == 1.0 and _pct(c, rows) == 1.0)]
            bins.append(
                {
                    "score_percentile_bin": [lo, hi],
                    "n": len(sel),
                    "mean_oracle_patch_fraction": _mean(c["oracle_patch_fraction"] for c in sel),
                    "fraction_useful": (
                        sum(1 for c in sel if c["oracle_patch_fraction"] > 0.0) / len(sel) if sel else None
                    ),
                }
            )
        world_rows = [r for r in per_world if r["arm"] == a]
        chosen_value = [
            (v["predicted_value"], r["hidden_state_error_improvement"])
            for r in world_rows
            for v in r["views"]
            if v["predicted_value"] is not None
        ]
        out[a] = {
            "n_plans": len(plans),
            "plans_with_a_useful_candidate": useful_plans,
            "spearman_score_vs_oracle_patch_fraction_mean": _mean(rhos),
            "spearman_value_vs_oracle_patch_fraction_mean": _mean(rhos_value),
            "spearman_n_plans": len(rhos),
            "oracle_patch_fraction_chosen": _mean(chosen_f),
            "oracle_patch_fraction_best_available": _mean(best_f),
            "oracle_patch_fraction_plan_mean": _mean(mean_f),
            "top1_regret_mean": _mean(regrets),
            "top1_regret_zero_fraction": (
                sum(1 for r in regrets if r <= 1e-9) / len(regrets) if regrets else None
            ),
            "top5_recall_of_useful": _mean(recall5),
            "top10_recall_of_useful": _mean(recall10),
            "score_bins": bins,
            "spearman_chosen_value_vs_world_hse": spearman(
                [x for x, _ in chosen_value], [y for _, y in chosen_value]
            ),
            "n_chosen_views": len(chosen_value),
        }
    return out


def _pct(c: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    key = (c["seed"], c["plan"])
    peers = [r["score"] for r in rows if (r["seed"], r["plan"]) == key]
    if len(peers) < 2:
        return 1.0
    return float(sum(1 for p in peers if p < c["score"]) / (len(peers) - 1))


def redundancy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == a and r["value_ratio"] is not None]
        high = [r for r in rs if r["overlap_with_taken_view"] >= 0.5]
        low = [r for r in rs if r["overlap_with_taken_view"] <= 0.1]
        out[a] = {
            "n": len(rs),
            "n_high_overlap": len(high),
            "n_low_overlap": len(low),
            "value_ratio_high_overlap": _mean(r["value_ratio"] for r in high),
            "value_ratio_low_overlap": _mean(r["value_ratio"] for r in low),
            "spearman_overlap_vs_value_ratio": spearman(
                [r["overlap_with_taken_view"] for r in rs], [r["value_ratio"] for r in rs]
            ),
        }
    return out


# ------------------------------------------------------------------------------------------------ driver
def run(dirs: Sequence[Path], out_dir: Path, tag: str) -> dict[str, Any]:
    bundles = sorted(p for d in dirs for p in d.iterdir() if p.is_dir())
    per_world, candidates, redundancy, per_view = [], [], [], []
    for p in bundles:
        b = load_bundle(p)
        rec = analyse_bundle(b)
        per_world.append(rec)
        for v in rec["views"]:
            per_view.append({"seed": b.seed, "arm": b.arm, **v})
        candidates.extend(candidate_report(b))
        redundancy.extend(redundancy_probe(b))
    summary = summarize(per_world, candidates, redundancy)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"per_world_{tag}.json").write_text(
        json.dumps(per_world, indent=1, sort_keys=True), encoding="utf-8"
    )
    (out_dir / f"per_view_{tag}.json").write_text(
        json.dumps(per_view, indent=1, sort_keys=True), encoding="utf-8"
    )
    (out_dir / f"candidates_{tag}.json").write_text(
        json.dumps(candidates, indent=1, sort_keys=True), encoding="utf-8"
    )
    (out_dir / f"redundancy_{tag}.json").write_text(
        json.dumps(redundancy, indent=1, sort_keys=True), encoding="utf-8"
    )
    (out_dir / f"summary_{tag}.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run2-dir", type=Path, default=RUN2_DIR)
    ap.add_argument("--run1-dir", type=Path, default=RUN1_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--runs", default="run2", help="run2, run1 or both")
    args = ap.parse_args()
    if args.runs in ("run2", "both"):
        s = run([args.run2_dir], args.out_dir, "run2")
        print(json.dumps(s["means"], indent=1, sort_keys=True))
    if args.runs in ("run1", "both"):
        s = run([args.run1_dir], args.out_dir, "run1")
        print(json.dumps(s["means"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
