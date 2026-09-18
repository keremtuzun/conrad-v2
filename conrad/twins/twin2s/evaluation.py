"""Truth-plane evaluation helpers for Model2S (ch14 metrics). TRUTH PLANE: never import from inference code.

UC = P(high-confidence spatial claim | insufficient evidence) is the main 2S metric.
The confidence threshold is a configurable evaluation default, not a validated operating point.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

import numpy as np
from numpy.typing import ArrayLike, NDArray

from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.frames import Pose
from conrad.schemas.world import SensorSpec
from conrad.twins.twin2s.coverage import CoverageAccumulator
from conrad.twins.twin2s.sdf import Arr
from conrad.twins.twin2s.twin import Twin2S
from conrad.twins.twin2s.visibility import sample_surface
from conrad.twins.twin2s.world import SpatialWorld

DEFAULT_CONFIDENCE_THRESHOLD = 0.9
"""Evaluation default: a claim is 'confident' when max(p, 1 - p) >= 0.9. Configurable."""


def occupancy_truth_at(world: SpatialWorld | Twin2S, points_m: ArrayLike) -> NDArray[np.bool_]:
    """True occupancy at WORLD points (sdf < 0) at the world's current physical time."""
    w = world.world if isinstance(world, Twin2S) else world
    return w.occupied(np.atleast_2d(np.asarray(points_m, dtype=np.float64)))


def visible_fraction(
    twin: Twin2S,
    entity_id: UUID,
    sensor: SensorSpec,
    true_robot_poses: Sequence[Pose],
    n_samples: int = 200,
    seed: int = 0,
) -> float:
    """Fraction of the entity's exposed surface OBSERVED (per criteria) from at least one of the poses."""
    pts, nrm = sample_surface(
        twin.world, twin.world.index_of(entity_id), n_samples, np.random.default_rng(seed)
    )
    if len(pts) == 0:
        return 0.0
    acc = CoverageAccumulator(pts, twin.cfg.observed)
    for k, pose in enumerate(true_robot_poses):
        acc.add(twin.visibility(sensor, pose, pts, nrm), float(k), sensor.modality)
    return acc.coverage()


def _status_array(status: Sequence[KnowledgeStatus | str]) -> NDArray[np.str_]:
    return np.array([KnowledgeStatus(s).value for s in status])


def unsupported_confidence(
    pred_occupancy_prob: ArrayLike,
    pred_status: Sequence[KnowledgeStatus | str],
    never_observed: ArrayLike,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, float]:
    """Rates of confident claims over query cells that were NEVER observed (truth-plane mask).

    - ``uc_rate``: fraction with max(p, 1-p) >= threshold (confidently occupied OR confidently free).
    - ``uc_rate_not_unknown``: confident AND status is not UNKNOWN (the model also asserts knowledge).
    - ``claimed_observed_rate``: status OBSERVED in a never-observed cell (always a violation).
    ``n_cells`` = number of never-observed cells; rates are NaN when it is 0 (no fabricated zero).
    """
    p = np.asarray(pred_occupancy_prob, dtype=np.float64)
    st = _status_array(pred_status)
    hidden = np.asarray(never_observed, dtype=bool)
    if not (p.shape == st.shape == hidden.shape):
        raise ValueError("prob, status and mask must have the same shape")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("occupancy probabilities must lie in [0, 1]")
    n = int(hidden.sum())
    if n == 0:
        nan = float("nan")
        return {"n_cells": 0.0, "uc_rate": nan, "uc_rate_not_unknown": nan, "claimed_observed_rate": nan}
    conf = np.maximum(p, 1.0 - p)[hidden] >= confidence_threshold
    sh = st[hidden]
    return {
        "n_cells": float(n),
        "uc_rate": float(conf.mean()),
        "uc_rate_not_unknown": float((conf & (sh != KnowledgeStatus.UNKNOWN.value)).mean()),
        "claimed_observed_rate": float((sh == KnowledgeStatus.OBSERVED.value).mean()),
    }


def occupancy_error_split(
    pred_occupancy_prob: ArrayLike, true_occupied: ArrayLike, observed: ArrayLike
) -> dict[str, float]:
    """Brier error split into E_observed and E_hidden (so easy visible geometry cannot hide failures)."""
    p = np.asarray(pred_occupancy_prob, dtype=np.float64)
    err = (p - np.asarray(true_occupied, dtype=np.float64)) ** 2
    obs = np.asarray(observed, dtype=bool)
    return {
        "E_observed": float(err[obs].mean()) if obs.any() else float("nan"),
        "E_hidden": float(err[~obs].mean()) if (~obs).any() else float("nan"),
    }


def coverage_error(predicted_coverage: float, true_coverage: float) -> float:
    """E_coverage = |V_hat - V_true|."""
    return abs(float(predicted_coverage) - float(true_coverage))


def occupancy_iou(pred_occupied: ArrayLike, true_occupied: ArrayLike) -> float:
    a, b = np.asarray(pred_occupied, dtype=bool), np.asarray(true_occupied, dtype=bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else float("nan")


def query_points_between(lo: Arr, hi: Arr, spacing_m: float) -> Arr:
    """Regular WORLD query grid (cell centres) for evaluation."""
    axes = [np.arange(lo[k] + 0.5 * spacing_m, hi[k], spacing_m) for k in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
