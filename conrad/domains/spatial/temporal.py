"""Spatial TBD for UAHSM: static geometry persists, dynamic cells decay toward ignorance. BELIEF PLANE.

Change detection lives in ``SparseLevel.apply_direct`` (contradicting re-observation -> change score U_C,
``dynamic`` once it exceeds the threshold). Prediction over PHYSICAL elapsed seconds:

- dynamic cells: log-odds decay by exp(-dt / tau), U_O grows by 1 - exp(-rate * dt), status PREDICTED,
  source TEMPORAL_PREDICTION (their direct evidence is kept, not claimed as support of the prediction);
- every other cell is left exactly as it was; its age (now - last update) simply grows.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math

import numpy as np

from conrad.domains.spatial.config import SpatialConfig
from conrad.domains.spatial.grid import SRC_TEMPORAL, SparseLevel
from conrad.domains.spatial.keys import FloatArr, IntArr
from conrad.schemas.timebase import NS_PER_S


def predict_level(level: SparseLevel, delta_t_s: float, cfg: SpatialConfig) -> IntArr:
    """Advance one level by ``delta_t_s`` seconds. Returns the codes of cells that were predicted."""
    if delta_t_s < 0:
        raise ValueError("delta_t_s must be >= 0 (physical elapsed time)")
    tc = cfg.temporal
    rows = np.nonzero(level.b["dynamic"][: level.n])[0]
    if rows.size == 0 or delta_t_s == 0:
        return np.zeros(0, dtype=np.int64)
    level.f["lo"][rows] *= math.exp(-delta_t_s / tc.dynamic_tau_s)
    grow = 1.0 - math.exp(-tc.dynamic_uncertainty_rate_per_s * delta_t_s)
    level.f["pred_uo"][rows] = 1.0 - (1.0 - level.f["pred_uo"][rows]) * (1.0 - grow)
    level.b["predicted"][rows] = True
    level.src[rows] = SRC_TEMPORAL
    return np.asarray(level.codes[rows], dtype=np.int64)


def age_s(level: SparseLevel, rows: IntArr, now_ns: int) -> FloatArr:
    """Seconds since the last direct update (NaN for cells never directly updated)."""
    last = level.last_ns[rows]
    return np.where(level.f["w"][rows] > 0, (now_ns - last) / NS_PER_S, np.nan)
