"""MissionTruthRecorder: the evaluation-only truth record of one mission run. TRUTH PLANE.

Written under ``truth/`` in the run directory. Nothing in here is ever handed to the deployment side; the
evaluator (``conrad.orchestration.evaluation.evaluate_run``) reads it from disk after the run.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from conrad.sim.kernel import TruthAccess
from conrad.twins.twin2t import Twin2T

TRUTH_FILE = "truth/truth_record.json"


class MissionTruthRecorder:
    def __init__(self, meta: dict[str, Any]) -> None:
        self.meta = dict(meta)
        self.series: dict[str, list[dict[str, Any]]] = {}
        self.target_states: dict[str, dict[str, Any]] = {}
        self.trajectory: list[list[float]] = []
        self._last_traj_s = -1e9

    def record(self, kind: str, payload: dict[str, Any]) -> None:
        self.series.setdefault(kind, []).append(payload)

    def component_state(self, twin: Twin2T, world_id: UUID) -> dict[str, Any]:
        arr, mask = twin.truth_arrays()
        i = list(twin.component_ids).index(world_id)
        names = (
            "corrosion_depth_m",
            "corrosion_area_fraction",
            "coating_breakdown_fraction",
            "crack_length_m",
            "crack_depth_m",
        )
        return {n: (float(arr[i, k]) if mask[i, k] else None) for k, n in enumerate(names)}

    def snapshot_target(self, label: str, twin: Twin2T, world_id: UUID, t_s: float) -> None:
        self.target_states[label] = {"t_s": t_s, **self.component_state(twin, world_id)}

    def track(self, truth: TruthAccess, period_s: float = 1.0) -> None:
        s = truth.true_state()
        if s.t_s - self._last_traj_s + 1e-9 < period_s:
            return
        self._last_traj_s = s.t_s
        p, q = s.position_world_m, s.orientation_wxyz
        self.trajectory.append([round(s.t_s, 3), *(float(v) for v in p), *(float(v) for v in q)])

    def finish(self, truth: TruthAccess, run_dir: Path) -> dict[str, Any]:
        body = {
            "meta": self.meta,
            "target_states": self.target_states,
            "series": self.series,
            "trajectory": self.trajectory,
            "vehicle": {
                "collisions": truth.collision_count,
                "min_clearance_m": _finite(truth.min_clearance_m),
                "energy_used_j": float(truth.energy_used_j),
            },
            "evaluation_only": True,
        }
        path = run_dir / TRUTH_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=1, sort_keys=True, default=str), encoding="utf-8")
        return body


def _finite(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None
