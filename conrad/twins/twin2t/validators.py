"""Sanity validators and reality-gap hooks for Twin2T (ch11 Validating Twin 2T, Reality-gap measurement).

Validators check invariants of generated sequences: bounds, monotone material loss absent intervention,
crack growth only above the dK threshold, repair/replacement resets, mechanism validity masks.
Reality-gap hooks compare simulated distributions against a SUPPLIED real dataset; without one they
return NOT_EVALUABLE (never a fabricated number).

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np
from scipy import stats

from conrad.twins.twin2t.mcde import MCDEStepResult
from conrad.twins.twin2t.state import STATE_DIMENSIONS, ComponentRuntime

_IDX = {d: i for i, d in enumerate(STATE_DIMENSIONS)}
_MONOTONE = STATE_DIMENSIONS
_TOL = 1e-15


@dataclass
class ValidationReport:
    violations: list[str] = field(default_factory=list)
    checks: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.violations

    def _count(self, name: str) -> None:
        self.checks[name] = self.checks.get(name, 0) + 1


def validate_sequence(
    results: Sequence[MCDEStepResult], runtimes: dict[UUID, ComponentRuntime]
) -> ValidationReport:
    rep = ValidationReport()
    for res in results:
        for eid, rec in res.records.items():
            rt = runtimes[eid]
            wall = rt.effective_wall_m
            after = rec.after
            tag = f"t={res.time_s:.0f}s {eid}"
            # bounds / conservation-style limits
            rep._count("bounds")
            if min(after) < -_TOL:
                rep.violations.append(f"{tag}: negative state {after}")
            for d in ("corrosion_depth_m", "crack_depth_m"):
                if after[_IDX[d]] > wall + 1e-12:
                    rep.violations.append(f"{tag}: {d} exceeds wall thickness")
            for d in ("corrosion_area_fraction", "coating_breakdown_fraction"):
                if after[_IDX[d]] > 1.0 + 1e-12:
                    rep.violations.append(f"{tag}: {d} > 1")
            # validity masks
            rep._count("masks")
            for d, valid in zip(STATE_DIMENSIONS, rec.mask, strict=True):
                if not valid and after[_IDX[d]] != 0.0:
                    rep.violations.append(f"{tag}: masked dimension {d} is non-zero")
            # monotone absent intervention
            if not rec.intervened:
                rep._count("monotone")
                for d in _MONOTONE:
                    if after[_IDX[d]] < rec.before[_IDX[d]] - _TOL:
                        rep.violations.append(f"{tag}: {d} decreased without intervention")
            # crack growth only above threshold (tick without events)
            grew = after[_IDX["crack_length_m"]] > rec.before[_IDX["crack_length_m"]] + _TOL
            if grew and not rec.event_applied:
                rep._count("paris_threshold")
                if rec.delta_k < rec.delta_k_threshold:
                    rep.violations.append(f"{tag}: crack grew below threshold dK={rec.delta_k:.3f}")
            # repair / replacement resets
            for ev in rec.events:
                _check_reset(rep, tag, ev)
    return rep


def _check_reset(rep: ValidationReport, tag: str, ev: dict[str, Any]) -> None:
    after = ev["state_after"]
    if ev["event_type"] == "REPLACEMENT":
        rep._count("replacement_reset")
        if any(v != 0.0 for v in after):
            rep.violations.append(f"{tag}: REPLACEMENT did not reset state")
    elif ev["event_type"] == "REPAIR":
        rep._count("repair_reset")
        p = ev["parameters"]
        if float(p.get("restore_fraction", 1.0)) == 1.0 and after[_IDX["corrosion_depth_m"]] != 0.0:
            rep.violations.append(f"{tag}: full REPAIR left corrosion depth")
        if p.get("remove_cracks", True) and after[_IDX["crack_length_m"]] != 0.0:
            rep.violations.append(f"{tag}: REPAIR left a crack")
        if after[_IDX["corrosion_depth_m"]] > ev["state_before"][_IDX["corrosion_depth_m"]] + _TOL:
            rep.violations.append(f"{tag}: REPAIR increased corrosion depth")


NOT_EVALUABLE = "NOT_EVALUABLE"
EVALUATED = "EVALUATED"


def reality_gap(
    simulated: np.ndarray,
    real: np.ndarray | None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """D(P_synthetic, P_real) per feature column: two-sample KS statistic and 1-Wasserstein distance.

    ``real`` must come from an actual dataset supplied by the caller. None/empty -> NOT_EVALUABLE.
    """
    if real is None or np.asarray(real).size == 0:
        return {"status": NOT_EVALUABLE, "reason": "no real dataset supplied", "metrics": {}}
    sim = np.asarray(simulated, dtype=np.float64)
    ref = np.asarray(real, dtype=np.float64)
    sim = sim.reshape(-1, 1) if sim.ndim == 1 else sim
    ref = ref.reshape(-1, 1) if ref.ndim == 1 else ref
    if sim.shape[1] != ref.shape[1]:
        raise ValueError(f"feature mismatch: simulated {sim.shape[1]} vs real {ref.shape[1]}")
    if sim.shape[0] < 2 or ref.shape[0] < 2:
        return {"status": NOT_EVALUABLE, "reason": "fewer than 2 samples", "metrics": {}}
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(sim.shape[1])]
    metrics: dict[str, dict[str, float]] = {}
    for j, name in enumerate(names):
        ks = stats.ks_2samp(sim[:, j], ref[:, j])
        metrics[name] = {
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "wasserstein_1": float(stats.wasserstein_distance(sim[:, j], ref[:, j])),
        }
    return {"status": EVALUATED, "n_sim": int(sim.shape[0]), "n_real": int(ref.shape[0]), "metrics": metrics}
