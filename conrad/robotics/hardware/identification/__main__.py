"""Run identification on a log manifest and write the report.

    python -m conrad.robotics.hardware.identification --manifest logs/manifest.yaml \\
        --mass-kg 11.48 --yaw-inertia 0.16 --out artifacts/identification/report.json [--envelope-rmse 0.02]

Each experiment kind present in the identification split is fitted; every kind present in the validation
split is scored on held-out logs only. Mass/inertia are the MEASURED rigid-body values from characterization.
"""

from __future__ import annotations

import argparse
import sys

from conrad.robotics.hardware.identification.dataset import ExperimentKind
from conrad.robotics.hardware.identification.experiments import (
    identify_axis,
    identify_displaced_volume,
    identify_thruster,
)
from conrad.robotics.hardware.identification.fitting import FitResult
from conrad.robotics.hardware.identification.logio import load_manifest
from conrad.robotics.hardware.identification.report import build_report
from conrad.robotics.hardware.identification.validation import (
    ValidationResult,
    validate_axis,
    validate_thruster,
)

_AXES = (
    ExperimentKind.SURGE_ACCELERATION,
    ExperimentKind.SWAY,
    ExperimentKind.HEAVE,
    ExperimentKind.YAW_ROTATION,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m conrad.robotics.hardware.identification")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mass-kg", type=float, required=True)
    ap.add_argument(
        "--yaw-inertia", type=float, default=None, help="kg*m^2, required for E_YAW_ROTATION logs"
    )
    ap.add_argument(
        "--envelope-rmse", type=float, default=None, help="held-out RMSE limit (same units as signal)"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ds = load_manifest(args.manifest)
    fits: dict[str, FitResult] = {}
    validations: dict[str, ValidationResult] = {}
    static = ds.by_kind(ExperimentKind.STATIC_TRIM)
    if static and all("net_weight_n" in s.signals for s in static):
        fits["A_displaced_volume"] = identify_displaced_volume(static, args.mass_kg)
    thr = ds.by_kind(ExperimentKind.THRUSTER_STEP)
    if thr:
        fit, dz = identify_thruster(thr)
        fits["B_thruster"] = fit
        held = ds.by_kind(ExperimentKind.THRUSTER_STEP, split="validation")
        if held:
            validations["B_thruster"] = validate_thruster(ds, held, fit, dz.value, args.envelope_rmse)
    for kind in _AXES:
        segs = ds.by_kind(kind)
        if not segs:
            continue
        inertia = args.yaw_inertia if kind is ExperimentKind.YAW_ROTATION else args.mass_kg
        if inertia is None:
            print(f"{kind.value}: --yaw-inertia is required", file=sys.stderr)
            return 2
        fits[kind.value] = identify_axis(segs, kind, inertia, fit_bias=kind is ExperimentKind.HEAVE)
        held = ds.by_kind(kind, split="validation")
        if held:
            validations[kind.value] = validate_axis(
                ds, held, kind, inertia, fits[kind.value], args.envelope_rmse
            )
    report = build_report(ds, fits, validations)
    path = report.write(args.out)
    print(f"{report.status.value}: {len(fits)} fits, {len(validations)} validations -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
