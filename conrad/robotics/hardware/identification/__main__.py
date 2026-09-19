"""Run identification on a log manifest and write the report.

    python -m conrad.robotics.hardware.identification --manifest logs/manifest.yaml \\
        --mass-kg 11.48 --yaw-inertia 0.16 --out artifacts/identification/report.json [--envelope-rmse 0.02]

Each experiment kind present in the identification split is fitted; every kind present in the validation
split is scored on held-out logs only. Mass/inertia are the MEASURED rigid-body values from characterization.

Protocol deliveries (docs/IDENTIFICATION_LOG_PROTOCOL.md)::

    python -m conrad.robotics.hardware.identification --manifest delivery/manifest.yaml --intake-only         --out artifacts/identification/intake.json          # exit 0 accepted, 3 rejected
    python -m conrad.robotics.hardware.identification --manifest delivery/manifest.yaml --protocol         --robot-config configs/robot/<measured>.yaml --out artifacts/identification/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from conrad.robotics.hardware.identification.dataset import ExperimentKind
from conrad.robotics.hardware.identification.experiments import (
    identify_axis,
    identify_displaced_volume,
    identify_thruster,
)
from conrad.robotics.hardware.identification.fitting import FitResult
from conrad.robotics.hardware.identification.intake import check_delivery
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
    ap.add_argument("--mass-kg", type=float, default=None, help="required unless --intake-only/--protocol")
    ap.add_argument("--intake-only", action="store_true", help="check a protocol delivery and stop")
    ap.add_argument("--protocol", action="store_true", help="intake + derive + fit + validate a delivery")
    ap.add_argument("--robot-config", default=None, help="RobotConfig with measured geometry (--protocol)")
    ap.add_argument("--pipeline-config", default=None, help="YAML PipelineConfig (--protocol)")
    ap.add_argument(
        "--yaw-inertia", type=float, default=None, help="kg*m^2, required for E_YAW_ROTATION logs"
    )
    ap.add_argument(
        "--envelope-rmse", type=float, default=None, help="held-out RMSE limit (same units as signal)"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.intake_only:
        intake = check_delivery(args.manifest)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(intake.model_dump(mode="json"), indent=2), encoding="utf-8")
        for f in intake.findings:
            if f.severity != "INFO":
                print(f"{f.severity} {f.code} {f.trajectory_id or '-'}: {f.message}", file=sys.stderr)
        print(f"{'ACCEPTED' if intake.accepted else 'REJECTED'}: {len(intake.failures())} failures -> {out}")
        return 0 if intake.accepted else 3
    if args.protocol:
        from conrad.robotics.hardware.config import load_robot_config
        from conrad.robotics.hardware.identification.pipeline import (
            IntakeRejectedError,
            PipelineConfig,
            run_pipeline,
        )

        if args.robot_config is None:
            print("--protocol needs --robot-config", file=sys.stderr)
            return 2
        if args.envelope_rmse is not None:
            print("protocol runs declare envelope_rmse per fit in --pipeline-config", file=sys.stderr)
            return 2
        pcfg = PipelineConfig()
        if args.pipeline_config:
            pcfg = PipelineConfig.model_validate(
                yaml.safe_load(Path(args.pipeline_config).read_text("utf-8"))
            )
        try:
            result = run_pipeline(args.manifest, load_robot_config(args.robot_config), pcfg)
        except IntakeRejectedError as exc:
            print(str(exc), file=sys.stderr)
            return 3
        path = result.report.write(args.out)
        print(f"{result.report.status.value}: {len(result.report.fits)} fits -> {path}")
        return 0
    if args.mass_kg is None:
        print("--mass-kg is required", file=sys.stderr)
        return 2

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
