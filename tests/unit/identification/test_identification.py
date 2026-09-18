"""Identification tooling recovers KNOWN parameters from SYNTHETIC logs; the split cannot leak.

Every "true" value below is a test fixture, not a property of the vehicle.
"""

from __future__ import annotations

import numpy as np
import pytest

from conrad.robotics.hardware.characterization import Category
from conrad.robotics.hardware.identification import (
    REAL_IDENTIFICATION_STATUS,
    ExperimentKind,
    IdentificationDataset,
    IdentificationError,
    IdentificationReport,
    ParameterMapping,
    ReportStatus,
    SplitLeakageError,
    SyntheticDataError,
    build_report,
    identify_axis,
    identify_displaced_volume,
    identify_station_keeping_current,
    identify_thruster,
    identify_trim,
    to_characterization_records,
    validate_axis,
    validate_thruster,
)
from conrad.robotics.hardware.identification import synthetic as syn
from conrad.robotics.hardware.identification.dataset import LogSegment
from conrad.robotics.hardware.identification.models import G0
from conrad.schemas.robot import SourceKind

TRUE = {
    "mass": 11.5,
    "volume": 0.0113,
    "rho": 1000.0,
    "z_bg": 0.02,
    "x_bg": 0.004,
    "k_fwd": 40.0,
    "k_rev": 30.0,
    "deadzone": 0.07,
    "tau": 0.15,
    "latency": 0.03,
    "added": 5.5,
    "d1": 4.0,
    "d2": 18.2,
    "current": 0.3,
}


def _within(fit_value: float, sigma: float | None, true: float, rel: float) -> None:
    assert sigma is not None and sigma > 0
    assert abs(fit_value - true) <= max(rel * abs(true), 4.0 * sigma), (fit_value, sigma, true)


def test_static_buoyancy_and_trim_recovered() -> None:
    rng = np.random.default_rng(1)
    logs = [
        syn.static_buoyancy_log(f"A{i}", rng, TRUE["mass"], TRUE["volume"], TRUE["rho"], 200, 0.2)
        for i in range(2)
    ]
    v = identify_displaced_volume(logs, TRUE["mass"]).get("displaced_volume_m3")
    _within(v.value, v.sigma, TRUE["volume"], 0.01)
    buoyancy = TRUE["rho"] * G0 * TRUE["volume"]
    tilt = syn.tilt_log(
        "A-tilt", rng, buoyancy, TRUE["z_bg"], TRUE["x_bg"], np.linspace(-1.2, 1.2, 25), 0.002
    )
    trim = identify_trim([tilt], buoyancy)
    _within(trim.value("z_bg_m"), trim.get("z_bg_m").sigma, TRUE["z_bg"], 0.05)
    _within(trim.value("x_bg_m"), trim.get("x_bg_m").sigma, TRUE["x_bg"], 0.15)
    no_rho = syn.static_buoyancy_log("A9", rng, 11.5, 0.0113, 1000.0, 10, 0.2)
    no_rho.metadata.clear()
    with pytest.raises(IdentificationError, match="water_density"):
        identify_displaced_volume([no_rho], 11.5)


def _thruster_logs(rng: np.random.Generator, prefix: str, levels: tuple[float, ...]):
    return syn.thruster_step_log(
        prefix,
        rng,
        levels,
        TRUE["k_fwd"],
        TRUE["k_rev"],
        TRUE["deadzone"],
        TRUE["tau"],
        TRUE["latency"],
        dwell_s=0.8,
        dt=0.005,
    )


def test_thruster_step_response_recovered_and_validated_on_held_out_steps() -> None:
    rng = np.random.default_rng(2)
    ident = _thruster_logs(rng, "B-id", (0.03, 0.06, 0.08, 0.3, 0.6, -0.4, -0.8))
    held = _thruster_logs(rng, "B-val", (0.5, -0.5, 0.9))
    ds = IdentificationDataset("thr-split-1", [ident], [held])
    fit, dz = identify_thruster(ds.identification)
    for name, key, rel in (
        ("k_fwd", "k_fwd", 0.02),
        ("k_rev", "k_rev", 0.02),
        ("time_constant_s", "tau", 0.05),
        ("latency_s", "latency", 0.1),
    ):
        p = fit.get(name)
        _within(p.value, p.sigma, TRUE[key], rel)
    assert dz.lower == pytest.approx(0.06) and dz.upper == pytest.approx(0.08)
    assert dz.lower < TRUE["deadzone"] < dz.upper
    val = validate_thruster(ds, ds.validation, fit, dz.value, envelope_rmse=0.2)
    assert val.within_envelope and val.pooled_rmse < 0.2
    with pytest.raises(SplitLeakageError, match="identification data"):
        validate_thruster(ds, ds.identification, fit, dz.value)
    only_fwd = _thruster_logs(rng, "B-fwd", (0.3, 0.6))
    with pytest.raises(IdentificationError, match="reverse"):
        identify_thruster([only_fwd])


def test_surge_drag_and_added_mass_recovered_heave_bias_and_yaw() -> None:
    rng = np.random.default_rng(3)
    dt = 0.02
    ident = [
        syn.axis_log(
            f"C-id{i}",
            rng,
            ExperimentKind.SURGE_ACCELERATION,
            TRUE["mass"],
            TRUE["added"],
            TRUE["d1"],
            TRUE["d2"],
            syn.pulse_profile(amps, 3.0, 3.0, dt),
            dt,
        )
        for i, amps in enumerate([(10.0, 25.0), (40.0, 5.0)])
    ]
    held = [
        syn.axis_log(
            "C-val",
            rng,
            ExperimentKind.SURGE_ACCELERATION,
            TRUE["mass"],
            TRUE["added"],
            TRUE["d1"],
            TRUE["d2"],
            syn.pulse_profile((18.0, 30.0), 2.0, 2.5, dt),
            dt,
        )
    ]
    ds = IdentificationDataset("surge-split", ident, held)
    fit = identify_axis(ds.identification, ExperimentKind.SURGE_ACCELERATION, TRUE["mass"])
    _within(fit.value("added_inertia"), fit.get("added_inertia").sigma, TRUE["added"], 0.05)
    _within(fit.value("linear_damping"), fit.get("linear_damping").sigma, TRUE["d1"], 0.1)
    _within(fit.value("quadratic_damping"), fit.get("quadratic_damping").sigma, TRUE["d2"], 0.05)
    val = validate_axis(
        ds, ds.validation, ExperimentKind.SURGE_ACCELERATION, TRUE["mass"], fit, envelope_rmse=0.01
    )
    assert val.within_envelope

    heave = [
        syn.axis_log(
            f"D-h{i}",
            rng,
            ExperimentKind.HEAVE,
            TRUE["mass"],
            14.6,
            5.2,
            37.0,
            syn.pulse_profile(amps, 3.0, 3.0, dt),
            dt,
            bias=-1.5,
        )
        for i, amps in enumerate([(15.0, -10.0), (30.0, -25.0)])
    ]
    hf = identify_axis(heave, ExperimentKind.HEAVE, TRUE["mass"], fit_bias=True)
    _within(hf.value("bias"), hf.get("bias").sigma, -1.5, 0.1)
    _within(hf.value("added_inertia"), hf.get("added_inertia").sigma, 14.6, 0.1)

    yaw = [
        syn.axis_log(
            f"E-y{i}",
            rng,
            ExperimentKind.YAW_ROTATION,
            0.16,
            0.12,
            0.07,
            1.55,
            syn.pulse_profile(amps, 3.0, 3.0, dt),
            dt,
            noise=0.002,
            force_units="N*m",
            velocity_units="rad/s",
        )
        for i, amps in enumerate([(0.5, 2.0), (1.2, -1.0)])
    ]
    yf = identify_axis(yaw, ExperimentKind.YAW_ROTATION, 0.16)
    assert yf.get("added_inertia").units == "kg*m^2"
    _within(yf.value("quadratic_damping"), yf.get("quadratic_damping").sigma, 1.55, 0.1)
    wrong_units = syn.axis_log(
        "E-bad",
        rng,
        ExperimentKind.SURGE_ACCELERATION,
        1.0,
        1.0,
        1.0,
        1.0,
        syn.pulse_profile((1.0,), 1.0, 1.0, dt),
        dt,
        force_units="N*m",
    )
    with pytest.raises(ValueError, match="expected 'N'"):
        identify_axis([wrong_units], ExperimentKind.SURGE_ACCELERATION, 1.0)


def test_station_keeping_current() -> None:
    rng = np.random.default_rng(4)
    log = syn.station_keeping_log("F1", rng, TRUE["d1"], TRUE["d2"], TRUE["current"], 300, 0.3)
    f = identify_station_keeping_current([log], TRUE["d1"], TRUE["d2"])
    _within(f.value("current_mps"), f.get("current_mps").sigma, TRUE["current"], 0.02)


def test_split_rules_and_synthetic_reports_are_never_promoted() -> None:
    rng = np.random.default_rng(5)
    a = syn.station_keeping_log("F1", rng, 4.0, 18.2, 0.3, 50, 0.3)
    b = syn.station_keeping_log("F2", rng, 4.0, 18.2, 0.3, 50, 0.3)
    with pytest.raises(SplitLeakageError, match="both splits"):
        IdentificationDataset("s", [a], [a])
    clone = LogSegment(
        "F1-copy", a.kind, a.time_s, dict(a.signals), dict(a.units), synthetic=True, source=a.source
    )
    with pytest.raises(SplitLeakageError, match="byte-identical"):
        IdentificationDataset("s", [a], [clone])
    ds = IdentificationDataset("s", [a], [b])
    with pytest.raises(SplitLeakageError, match="not part"):
        ds.assert_validation_only([syn.station_keeping_log("F3", rng, 4.0, 18.2, 0.3, 50, 0.3)])
    fit = identify_station_keeping_current(ds.identification, 4.0, 18.2)
    report = build_report(ds, {"F": fit}, {})
    assert report.status is ReportStatus.SYNTHETIC_TOOLING_CHECK
    assert report.real_identification_status == REAL_IDENTIFICATION_STATUS == "BLOCKED_EXTERNAL"
    assert report.dataset_lineage["synthetic"] is True
    mapping = (
        ParameterMapping(fit_name="current_mps", target="safety.max_speed_mps", category=Category.SAFETY),
    )
    with pytest.raises(SyntheticDataError):
        to_characterization_records(report, "reports/ident.json", mapping)
    # The promotion path itself, on a report whose status says real, validated logs (unit fixture of the function).
    real = IdentificationReport.model_validate(
        {**report.model_dump(), "status": ReportStatus.IDENTIFIED_AND_VALIDATED}
    )
    (rec,) = to_characterization_records(real, "reports/ident.json", mapping)
    assert rec.source is SourceKind.IDENTIFIED and rec.uncertainty_1sigma is not None
    assert rec.provenance is not None and rec.provenance.endswith(real.digest())


def test_manifest_cli_runs_on_synthetic_csv_logs(tmp_path) -> None:
    import json

    import yaml

    from conrad.robotics.hardware.identification.__main__ import main
    from conrad.robotics.hardware.identification.logio import LogFormatError, load_manifest

    rng = np.random.default_rng(6)
    dt = 0.02
    entries = []
    for tid, amps, split in (
        ("C1", (10.0, 25.0), "identification"),
        ("C2", (40.0, 5.0), "identification"),
        ("C3", (18.0, 30.0), "validation"),
    ):
        seg = syn.axis_log(
            tid,
            rng,
            ExperimentKind.SURGE_ACCELERATION,
            TRUE["mass"],
            TRUE["added"],
            TRUE["d1"],
            TRUE["d2"],
            syn.pulse_profile(amps, 3.0, 3.0, dt),
            dt,
        )
        rows = np.column_stack([seg.time_s, seg.signals["force"], seg.signals["velocity"]])
        np.savetxt(tmp_path / f"{tid}.csv", rows, delimiter=",", header="time_s,force,velocity", comments="")
        entries.append(
            {
                "trajectory_id": tid,
                "kind": "C_SURGE_ACCELERATION",
                "split": split,
                "file": f"{tid}.csv",
                "units": {"force": "N", "velocity": "m/s"},
                "synthetic": True,
            }
        )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"split_id": "cli", "segments": entries}), encoding="utf-8")
    out = tmp_path / "report.json"
    assert (
        main(
            [
                "--manifest",
                str(manifest),
                "--mass-kg",
                str(TRUE["mass"]),
                "--out",
                str(out),
                "--envelope-rmse",
                "0.01",
            ]
        )
        == 0
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert (
        doc["status"] == "SYNTHETIC_TOOLING_CHECK" and doc["real_identification_status"] == "BLOCKED_EXTERNAL"
    )
    added = next(p for p in doc["fits"]["C_SURGE_ACCELERATION"]["parameters"] if p["name"] == "added_inertia")
    assert abs(added["value"] - TRUE["added"]) < 0.05 * TRUE["added"]
    assert doc["validations"]["C_SURGE_ACCELERATION"]["within_envelope"] is True
    del entries[0]["synthetic"]
    manifest.write_text(yaml.safe_dump({"split_id": "cli", "segments": entries}), encoding="utf-8")
    with pytest.raises(LogFormatError, match="synthetic"):
        load_manifest(manifest)
