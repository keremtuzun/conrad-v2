"""Characterization ingestion, unit conversion, RobotConfig merge and the gate ledger.

All numbers below are SYNTHETIC test fixtures written into tmp_path; none is a vehicle measurement.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conrad.robotics.hardware.characterization import (
    CSV_COLUMNS,
    AutonomyStage,
    GateError,
    GateLedger,
    IngestError,
    MergeError,
    ReadinessLevel,
    UnitError,
    convert_value,
    load_bundle,
    merge_characterization,
    sha256_file,
)
from conrad.robotics.hardware.config import load_robot_config
from conrad.schemas.robot import SourceKind

TEMPLATE = load_robot_config("configs/robot/physical_template.yaml")
STAMP = {"time_ns": 1_780_000_000_000_000_000, "clock_domain": "UTC"}


def _log(root: Path, name: str, text: str = "t_s,value\n0,1\n") -> str:
    (root / "raw").mkdir(exist_ok=True)
    (root / "raw" / name).write_text("# SYNTHETIC TEST FIXTURE\n" + text, encoding="utf-8")
    return f"raw/{name}"


def _thruster(tid: str, root: Path) -> list[dict]:
    log = _log(root, f"{tid}_bench.csv")
    vals = {
        "position_body_m": ((200.0, 150.0, 0.0), "mm"),
        "direction_body": ((1.0, 0.0, 0.0), "unit"),
        "max_forward_thrust_n": (4.0, "kgf"),
        "max_reverse_thrust_n": (30.0, "N"),
        "deadzone_command": (5.0, "%"),
        "time_constant_s": (150.0, "ms"),
        "latency_s": (20.0, "ms"),
        "thrust_coefficient": (39.0, "N"),
    }
    out = []
    for field, (value, units) in vals.items():
        rec = {
            "record_id": f"{tid}-{field}",
            "category": "thrusters",
            "target": f"thrusters[{tid}].{field}",
            "value": list(value) if isinstance(value, tuple) else value,
            "units": units,
            "source": "MEASURED",
            "method": "bench thrust stand",
            "raw_log": log,
            "timestamp": STAMP,
        }
        if field in ("position_body_m", "direction_body"):
            rec["frame_id"] = "ROBOT"
        out.append(rec)
    return out


def _bundle(root: Path, records: list[dict]) -> Path:
    path = root / "handoff.yaml"
    path.write_text(
        yaml.safe_dump({"bundle_id": "burak-2026-10", "robot_config_name": "physical", "records": records}),
        encoding="utf-8",
    )
    return path


def _base_records(root: Path) -> list[dict]:
    return [
        {
            "record_id": "m1",
            "category": "mass",
            "target": "mass_kg",
            "value": 11480.0,
            "units": "g",
            "uncertainty_1sigma": 5.0,
            "source": "MEASURED",
            "method": "scale x3",
            "raw_log": _log(root, "scale.csv"),
            "timestamp": STAMP,
        },
        {
            "record_id": "c1",
            "category": "com",
            "target": "center_of_mass_body_m",
            "value": [1.0, -2.0, -15.0],
            "units": "mm",
            "frame_id": "ROBOT",
            "source": "MEASURED",
            "method": "suspension",
            "raw_log": _log(root, "com.csv"),
            "timestamp": STAMP,
        },
        {
            "record_id": "v1",
            "category": "displaced_volume",
            "target": "displaced_volume_m3",
            "value": 11.3,
            "units": "L",
            "source": "ENGINEERING_ESTIMATE",
            "method": "CAD volume",
        },
        {
            "record_id": "b1",
            "category": "power",
            "target": "battery.capacity_j",
            "value": 266.4,
            "units": "Wh",
            "source": "LITERATURE_PRIOR",
            "method": "datasheet",
            "provenance": "pack datasheet rev B",
        },
        {
            "record_id": "h1",
            "category": "health",
            "target": "health.leak_sensor",
            "value": "OK",
            "units": "enum",
            "source": "MEASURED",
            "method": "dunk test",
            "raw_log": _log(root, "leak.txt"),
            "timestamp": STAMP,
        },
        *_thruster("T1", root),
    ]


def test_unit_conversion_is_exact_or_refused() -> None:
    assert convert_value(4.0, "kgf", "N") == pytest.approx(39.2266)
    assert convert_value((1.0, 2.0), "mm", "m") == pytest.approx((0.001, 0.002))
    assert convert_value(300.0, "K", "degC") == pytest.approx(26.85)
    assert convert_value(1.0, "N*s/m|N*m*s/rad", "N*s/m|N*m*s/rad") == 1.0
    with pytest.raises(UnitError, match="voltage"):
        convert_value(5000.0, "mAh", "J")
    with pytest.raises(UnitError):
        convert_value(1.0, "kg", "m")
    with pytest.raises(UnitError, match="unknown"):
        convert_value(1.0, "furlong", "m")


def test_yaml_ingest_and_merge_produce_a_new_versioned_config(tmp_path: Path) -> None:
    bundle = load_bundle(_bundle(tmp_path, _base_records(tmp_path)))
    mass = next(r for r in bundle.records if r.record_id == "m1")
    assert mass.raw_log_digest == sha256_file(tmp_path / "raw" / "scale.csv")
    merged, report = merge_characterization(TEMPLATE, bundle, "0.3.0")
    assert TEMPLATE.mass_kg.is_open and TEMPLATE.thrusters == ()  # the base config is untouched
    assert (
        merged.config_version == "0.3.0"
        and report.new_digest == merged.content_digest() != report.base_digest
    )
    assert merged.mass_kg.value == pytest.approx(11.48) and merged.mass_kg.source is SourceKind.MEASURED
    assert merged.mass_kg.uncertainty_1sigma == pytest.approx(0.005)
    assert "sha256:" in (merged.mass_kg.provenance or "") and merged.mass_kg.measured_at is not None
    assert merged.center_of_mass_body_m.value == pytest.approx((0.001, -0.002, -0.015))
    assert merged.displaced_volume_m3.value == pytest.approx(0.0113)
    assert merged.battery.capacity_j.value == pytest.approx(266.4 * 3600)
    t1 = merged.thrusters[0]
    assert t1.thruster_id == "T1" and t1.max_forward_thrust_n.value == pytest.approx(39.2266)
    assert t1.deadzone_command.value == pytest.approx(0.05) and t1.time_constant_s.value == pytest.approx(
        0.15
    )
    assert report.non_config_records == ("h1",)
    assert any(c.created for c in report.changes) and "mass_kg" not in report.remaining_open
    assert "inertia_diag_kgm2" in report.remaining_open  # never filled in by the merge


def test_csv_ingest_matches_yaml(tmp_path: Path) -> None:
    log = _log(tmp_path, "scale.csv")
    row = dict.fromkeys(CSV_COLUMNS, "") | {
        "record_id": "m1",
        "category": "mass",
        "target": "mass_kg",
        "value": "11.48",
        "units": "kg",
        "time_ns": str(STAMP["time_ns"]),
        "clock_domain": "UTC",
        "uncertainty_1sigma": "0.005",
        "source": "MEASURED",
        "method": "scale",
        "raw_log": log,
    }
    rows = [
        row,
        dict.fromkeys(CSV_COLUMNS, "")
        | {
            "record_id": "c1",
            "category": "com",
            "target": "center_of_mass_body_m",
            "value": "0.001;0;-0.01",
            "units": "m",
            "frame_id": "ROBOT",
            "source": "ENGINEERING_ESTIMATE",
            "method": "CAD",
        },
    ]
    path = tmp_path / "handoff.csv"
    path.write_text(
        ",".join(CSV_COLUMNS) + "\n" + "\n".join(",".join(r[c] for c in CSV_COLUMNS) for r in rows),
        encoding="utf-8",
    )
    merged, _ = merge_characterization(TEMPLATE, load_bundle(path), "0.3.0")
    assert merged.mass_kg.value == pytest.approx(11.48)
    assert merged.center_of_mass_body_m.value == pytest.approx((0.001, 0.0, -0.01))


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda r, root: r[0].pop("raw_log"), IngestError, "MEASURED requires raw_log"),
        (lambda r, root: r[0].update(raw_log="raw/missing.csv"), IngestError, "not found"),
        (lambda r, root: r[0].update(raw_log_digest="0" * 64), IngestError, "digest mismatch"),
        (lambda r, root: r[0].pop("timestamp"), IngestError, "timestamp"),
        (lambda r, root: r[3].pop("provenance"), IngestError, "citation"),
        (lambda r, root: r[2].update(source="OPEN"), IngestError, "OPEN record cannot carry a value"),
    ],
    ids=["no-raw-log", "missing-log", "digest", "no-timestamp", "literature-no-citation", "open-with-value"],
)
def test_ingest_rejects_unprovenanced_records(tmp_path: Path, mutate, error, match) -> None:
    records = _base_records(tmp_path)
    mutate(records, tmp_path)
    with pytest.raises(error, match=match):
        load_bundle(_bundle(tmp_path, records))


def test_merge_rejects_wrong_frame_units_incomplete_thruster_and_downgrade(tmp_path: Path) -> None:
    records = _base_records(tmp_path)
    records[1]["frame_id"] = "WORLD"
    with pytest.raises(MergeError, match="frame"):
        merge_characterization(TEMPLATE, load_bundle(_bundle(tmp_path, records)), "0.3.0")
    records = _base_records(tmp_path)
    records[0]["units"] = "m"
    with pytest.raises(UnitError):
        merge_characterization(TEMPLATE, load_bundle(_bundle(tmp_path, records)), "0.3.0")
    records = [r for r in _base_records(tmp_path) if r["target"] != "thrusters[T1].latency_s"]
    with pytest.raises(MergeError, match="incomplete"):
        merge_characterization(TEMPLATE, load_bundle(_bundle(tmp_path, records)), "0.3.0")
    merged, _ = merge_characterization(
        TEMPLATE, load_bundle(_bundle(tmp_path, _base_records(tmp_path))), "0.3.0"
    )
    weaker = [
        {
            "record_id": "m2",
            "category": "mass",
            "target": "mass_kg",
            "value": 12.0,
            "units": "kg",
            "source": "ENGINEERING_ESTIMATE",
            "method": "guess",
        }
    ]
    with pytest.raises(MergeError, match="downgrade"):
        merge_characterization(merged, load_bundle(_bundle(tmp_path, weaker)), "0.4.0")
    with pytest.raises(MergeError, match="differ"):
        merge_characterization(merged, load_bundle(_bundle(tmp_path, weaker)), "0.3.0")


def test_gate_ledger_cannot_skip_and_needs_evidence(tmp_path: Path) -> None:
    ev = tmp_path / "evidence"
    ev.mkdir()
    for name in ("r0", "r1", "r2", "r3", "r4", "manual", "hold"):
        (ev / f"{name}.json").write_text(f'{{"gate": "{name}", "fixture": "SYNTHETIC"}}', encoding="utf-8")
    ledger = GateLedger(tmp_path / "ledger.json")
    with pytest.raises(GateError, match="prerequisites"):
        ledger.pass_readiness(ReadinessLevel.R2_HARDWARE_IN_THE_LOOP, "evidence/r2.json", "kerem", 1)
    with pytest.raises(GateError, match="does not exist"):
        ledger.pass_readiness(ReadinessLevel.R0_PURE_SIMULATION, "evidence/none.json", "kerem", 1)
    ledger.pass_readiness(ReadinessLevel.R0_PURE_SIMULATION, "evidence/r0.json", "kerem", 1)
    ledger.pass_readiness(ReadinessLevel.R1_SOFTWARE_IN_THE_LOOP, "evidence/r1.json", "kerem", 2)
    with pytest.raises(GateError, match="needs readiness"):
        ledger.activate_autonomy(AutonomyStage.MANUAL, "evidence/manual.json", "burak", 3)
    ledger.pass_readiness(ReadinessLevel.R2_HARDWARE_IN_THE_LOOP, "evidence/r2.json", "kerem", 3)
    ledger.pass_readiness(ReadinessLevel.R3_DRY_BENCH, "evidence/r3.json", "burak", 4)
    ledger.activate_autonomy(AutonomyStage.MANUAL, "evidence/manual.json", "burak", 5)
    with pytest.raises(GateError, match="earlier stages"):
        ledger.activate_autonomy(AutonomyStage.WAYPOINT, "evidence/hold.json", "burak", 6)
    with pytest.raises(GateError, match="R4_CONTROLLED_WATER"):
        ledger.activate_autonomy(AutonomyStage.ATTITUDE_DEPTH_HOLD, "evidence/hold.json", "burak", 6)
    ledger.pass_readiness(ReadinessLevel.R4_CONTROLLED_WATER, "evidence/r4.json", "burak", 7)
    ledger.activate_autonomy(AutonomyStage.ATTITUDE_DEPTH_HOLD, "evidence/hold.json", "burak", 8)
    assert ledger.active_autonomy() is AutonomyStage.ATTITUDE_DEPTH_HOLD

    reloaded = GateLedger(tmp_path / "ledger.json")  # persisted + evidence re-verified
    assert reloaded.highest_readiness() is ReadinessLevel.R4_CONTROLLED_WATER
    reloaded.revoke("readiness", ReadinessLevel.R4_CONTROLLED_WATER.value, "burak", 9, "leak in pool test")
    assert reloaded.highest_readiness() is ReadinessLevel.R3_DRY_BENCH
    assert reloaded.active_autonomy() is AutonomyStage.MANUAL  # hold needed R4; it no longer counts

    (ev / "r1.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(GateError, match="changed"):
        GateLedger(tmp_path / "ledger.json")


def test_cli_writes_new_config_and_report_and_refuses_overwrite(tmp_path: Path) -> None:
    import json

    from conrad.robotics.hardware.characterization.__main__ import main

    handoff = _bundle(tmp_path, _base_records(tmp_path))
    out, rep = tmp_path / "physical_v0_3_0.yaml", tmp_path / "change.json"
    args = [
        "--base",
        "configs/robot/physical_template.yaml",
        "--handoff",
        str(handoff),
        "--version",
        "0.3.0",
        "--out",
        str(out),
        "--report",
        str(rep),
    ]
    assert main(args) == 0
    reloaded = load_robot_config(out)
    assert reloaded.config_version == "0.3.0" and reloaded.mass_kg.source is SourceKind.MEASURED
    assert json.loads(rep.read_text(encoding="utf-8"))["new_digest"] == reloaded.content_digest()
    with pytest.raises(SystemExit, match="overwrite"):
        main(args)
