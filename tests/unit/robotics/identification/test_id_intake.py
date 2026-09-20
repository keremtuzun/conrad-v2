"""The intake checker catches every defect class of a protocol delivery (EXT-HW-04). No simulator needed.

The logs here are random SYNTHETIC fixtures shaped like a delivery; no value is a vehicle property.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from conrad.robotics.hardware.identification.dataset import ExperimentKind
from conrad.robotics.hardware.identification.intake import IntakeConfig, check_delivery
from conrad.robotics.hardware.identification.protocol import (
    PROTOCOL_VERSION,
    Variant,
    required_channels,
)

THRUSTERS = ("T1", "T2")
DT = 0.01
N = 120


def _columns(kind, variant, rng, thruster=None):
    t = np.arange(N) * DT
    cols = {"time_s": (t, "s")}
    for spec in required_channels(kind, variant, THRUSTERS):
        if spec.name.startswith("cmd_"):
            u = np.zeros(N)
            if kind is ExperimentKind.THRUSTER_STEP and spec.name != f"cmd_{thruster}":
                pass
            else:
                u[20:50], u[70:100] = 0.4, -0.4
            cols[spec.name] = (u, spec.units)
        else:
            cols[spec.name] = (rng.normal(0.0, 1.0, N), spec.units)
    return cols


def _write(root: Path, name: str, cols) -> str:
    rel = f"logs/{name}.csv"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    names = list(cols)
    data = np.stack([np.asarray(cols[n][0], dtype=np.float64) for n in names], axis=1)
    np.savetxt(root / rel, data, delimiter=",", header=",".join(names), comments="", fmt="%.17g")
    return rel


def build(root: Path, reps: int = 4, mutate_cols=None) -> Path:
    """A minimal valid delivery: B for both thrusters, C surge and F station keeping."""
    rng = np.random.default_rng(7)
    groups = [(ExperimentKind.THRUSTER_STEP, Variant.STANDARD, t) for t in THRUSTERS] + [
        (ExperimentKind.SURGE_ACCELERATION, Variant.STANDARD, None),
        (ExperimentKind.STATION_KEEPING, Variant.STANDARD, None),
    ]
    segments: list[dict[str, Any]] = []
    ident: list[str] = []
    val: list[str] = []
    for kind, variant, thr in groups:
        for r in range(reps):
            rid = f"{kind.value}-{thr or 'x'}-r{r + 1}"
            cols = _columns(kind, variant, rng, thr)
            if mutate_cols is not None:
                mutate_cols(rid, cols)
            split = "validation" if r == reps - 1 else "identification"
            (val if split == "validation" else ident).append(rid)
            entry = {
                "trajectory_id": rid,
                "kind": kind.value,
                "variant": variant.value,
                "repetition_id": rid,
                "split": split,
                "file": _write(root, rid, cols),
                "units": {n: u for n, (_, u) in cols.items() if n != "time_s"},
                "metadata": {"water_density_kgm3": 1000.0, "water_temperature_c": 20.0},
                "synthetic": True,
                "source": "SYNTHETIC_ONLY:test-fixture",
                "clock_domain": "SIM",
            }
            if thr:
                entry["thruster_id"] = thr
            segments.append(entry)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "split_id": "fixture",
        "synthetic": True,
        "source": "SYNTHETIC_ONLY:test-fixture",
        "thruster_ids": list(THRUSTERS),
        "clock": {"domain": "SIM", "synchronization": "single clock", "max_offset_s": 0.0},
        "motion_reference": {
            "kind": "SIMULATOR_TRUTH",
            "device": "fixture",
            "independent_of_robot_estimator": True,
        },
        "thrust_measurement": {"device": "fixture load cell"},
        "current_measurement": {"device": "fixture current meter"},
        "configuration": {"ballast": "none", "payload": "none", "tether": "none"},
        "split_plan": {
            "assigned_before_fitting": True,
            "identification_repetitions": ident,
            "validation_repetitions": val,
        },
        "segments": segments,
    }
    path = root / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _edit(path: Path, fn) -> Path:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(doc)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def test_valid_fixture_is_accepted(tmp_path):
    report = check_delivery(build(tmp_path))
    assert report.accepted, [f.message for f in report.failures()]
    assert report.synthetic is True
    groups = {s.group for s in report.repeatability}
    assert "B_THRUSTER_STEP/STANDARD/T1" in groups
    assert all(s.cv is not None for s in report.repeatability)


def _seg(doc, kind):
    return next(s for s in doc["segments"] if s["kind"] == kind.value)


SURGE, B, F = ExperimentKind.SURGE_ACCELERATION, ExperimentKind.THRUSTER_STEP, ExperimentKind.STATION_KEEPING


def _drop_units(doc, name):
    _seg(doc, SURGE)["units"].pop(name)


def _wrong_units(doc):
    _seg(doc, SURGE)["units"]["ref_u_mps"] = "km/h"


def _mix_split(doc):
    seg = next(s for s in doc["segments"] if s["kind"] == SURGE.value and s["split"] == "identification")
    twin = copy.deepcopy(seg)
    twin["trajectory_id"] += "-b"
    twin["split"] = "validation"
    doc["segments"].append(twin)


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        (lambda d: _drop_units(d, "imu_ax_mps2"), "UNITS_MISSING"),
        (_wrong_units, "UNITS_WRONG"),
        (lambda d: d["clock"].update(max_offset_s=0.05), "CLOCK_OFFSET"),
        (lambda d: d["clock"].pop("synchronization"), "CLOCK_UNDECLARED"),
        (lambda d: _seg(d, SURGE).update(clock_domain="OTHER"), "CLOCK_DOMAIN"),
        (lambda d: d["split_plan"].update(assigned_before_fitting=False), "SPLIT_PLAN"),
        (
            lambda d: d["split_plan"]["identification_repetitions"].append(
                d["split_plan"]["validation_repetitions"][0]
            ),
            "SPLIT_PLAN_OVERLAP",
        ),
        (lambda d: _seg(d, SURGE).update(split="validation"), "SPLIT_NOT_IN_PLAN"),
        (lambda d: _seg(d, SURGE).pop("split"), "SPLIT_MISSING"),
        (_mix_split, "SPLIT_MIXED"),
        (lambda d: d["segments"].remove(_seg(d, SURGE)), "REPETITIONS"),
        (
            lambda d: d["motion_reference"].update(independent_of_robot_estimator=False),
            "REFERENCE_NOT_INDEPENDENT",
        ),
        (lambda d: d["motion_reference"].update(kind="ROBOT_ESTIMATOR"), "REFERENCE_MISSING"),
        (lambda d: d.pop("current_measurement"), "CURRENT_NOT_MEASURED"),
        (lambda d: d.pop("thrust_measurement"), "THRUST_NOT_MEASURED"),
        (lambda d: _seg(d, SURGE)["metadata"].pop("water_temperature_c"), "METADATA"),
        (lambda d: d["configuration"].pop("tether"), "METADATA"),
        (lambda d: _seg(d, B).pop("thruster_id"), "THRUSTER_ID"),
        (lambda d: _seg(d, SURGE).pop("synthetic"), "SYNTHETIC_UNSTATED"),
        (
            lambda d: [s.update(synthetic=False) for s in d["segments"]] and d.update(synthetic=False),
            "SYNTHETIC_DISHONEST",
        ),
        (lambda d: d.update(source="sim_kernel"), "SYNTHETIC_DISHONEST"),
        (lambda d: d.update(protocol_version="v0"), "PROTOCOL_VERSION"),
        (lambda d: _seg(d, SURGE).update(file="logs/missing.csv"), "FILE_MISSING"),
    ],
)
def test_manifest_defects_fail_closed(tmp_path, edit, code):
    path = _edit(build(tmp_path), edit)
    report = check_delivery(path)
    assert not report.accepted
    assert code in report.codes(), [f"{f.code}: {f.message}" for f in report.failures()]


def _mut(target_prefix, fn):
    def mutate(rid, cols):
        if rid.startswith(target_prefix):
            fn(cols)

    return mutate


def _drop(name):
    return lambda cols: cols.pop(name)


def _non_monotonic(cols):
    t = cols["time_s"][0].copy()
    t[40] = t[39]
    cols["time_s"] = (t, "s")


def _gap(cols):
    t = cols["time_s"][0].copy()
    t[60:] += 1.0
    cols["time_s"] = (t, "s")


def _slow(name):
    def f(cols):
        x = cols[name][0].copy()
        cols[name] = (np.repeat(x[::50], 50)[:N], cols[name][1])  # 2 Hz sample-and-hold

    return f


def _nan(cols):
    x = cols["ref_u_mps"][0].copy()
    x[10] = np.nan
    cols["ref_u_mps"] = (x, "m/s")


def _forward_only(cols):
    u = cols["cmd_T1"][0].copy()
    u[u < 0] = 0.0
    cols["cmd_T1"] = (u, "unitless")


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (_mut(SURGE.value, _drop("ref_u_mps")), "CHANNEL_MISSING"),
        (_mut(F.value, _drop("current_x_mps")), "CHANNEL_MISSING"),
        (_mut(B.value, _drop("thrust_n")), "CHANNEL_MISSING"),
        (_mut(SURGE.value, _drop("imu_gz_rps")), "CHANNEL_MISSING"),
        (_mut(SURGE.value, _drop("battery_current_a")), "CHANNEL_MISSING"),
        (_mut(SURGE.value, _non_monotonic), "TIME_NOT_MONOTONIC"),
        (_mut(SURGE.value, _gap), "TIME_GAP"),
        (_mut(B.value, _slow("thrust_n")), "SAMPLE_RATE"),
        (_mut(SURGE.value, _slow("imu_ax_mps2")), "SAMPLE_RATE"),
        (_mut(SURGE.value, _nan), "CHANNEL_NOT_FINITE"),
        (_mut(f"{B.value}-T1", _forward_only), "ASYMMETRY_UNCOVERED"),
    ],
)
def test_log_defects_fail_closed(tmp_path, mutate, code):
    report = check_delivery(build(tmp_path, mutate_cols=mutate))
    assert not report.accepted
    assert code in report.codes(), [f"{f.code}: {f.message}" for f in report.failures()]


def test_motion_runs_need_thruster_models_for_every_commanded_thruster(tmp_path):
    path = _edit(
        build(tmp_path),
        lambda d: d.update(segments=[s for s in d["segments"] if s.get("thruster_id") != "T2"]),
    )
    report = check_delivery(path)
    assert "THRUSTER_MODEL_MISSING" in report.codes()


def test_not_modelled_exemption_only_for_synthetic(tmp_path):
    def edit(d):
        for s in d["segments"]:
            s["metadata"].pop("water_temperature_c")
        d["not_modelled"] = {"water_temperature_c": "fixture"}

    assert check_delivery(_edit(build(tmp_path), edit)).accepted

    def dishonest(d):
        edit(d)
        d["synthetic"] = False
        d["source"] = "PHYSICAL:rig"
        d["motion_reference"]["kind"] = "EXTERNAL_TRACKING"
        d["clock"]["domain"] = "PTP"
        for s in d["segments"]:
            s.update(synthetic=False, source="PHYSICAL:rig", clock_domain="PTP")

    report = check_delivery(_edit(build(tmp_path / "b"), dishonest))
    assert "METADATA" in report.codes()


def test_repeatability_outlier_is_flagged_but_does_not_reject(tmp_path):
    def big(cols):
        cols["ref_u_mps"] = (cols["ref_u_mps"][0] * 25.0, "m/s")

    report = check_delivery(build(tmp_path, reps=6, mutate_cols=_mut(f"{SURGE.value}-x-r2", big)))
    assert report.accepted
    stat = next(s for s in report.repeatability if s.group.startswith(SURGE.value))
    assert stat.outliers == (f"{SURGE.value}-x-r2",)
    assert any(f.code == "REPEATABILITY_OUTLIER" for f in report.findings)


def test_min_repetitions_are_configurable(tmp_path):
    path = build(tmp_path, reps=2)
    assert "REPETITIONS" in check_delivery(path).codes()
    assert check_delivery(path, IntakeConfig(min_identification_repetitions=1)).accepted
