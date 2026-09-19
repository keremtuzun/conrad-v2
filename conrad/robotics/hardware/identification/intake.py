"""Intake checker: does a delivered identification log package satisfy the protocol? Fails closed.

Run it on the hardware owner's manifest before any fitting::

    python -m conrad.robotics.hardware.identification --manifest logs/manifest.yaml --intake-only \\
        --out artifacts/identification/intake.json

Every problem is a :class:`IntakeFinding` with a stable code. Any ``FAIL`` finding rejects the delivery.
``WARN`` findings (constant channels, repeatability outliers, protocol kinds not delivered yet) are reported
but do not reject. Repeatability is reported per group as a coefficient of variation; there is no
acceptance threshold for it until the owners declare one (OPEN).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import Field

from conrad.robotics.hardware.identification.dataset import ExperimentKind
from conrad.robotics.hardware.identification.protocol import (
    MIN_IDENTIFICATION_REPETITIONS,
    MIN_VALIDATION_REPETITIONS,
    NEEDS_THRUSTER_MODELS,
    PRIMARY_RESPONSE,
    PROTOCOL_VERSION,
    REQUIRED_CHANNELS,
    REQUIRED_CONFIGURATION,
    REQUIRED_SEGMENT_METADATA,
    MotionReferenceKind,
    Variant,
    optional_channels,
    required_channels,
)
from conrad.schemas.base import ConradModel

FAIL, WARN, INFO = "FAIL", "WARN", "INFO"


class IntakeConfig(ConradModel):
    """Thresholds of the check itself. All ENGINEERING_ESTIMATE; the hardware owner may tighten them."""

    min_identification_repetitions: int = Field(default=MIN_IDENTIFICATION_REPETITIONS, ge=1)
    min_validation_repetitions: int = Field(default=MIN_VALIDATION_REPETITIONS, ge=1)
    max_clock_offset_s: float = Field(default=0.005, gt=0)
    max_gap_factor: float = Field(default=5.0, gt=1, description="gap > factor * median dt fails")
    outlier_robust_z: float = Field(default=3.5, gt=0)
    min_samples: int = Field(default=20, ge=2)


class IntakeFinding(ConradModel):
    severity: str
    code: str
    trajectory_id: str | None = None
    message: str


class RepeatabilityStat(ConradModel):
    group: str
    feature: str
    n_repetitions: int
    mean: float
    std: float
    cv: float | None
    per_repetition: dict[str, float]
    outliers: tuple[str, ...]


class IntakeReport(ConradModel):
    protocol_version: str = PROTOCOL_VERSION
    manifest: str
    accepted: bool
    synthetic: bool | None
    findings: tuple[IntakeFinding, ...]
    repeatability: tuple[RepeatabilityStat, ...]
    groups: dict[str, dict[str, int]]

    def failures(self) -> tuple[IntakeFinding, ...]:
        return tuple(f for f in self.findings if f.severity == FAIL)

    def codes(self) -> set[str]:
        return {f.code for f in self.findings if f.severity == FAIL}


class _Checker:
    def __init__(self, cfg: IntakeConfig) -> None:
        self.cfg = cfg
        self.findings: list[IntakeFinding] = []

    def add(self, severity: str, code: str, message: str, tid: str | None = None) -> None:
        self.findings.append(IntakeFinding(severity=severity, code=code, trajectory_id=tid, message=message))


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError("empty file")
    header, body = rows[0], [r for r in rows[1:] if r]
    if any(len(r) != len(header) for r in body):
        raise ValueError("ragged rows")
    try:
        data = np.asarray([[float(x) if x.strip() else math.nan for x in r] for r in body], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"non-numeric value ({exc})") from exc
    return header, data.reshape(len(body), len(header))


def _effective_rate(t: np.ndarray, x: np.ndarray) -> float | None:
    """Updates per second, counting value changes (a held sample is not a new sample). None if constant."""
    changes = int(np.count_nonzero(np.diff(x) != 0))
    if changes == 0:
        return None
    return (changes + 1) / float(t[-1] - t[0])


def _group_key(kind: ExperimentKind, variant: Variant, thruster_id: str | None) -> str:
    return f"{kind.value}/{variant.value}" + (f"/{thruster_id}" if thruster_id else "")


def check_delivery(manifest_path: str | Path, config: IntakeConfig | None = None) -> IntakeReport:
    cfg = config or IntakeConfig()
    c = _Checker(cfg)
    p = Path(manifest_path)
    try:
        doc: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        c.add(FAIL, "MANIFEST_UNREADABLE", str(exc))
        return _report(p, c, None, [], {})
    if not isinstance(doc, dict) or not isinstance(doc.get("segments"), list) or not doc["segments"]:
        c.add(FAIL, "MANIFEST_STRUCTURE", "manifest needs a non-empty 'segments' list")
        return _report(p, c, None, [], {})

    synthetic = _check_top_level(c, doc)
    thruster_ids = tuple(str(t) for t in doc.get("thruster_ids") or ())
    plan = doc.get("split_plan") or {}
    plan_val = {str(r) for r in plan.get("validation_repetitions") or ()}
    plan_ident = {str(r) for r in plan.get("identification_repetitions") or ()}
    if plan_val & plan_ident:
        c.add(FAIL, "SPLIT_PLAN_OVERLAP", f"repetitions in both splits: {sorted(plan_val & plan_ident)}")
    not_modelled = doc.get("not_modelled") or {}
    rep_split: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"identification": set(), "validation": set()}
    )
    features: dict[str, dict[str, float]] = defaultdict(dict)
    b_thrusters: set[str] = set()
    commanded: set[str] = set()
    seen_ids: set[str] = set()
    present_kinds: set[tuple[ExperimentKind, Variant]] = set()

    for entry in doc["segments"]:
        tid = str(entry.get("trajectory_id") or "")
        if not tid:
            c.add(FAIL, "SEGMENT_FIELD", "segment without trajectory_id")
            continue
        if tid in seen_ids:
            c.add(FAIL, "DUPLICATE_TRAJECTORY", "trajectory_id used twice", tid)
        seen_ids.add(tid)
        try:
            kind = ExperimentKind(entry.get("kind"))
            variant = Variant(entry.get("variant", Variant.STANDARD.value))
        except ValueError:
            c.add(FAIL, "UNKNOWN_KIND", f"kind/variant {entry.get('kind')}/{entry.get('variant')}", tid)
            continue
        if (kind, variant) not in REQUIRED_CHANNELS:
            c.add(FAIL, "UNKNOWN_KIND", f"{kind.value}/{variant.value} is not a protocol procedure", tid)
            continue
        present_kinds.add((kind, variant))
        rep = str(entry.get("repetition_id") or "")
        split = entry.get("split")
        if not rep:
            c.add(FAIL, "SEGMENT_FIELD", "repetition_id is required (splits are by whole repetition)", tid)
        if split not in ("identification", "validation"):
            c.add(FAIL, "SPLIT_MISSING", "split must be identification or validation", tid)
        elif rep:
            rep_split[rep].add(split)
            planned = plan_val if split == "validation" else plan_ident
            if rep not in planned:
                c.add(FAIL, "SPLIT_NOT_IN_PLAN", f"repetition {rep} ({split}) is not in split_plan", tid)
        if "synthetic" not in entry:
            c.add(FAIL, "SYNTHETIC_UNSTATED", "'synthetic' must be stated on every segment", tid)
        elif synthetic is not None and bool(entry["synthetic"]) != synthetic:
            c.add(FAIL, "SYNTHETIC_INCONSISTENT", "segment synthetic flag differs from the manifest", tid)
        seg_source = str(entry.get("source", doc.get("source", "")))
        if entry.get("synthetic") is False and _looks_synthetic(seg_source, entry.get("clock_domain")):
            c.add(FAIL, "SYNTHETIC_DISHONEST", f"source {seg_source!r} is simulated but synthetic=false", tid)
        clock = (doc.get("clock") or {}).get("domain")
        if entry.get("clock_domain") != clock:
            c.add(
                FAIL, "CLOCK_DOMAIN", f"clock_domain {entry.get('clock_domain')!r} != manifest {clock!r}", tid
            )
        thr = entry.get("thruster_id")
        if kind is ExperimentKind.THRUSTER_STEP:
            if thr is None or str(thr) not in thruster_ids:
                c.add(
                    FAIL, "THRUSTER_ID", f"thruster step needs a thruster_id from {list(thruster_ids)}", tid
                )
                thr = None
            else:
                thr = str(thr)
        else:
            thr = None
        meta = entry.get("metadata") or {}
        for key in REQUIRED_SEGMENT_METADATA:
            if key in meta:
                if not isinstance(meta[key], (int, float)) or not math.isfinite(float(meta[key])):
                    c.add(FAIL, "METADATA", f"metadata {key} must be a finite number", tid)
            elif key in not_modelled and synthetic:
                pass
            else:
                c.add(FAIL, "METADATA", f"metadata {key} missing", tid)
        units = entry.get("units") or {}
        f = entry.get("file")
        path = p.parent / str(f) if f else None
        if path is None or not path.is_file():
            c.add(FAIL, "FILE_MISSING", f"log file {f!r} not found", tid)
            continue
        try:
            header, data = _read_csv(path)
        except (OSError, ValueError) as exc:
            c.add(FAIL, "CSV_FORMAT", str(exc), tid)
            continue
        if not header or header[0] != "time_s":
            c.add(FAIL, "CSV_FORMAT", "first column must be time_s", tid)
            continue
        if data.shape[0] < cfg.min_samples:
            c.add(FAIL, "TOO_FEW_SAMPLES", f"{data.shape[0]} samples < {cfg.min_samples}", tid)
            continue
        t = data[:, 0]
        dt = np.diff(t)
        if not np.all(np.isfinite(t)) or np.any(dt <= 0):
            c.add(FAIL, "TIME_NOT_MONOTONIC", "time_s must be finite and strictly increasing", tid)
            continue
        med = float(np.median(dt))
        if float(np.max(dt)) > cfg.max_gap_factor * med:
            c.add(
                FAIL,
                "TIME_GAP",
                f"gap {float(np.max(dt)):.4f} s > {cfg.max_gap_factor} x median {med:.4f} s",
                tid,
            )
        cols = {n: data[:, i] for i, n in enumerate(header)}
        undeclared = [n for n in header[1:] if n not in units]
        if undeclared:
            c.add(FAIL, "UNITS_MISSING", f"columns without units {undeclared}", tid)
        spec_by_name = {s.name: s for s in optional_channels(thruster_ids)}
        required = required_channels(kind, variant, thruster_ids)
        for s in required:
            spec_by_name[s.name] = s
            if s.name not in cols:
                c.add(
                    FAIL, "CHANNEL_MISSING", f"required channel {s.name} [{s.units}] ({s.group}) missing", tid
                )
        for name, s in spec_by_name.items():
            if name not in cols:
                continue
            if name in units and units[name] != s.units:
                c.add(
                    FAIL, "UNITS_WRONG", f"{name} is in {units[name]!r}, protocol requires {s.units!r}", tid
                )
            x = cols[name]
            is_required = any(r.name == name for r in required)
            if is_required and not np.all(np.isfinite(x)):
                c.add(FAIL, "CHANNEL_NOT_FINITE", f"{name} has {int(np.sum(~np.isfinite(x)))} NaN/inf", tid)
                continue
            if not is_required or not s.rate_checked:
                continue
            rate = _effective_rate(t, x)
            if rate is None:
                c.add(WARN, "CHANNEL_CONSTANT", f"{name} never changes (no information about its rate)", tid)
            elif rate < s.min_rate_hz:
                c.add(FAIL, "SAMPLE_RATE", f"{name} updates at {rate:.2f} Hz < {s.min_rate_hz} Hz", tid)
        if kind is ExperimentKind.THRUSTER_STEP and thr is not None:
            b_thrusters.add(thr)
            u = cols.get(f"cmd_{thr}")
            if u is not None and not (np.any(u > 0) and np.any(u < 0)):
                c.add(FAIL, "ASYMMETRY_UNCOVERED", "thruster step needs forward AND reverse commands", tid)
        elif (kind, variant) in NEEDS_THRUSTER_MODELS:
            for tt in thruster_ids:
                ucol = cols.get(f"cmd_{tt}")
                if ucol is not None and np.any(ucol != 0):
                    commanded.add(tt)
        if kind is ExperimentKind.STATION_KEEPING and not (doc.get("current_measurement") or {}).get(
            "device"
        ):
            c.add(FAIL, "CURRENT_NOT_MEASURED", "station keeping needs current_measurement.device", tid)
        if kind is ExperimentKind.THRUSTER_STEP and not (doc.get("thrust_measurement") or {}).get("device"):
            c.add(
                FAIL, "THRUST_NOT_MEASURED", "thruster steps need thrust_measurement.device (load cell)", tid
            )
        key = _group_key(kind, variant, thr)
        if rep and split in ("identification", "validation"):
            groups[key][str(split)].add(rep)
        feat = PRIMARY_RESPONSE[(kind, variant)]
        if feat in cols and np.all(np.isfinite(cols[feat])) and rep:
            x = cols[feat]
            features[f"{key}|{feat}"][rep] = float(np.sqrt(np.mean((x - x[0]) ** 2)))

    for rep, splits in rep_split.items():
        if len(splits) > 1:
            c.add(FAIL, "SPLIT_MIXED", f"repetition {rep} has segments in both splits")
    counts: dict[str, dict[str, int]] = {}
    for key, g in sorted(groups.items()):
        counts[key] = {k: len(v) for k, v in g.items()}
        if len(g["identification"]) < cfg.min_identification_repetitions:
            c.add(
                FAIL,
                "REPETITIONS",
                f"{key}: {len(g['identification'])} identification repetitions < "
                f"{cfg.min_identification_repetitions}",
            )
        if len(g["validation"]) < cfg.min_validation_repetitions:
            c.add(
                FAIL,
                "REPETITIONS",
                f"{key}: {len(g['validation'])} held-out repetitions < {cfg.min_validation_repetitions}",
            )
    missing_models = sorted(commanded - b_thrusters)
    if missing_models:
        c.add(
            FAIL,
            "THRUSTER_MODEL_MISSING",
            f"thrusters {missing_models} drive motion runs but have no measured thruster-step (B) logs",
        )
    for kv in sorted(set(REQUIRED_CHANNELS) - present_kinds, key=lambda k: (k[0].value, k[1].value)):
        c.add(INFO, "KIND_NOT_DELIVERED", f"{kv[0].value}/{kv[1].value} not in this delivery")
    stats = _repeatability(c, features)
    return _report(p, c, synthetic, stats, counts)


def _looks_synthetic(source: str, clock_domain: Any) -> bool:
    s = source.upper()
    return "SYNTHETIC" in s or "SIM" in s.split(":")[0] or clock_domain == "SIM"


def _check_top_level(c: _Checker, doc: dict[str, Any]) -> bool | None:
    if doc.get("protocol_version") != PROTOCOL_VERSION:
        c.add(FAIL, "PROTOCOL_VERSION", f"protocol_version must be {PROTOCOL_VERSION!r}")
    if not doc.get("split_id"):
        c.add(FAIL, "MANIFEST_STRUCTURE", "split_id missing")
    if not doc.get("thruster_ids"):
        c.add(FAIL, "MANIFEST_STRUCTURE", "thruster_ids missing")
    synthetic: bool | None = None
    if "synthetic" not in doc:
        c.add(FAIL, "SYNTHETIC_UNSTATED", "manifest must state synthetic: true|false")
    else:
        synthetic = bool(doc["synthetic"])
    source = str(doc.get("source") or "")
    if not source:
        c.add(FAIL, "SOURCE_MISSING", "manifest must name its source")
    clock = doc.get("clock") or {}
    ref = doc.get("motion_reference") or {}
    if synthetic is False and _looks_synthetic(source, clock.get("domain")):
        c.add(FAIL, "SYNTHETIC_DISHONEST", f"source {source!r} / clock {clock.get('domain')!r} is simulated")
    if synthetic is False and ref.get("kind") == MotionReferenceKind.SIMULATOR_TRUTH.value:
        c.add(FAIL, "SYNTHETIC_DISHONEST", "simulator truth reference in a delivery marked synthetic=false")
    if synthetic is True and not source.startswith("SYNTHETIC_ONLY"):
        c.add(FAIL, "SYNTHETIC_DISHONEST", "synthetic logs must name a SYNTHETIC_ONLY source")
    if synthetic is False and doc.get("not_modelled"):
        c.add(FAIL, "METADATA", "not_modelled exemptions are only valid for synthetic logs")
    if not clock.get("domain"):
        c.add(FAIL, "CLOCK_UNDECLARED", "clock.domain missing (one common clock for every column)")
    if not clock.get("synchronization"):
        c.add(FAIL, "CLOCK_UNDECLARED", "clock.synchronization method missing")
    off = clock.get("max_offset_s")
    if not isinstance(off, (int, float)):
        c.add(FAIL, "CLOCK_UNDECLARED", "clock.max_offset_s (measured worst-case offset) missing")
    elif float(off) > c.cfg.max_clock_offset_s:
        c.add(FAIL, "CLOCK_OFFSET", f"clock offset {off} s > {c.cfg.max_clock_offset_s} s")
    try:
        kind = MotionReferenceKind(ref.get("kind"))
    except ValueError:
        c.add(
            FAIL,
            "REFERENCE_MISSING",
            f"motion_reference.kind must be one of {[k.value for k in MotionReferenceKind]}",
        )
    else:
        if kind is MotionReferenceKind.SIMULATOR_TRUTH and synthetic is not True:
            c.add(FAIL, "REFERENCE_NOT_INDEPENDENT", "simulator truth is only a reference for synthetic logs")
    if ref.get("independent_of_robot_estimator") is not True:
        c.add(
            FAIL, "REFERENCE_NOT_INDEPENDENT", "motion reference must be independent of the robot's estimator"
        )
    if not ref.get("device"):
        c.add(FAIL, "REFERENCE_MISSING", "motion_reference.device missing")
    conf = doc.get("configuration") or {}
    for key in REQUIRED_CONFIGURATION:
        if not str(conf.get(key) or "").strip():
            c.add(FAIL, "METADATA", f"configuration.{key} missing (write 'none' if there is none)")
    plan = doc.get("split_plan") or {}
    if plan.get("assigned_before_fitting") is not True:
        c.add(FAIL, "SPLIT_PLAN", "split_plan.assigned_before_fitting must be true")
    if not plan.get("validation_repetitions"):
        c.add(FAIL, "SPLIT_PLAN", "split_plan.validation_repetitions is empty")
    return synthetic


def _repeatability(c: _Checker, features: dict[str, dict[str, float]]) -> list[RepeatabilityStat]:
    out = []
    for key, per_rep in sorted(features.items()):
        group, feat = key.split("|")
        vals = np.asarray(list(per_rep.values()))
        mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        outliers: list[str] = []
        if vals.size >= 3 and mad > 0:
            for rep, v in per_rep.items():
                if abs(0.6745 * (v - med) / mad) > c.cfg.outlier_robust_z:
                    outliers.append(rep)
        for rep in outliers:
            c.add(WARN, "REPEATABILITY_OUTLIER", f"{group}: repetition {rep} {feat} RMS deviates (robust z)")
        out.append(
            RepeatabilityStat(
                group=group,
                feature=f"rms({feat} - initial)",
                n_repetitions=int(vals.size),
                mean=mean,
                std=std,
                cv=std / abs(mean) if mean != 0 else None,
                per_repetition=dict(per_rep),
                outliers=tuple(outliers),
            )
        )
    return out


def _report(
    p: Path,
    c: _Checker,
    synthetic: bool | None,
    stats: list[RepeatabilityStat],
    counts: dict[str, dict[str, int]],
) -> IntakeReport:
    return IntakeReport(
        manifest=str(p),
        accepted=not any(f.severity == FAIL for f in c.findings),
        synthetic=synthetic,
        findings=tuple(c.findings),
        repeatability=tuple(stats),
        groups=counts,
    )
