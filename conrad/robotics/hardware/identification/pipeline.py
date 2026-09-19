"""Protocol pipeline: intake -> derive fit inputs -> identify on the identification split -> held-out validation.

Derivations (all from delivered channels; nothing is taken from a simulator or a nominal value):

* B: ``command = cmd_<thruster>`` and the load-cell ``thrust_n`` per thruster -> k_fwd, k_rev, tau, latency,
  deadzone per thruster.
* C/D/E: body force/moment on the excited axis = row of the allocation matrix (RobotConfig geometry) times
  every thruster's thrust predicted by its IDENTIFIED B model from the logged commands; velocity is the
  independent reference.
* A (tilt): per constant-command plateau, the settled tail mean of applied moment and reference pitch.
* D_HEAVE bias: net buoyancy; displaced volume follows with the measured mass and water density.
* F: hold force = minus the thruster force on body x over the settled tail; the current is fitted with
  the surge drag and checked against the current meter on held-out runs.

Rigid-body mass and inertia are read from the RobotConfig (Phase 12 measurements); an OPEN value stops
the pipeline instead of being assumed.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import Field

from conrad.robotics.hardware.identification.dataset import ExperimentKind, IdentificationDataset, LogSegment
from conrad.robotics.hardware.identification.experiments import (
    identify_axis,
    identify_displaced_volume,
    identify_station_keeping_current,
    identify_thruster,
    identify_trim,
)
from conrad.robotics.hardware.identification.fitting import FitResult, FittedParameter, IdentificationError
from conrad.robotics.hardware.identification.intake import IntakeConfig, IntakeReport, check_delivery
from conrad.robotics.hardware.identification.logio import load_csv_segment
from conrad.robotics.hardware.identification.models import G0, thruster_response
from conrad.robotics.hardware.identification.protocol import REFERENCE_VELOCITY, WRENCH_AXIS, Variant
from conrad.robotics.hardware.identification.report import IdentificationReport, build_report
from conrad.robotics.hardware.identification.validation import (
    ValidationResult,
    validate_axis,
    validate_station_keeping,
    validate_thruster,
    validate_trim,
)
from conrad.schemas.base import ConradModel
from conrad.schemas.robot import RobotConfig

_AXES = (
    ExperimentKind.SURGE_ACCELERATION,
    ExperimentKind.SWAY,
    ExperimentKind.HEAVE,
    ExperimentKind.YAW_ROTATION,
)


class IntakeRejectedError(ValueError):
    def __init__(self, report: IntakeReport) -> None:
        self.report = report
        reasons = "; ".join(f"{f.code}: {f.message}" for f in report.failures()[:10])
        super().__init__(f"delivery rejected by intake ({len(report.failures())} failures): {reasons}")


class PipelineConfig(ConradModel):
    fit_rate_hz: float | None = Field(
        default=None, gt=0, description="block-average axis logs to this rate before fitting (speed only)"
    )
    plateau_settled_fraction: float = Field(default=0.25, gt=0, le=1)
    station_settled_fraction: float = Field(default=0.5, gt=0, le=1)
    max_station_heading_error_rad: float = Field(
        default=0.17, gt=0, description="F: current must lie along body surge (ENGINEERING_ESTIMATE)"
    )
    envelope_rmse: dict[str, float] = Field(
        default_factory=dict, description="pre-declared held-out RMSE limits per fit key; OPEN when absent"
    )
    intake: IntakeConfig = Field(default_factory=IntakeConfig)


@dataclass(frozen=True)
class ThrusterModel:
    k_fwd: float
    k_rev: float
    deadzone: float
    tau_s: float
    latency_s: float

    def thrust(self, t: np.ndarray, u: np.ndarray) -> np.ndarray:
        return thruster_response(t, u, self.k_fwd, self.k_rev, self.deadzone, self.tau_s, self.latency_s)


@dataclass
class _Delivered:
    entry: dict[str, Any]
    raw: LogSegment
    split: str
    variant: Variant


@dataclass
class PipelineResult:
    intake: IntakeReport
    report: IdentificationReport
    thruster_models: dict[str, ThrusterModel]
    notes: list[str] = field(default_factory=list)


def allocation_matrix(robot_config: RobotConfig) -> tuple[np.ndarray, tuple[str, ...]]:
    cols, ids = [], []
    for t in robot_config.thrusters:
        p = f"thrusters[{t.thruster_id}]"
        r = np.asarray(t.position_body_m.require(f"{p}.position_body_m"), dtype=np.float64)
        d = np.asarray(t.direction_body.require(f"{p}.direction_body"), dtype=np.float64)
        d = d / np.linalg.norm(d)
        cols.append(np.concatenate([d, np.cross(r, d)]))
        ids.append(t.thruster_id)
    return np.stack(cols, axis=1), tuple(ids)


def _load(manifest: Path) -> tuple[dict[str, Any], list[_Delivered]]:
    doc: dict[str, Any] = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    out = []
    for e in doc["segments"]:
        seg = load_csv_segment(
            manifest.parent / e["file"],
            str(e["trajectory_id"]),
            ExperimentKind(e["kind"]),
            dict(e["units"]),
            synthetic=bool(e["synthetic"]),
            metadata={k: float(v) for k, v in (e.get("metadata") or {}).items()},
        )
        out.append(_Delivered(e, seg, str(e["split"]), Variant(e.get("variant", "STANDARD"))))
    return doc, out


def _derived(
    raw: LogSegment, t: np.ndarray, signals: dict[str, np.ndarray], units: dict[str, str]
) -> LogSegment:
    return LogSegment(
        raw.trajectory_id,
        raw.kind,
        t,
        signals,
        units,
        synthetic=raw.synthetic,
        source=f"derived:{raw.source}",
        metadata=raw.metadata,
    )


def _body_thrust(
    seg: LogSegment, models: dict[str, ThrusterModel], ids: tuple[str, ...], b: np.ndarray, row: int
) -> np.ndarray:
    total = np.zeros(seg.time_s.size)
    for i, tid in enumerate(ids):
        u = seg.signal(f"cmd_{tid}", "unitless")
        if b[row, i] == 0.0 or not np.any(u != 0):
            continue
        if tid not in models:
            raise IdentificationError(f"{seg.trajectory_id}: thruster {tid} is commanded but has no B model")
        total += b[row, i] * models[tid].thrust(seg.time_s, u)
    return total


def _block(t: np.ndarray, f: np.ndarray, v: np.ndarray, rate_hz: float | None) -> tuple[np.ndarray, ...]:
    """Zero-order-hold equivalent of a continuous force for ``simulate_axis``.

    ``simulate_axis`` holds f[k] over [t_k, t_k+1). The thrust is continuous, so f[k] is the trapezoid
    mean over that interval (holding the start value instead lags the input by half a sample, which
    biased the fast yaw fit in the kernel rehearsal). With ``rate_hz`` the interval means are further
    block-averaged and the velocity is sampled at block starts.
    """
    f = np.append(0.5 * (f[:-1] + f[1:]), f[-1])
    if rate_hz is None:
        return t, f, v
    step = max(1, round(1.0 / (rate_hz * float(np.median(np.diff(t))))))
    n = (t.size // step) * step
    if n < 2 * step:
        return t, f, v
    return t[:n:step], f[:n].reshape(-1, step).mean(axis=1), v[:n:step]


def _plateaus(key: np.ndarray, settled_fraction: float, min_len: int = 4) -> list[slice]:
    """Settled tails of runs where every column of ``key`` is constant."""
    change = np.flatnonzero(np.any(np.diff(key, axis=0) != 0, axis=1)) + 1
    bounds = [0, *change.tolist(), key.shape[0]]
    out = []
    for a, b in pairwise(bounds):
        if b - a >= min_len:
            out.append(slice(b - max(2, int((b - a) * settled_fraction)), b))
    return out


def run_pipeline(
    manifest_path: str | Path, robot_config: RobotConfig, config: PipelineConfig | None = None
) -> PipelineResult:
    cfg = config or PipelineConfig()
    manifest = Path(manifest_path)
    intake = check_delivery(manifest, cfg.intake)
    if not intake.accepted:
        raise IntakeRejectedError(intake)
    doc, delivered = _load(manifest)
    mass = float(robot_config.mass_kg.require("mass_kg"))
    inertia = np.asarray(robot_config.inertia_diag_kgm2.require("inertia_diag_kgm2"), dtype=np.float64)
    b, ids = allocation_matrix(robot_config)
    notes: list[str] = [f"protocol {doc.get('protocol_version')}; manifest {manifest.name}"]
    fits: dict[str, FitResult] = {}
    validations: dict[str, ValidationResult] = {}
    ident: list[LogSegment] = []
    valid: list[LogSegment] = []
    pending: list[tuple[str, str, Any]] = []  # (fit key, split, derived segment) bookkeeping

    def add(seg: LogSegment, split: str, key: str) -> None:
        (ident if split == "identification" else valid).append(seg)
        pending.append((key, split, seg))

    # B: thruster models --------------------------------------------------------------------------------
    models: dict[str, ThrusterModel] = {}
    b_segments: dict[str, list[tuple[str, LogSegment]]] = {}
    for d in delivered:
        if d.raw.kind is ExperimentKind.THRUSTER_STEP:
            tid = str(d.entry["thruster_id"])
            seg = _derived(
                d.raw,
                d.raw.time_s,
                {
                    "command": d.raw.signal(f"cmd_{tid}", "unitless"),
                    "thrust_n": d.raw.signal("thrust_n", "N"),
                },
                {"command": "unitless", "thrust_n": "N"},
            )
            b_segments.setdefault(tid, []).append((d.split, seg))
            add(seg, d.split, f"B_thruster_{tid}")
    dz_by: dict[str, float] = {}
    for tid, segs in sorted(b_segments.items()):
        fit, dz = identify_thruster([s for sp, s in segs if sp == "identification"])
        fits[f"B_thruster_{tid}"] = fit.model_copy(
            update={
                "parameters": (
                    *fit.parameters,
                    FittedParameter(name="deadzone", value=dz.value, sigma=dz.sigma, units="unitless"),
                )
            }
        )
        dz_by[tid] = dz.value
        models[tid] = ThrusterModel(
            fit.value("k_fwd"),
            fit.value("k_rev"),
            dz.value,
            fit.value("time_constant_s"),
            fit.value("latency_s"),
        )

    # A (scale), C/D/E, F, A (tilt): derive -------------------------------------------------------------
    rho_values = {
        float(d.raw.metadata["water_density_kgm3"])
        for d in delivered
        if "water_density_kgm3" in d.raw.metadata
    }
    for d in delivered:
        k, raw = d.raw.kind, d.raw
        if k is ExperimentKind.STATIC_TRIM and d.variant is Variant.SCALE:
            add(
                _derived(
                    raw, raw.time_s, {"net_weight_n": raw.signal("net_weight_n", "N")}, {"net_weight_n": "N"}
                ),
                d.split,
                "A_displaced_volume",
            )
        elif k in _AXES:
            row = WRENCH_AXIS[k]
            vname, vunits = REFERENCE_VELOCITY[k]
            force = _body_thrust(raw, models, ids, b, row)
            t, f, v = _block(raw.time_s, force, raw.signal(vname, vunits), cfg.fit_rate_hz)
            funits = "N*m" if k is ExperimentKind.YAW_ROTATION else "N"
            add(
                _derived(raw, t, {"force": f, "velocity": v}, {"force": funits, "velocity": vunits}),
                d.split,
                k.value,
            )
        elif k is ExperimentKind.STATION_KEEPING:
            n = raw.time_s.size
            tail = slice(n - max(2, int(n * cfg.station_settled_fraction)), n)
            hold = -_body_thrust(raw, models, ids, b, 0)[tail]
            yaw = raw.signal("ref_yaw_rad", "rad")[tail]
            cx, cy = raw.signal("current_x_mps", "m/s")[tail], raw.signal("current_y_mps", "m/s")[tail]
            meas = cx * np.cos(yaw) + cy * np.sin(yaw)
            misalign = float(np.mean(np.abs(np.arctan2(-cx * np.sin(yaw) + cy * np.cos(yaw), meas))))
            if misalign > cfg.max_station_heading_error_rad:
                raise IdentificationError(
                    f"{raw.trajectory_id}: current is {misalign:.2f} rad off the surge axis on average "
                    f"(limit {cfg.max_station_heading_error_rad}); the F model needs heading into the current"
                )
            add(
                _derived(
                    raw,
                    raw.time_s[tail],
                    {"hold_force_n": hold, "measured_current_mps": meas},
                    {"hold_force_n": "N", "measured_current_mps": "m/s"},
                ),
                d.split,
                "F_station_keeping",
            )
        elif k is ExperimentKind.STATIC_TRIM:
            if d.variant is Variant.TILT_THRUSTERS:
                # plateaus of the commanded pitch moment (the hold on other axes changes commands every tick)
                if "wrench_cmd_my_nm" in raw.signals:
                    key = raw.signal("wrench_cmd_my_nm", "N*m")[:, None]
                else:
                    key = np.stack([raw.signal(f"cmd_{t}", "unitless") for t in ids], axis=1)
                moment = _body_thrust(raw, models, ids, b, 4)
            else:
                moment = raw.signal("applied_pitch_moment_nm", "N*m")
                key = moment[:, None]
            pitch = raw.signal("ref_pitch_rad", "rad")
            tails = _plateaus(key, cfg.plateau_settled_fraction)
            if len(tails) < 3:
                raise IdentificationError(f"{raw.trajectory_id}: fewer than 3 settled plateaus")
            m = np.array([float(np.mean(moment[s])) for s in tails])
            th = np.array([float(np.mean(pitch[s])) for s in tails])
            add(
                _derived(
                    raw,
                    np.arange(m.size, dtype=np.float64),
                    {"applied_pitch_moment_nm": m, "pitch_rad": th},
                    {"applied_pitch_moment_nm": "N*m", "pitch_rad": "rad"},
                ),
                d.split,
                "A_trim",
            )
    ds = IdentificationDataset(str(doc["split_id"]), ident, valid)

    def split_of(key: str, split: str) -> list[LogSegment]:
        return [s for k2, sp, s in pending if k2 == key and sp == split]

    env = cfg.envelope_rmse.get
    for tid in sorted(b_segments):
        held = split_of(f"B_thruster_{tid}", "validation")
        if held:
            bkey = f"B_thruster_{tid}"
            validations[bkey] = validate_thruster(ds, held, fits[bkey], dz_by[tid], env(bkey))
    for kind in _AXES:
        axis_segs = split_of(kind.value, "identification")
        if not axis_segs:
            continue
        rigid = float(inertia[2]) if kind is ExperimentKind.YAW_ROTATION else mass
        fits[kind.value] = identify_axis(axis_segs, kind, rigid, fit_bias=kind is ExperimentKind.HEAVE)
        held = split_of(kind.value, "validation")
        if held:
            validations[kind.value] = validate_axis(ds, held, kind, rigid, fits[kind.value], env(kind.value))

    volume: float | None = None
    scale = split_of("A_displaced_volume", "identification")
    if scale:
        fits["A_displaced_volume"] = identify_displaced_volume(scale, mass)
        volume = fits["A_displaced_volume"].value("displaced_volume_m3")
    if ExperimentKind.HEAVE.value in fits and len(rho_values) == 1:
        heave = fits[ExperimentKind.HEAVE.value]
        rho = next(iter(rho_values))
        bias = heave.get("bias")
        v_heave = (mass * G0 + bias.value) / (rho * G0)
        sigma = None if bias.sigma is None else bias.sigma / (rho * G0)
        fits["D_HEAVE_displaced_volume"] = heave.model_copy(
            update={
                "parameters": (
                    FittedParameter(
                        name="displaced_volume_m3",
                        value=v_heave,
                        sigma=sigma,
                        units="m^3",
                        ci95=None if sigma is None else (v_heave - 1.96 * sigma, v_heave + 1.96 * sigma),
                    ),
                ),
                "message": "derived: V = (m g + heave bias) / (rho g)",
                "correlation": None,
            }
        )
        notes.append("displaced volume derived from the D_HEAVE net-buoyancy bias and the measured mass")
        volume = v_heave if volume is None else volume
    trim = split_of("A_trim", "identification")
    if trim:
        if volume is None or len(rho_values) != 1:
            notes.append(
                "A_trim skipped: buoyancy needs a displaced volume (A scale or D_HEAVE) and one water density"
            )
        else:
            buoyancy = next(iter(rho_values)) * G0 * volume
            fits["A_trim"] = identify_trim(trim, buoyancy)
            held = split_of("A_trim", "validation")
            if held:
                validations["A_trim"] = validate_trim(ds, held, buoyancy, fits["A_trim"], env("A_trim"))
    sk = split_of("F_station_keeping", "identification")
    if sk:
        surge = fits.get(ExperimentKind.SURGE_ACCELERATION.value)
        if surge is None:
            notes.append("F skipped: the current fit needs the C surge drag")
        else:
            d1, d2 = surge.value("linear_damping"), surge.value("quadratic_damping")
            fits["F_station_keeping"] = identify_station_keeping_current(
                [
                    _derived(s, s.time_s, {"hold_force_n": s.signals["hold_force_n"]}, {"hold_force_n": "N"})
                    for s in sk
                ],
                d1,
                d2,
            )
            held = split_of("F_station_keeping", "validation")
            if held:
                validations["F_station_keeping"] = validate_station_keeping(
                    ds, held, d1, d2, env("F_station_keeping")
                )
    report = build_report(ds, fits, validations, tuple(notes))
    return PipelineResult(intake, report, models, notes)
