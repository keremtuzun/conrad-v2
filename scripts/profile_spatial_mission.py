"""Development-only kernel mission profile; never records gate evidence."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sqlite3
import time
import tracemalloc
from pathlib import Path
from typing import Any

from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.schemas.structural_sensor import StructuralSensorModelV2
from conrad.schemas.structural_support import ParameterAuthority
from conrad.settings import load_settings
from conrad.sim.mission.options import MissionWorldOptions, SpatialTruthOptions
from conrad.sim.mission.run import prepare


def _sensor() -> StructuralSensorModelV2:
    return StructuralSensorModelV2(
        footprint_width_m=1.0,
        footprint_height_m=1.0,
        axial_resolution_m=0.2,
        lateral_resolution_m=0.2,
        minimum_resolvable_corrosion_m=0.05,
        minimum_resolvable_crack_m=0.05,
        range_min_m=0.3,
        range_max_m=4.0,
        noise_sigma_m=0.0,
        position_uncertainty_m=0.0,
        orientation_uncertainty_rad=0.0,
        footprint_uncertainty_m=0.0,
        authority=ParameterAuthority.ENGINEERING_ESTIMATE,
    )


def _settings(spatial: bool, duration_s: float) -> tuple[MissionWorldOptions, MissionRuntimeConfig]:
    base: dict[str, Any] = {
        "family": "pipeline_with_supports",
        "survey_sigma_m": 0.0,
        "ecological_enabled": False,
    }
    runtime: dict[str, Any] = {
        "duration_s": duration_s,
        "control_period_s": 0.1,
        "model2e_enabled": False,
    }
    if spatial:
        sensor = _sensor()
        base.update(
            twin2t_truth_model="spatial_v1",
            spatial_truth=SpatialTruthOptions(axial_cells=2, sectors=4),
            spatial_sensor_model=sensor,
        )
        runtime.update(
            model2t_backend="spatial_v1",
            max_plans_per_need=24,
            decision={"max_information_attempts": 24},
            model2t_spatial={
                "axial_cells": 2,
                "sectors": 4,
                "sensor": sensor.model_dump(mode="json"),
                "thresholds": {
                    "corrosion_degraded_m": 0.002,
                    "corrosion_severe_m": 0.006,
                    "corrosion_failed_m": 0.012,
                    "crack_degraded_m": 0.003,
                    "crack_severe_m": 0.01,
                    "crack_failed_m": 0.03,
                },
                "required_looks": 1,
                "required_domain": {"axial_fraction": [0.05, 0.95], "sectors": [2, 3]},
            },
        )
    return MissionWorldOptions.model_validate(base), MissionRuntimeConfig.model_validate(runtime)


def _counts(run_dir: Path) -> dict[str, int]:
    con = sqlite3.connect(str(run_dir / "conrad.sqlite"))
    try:
        return {
            table: int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("observations", "evidence", "belief_revisions", "commands")
        }
    finally:
        con.close()


def profile(out: Path, label: str, *, spatial: bool, duration_s: float) -> dict[str, Any]:
    run_dir = out / label
    if run_dir.exists():
        raise FileExistsError(f"refusing to replace profile run {run_dir}")
    world, runtime = _settings(spatial, duration_s)
    settings = load_settings("configs/sim/mission_test_small.yaml")
    tracemalloc.start()
    started = time.perf_counter()
    session = prepare(
        "GOLDEN-SMOKE",
        settings,
        run_id=label,
        runs_root=out,
        stored_world=world,
        stored_runtime=runtime,
    )
    prepared = time.perf_counter()
    cpu = cProfile.Profile()
    cpu.enable()
    try:
        session.run()
    finally:
        cpu.disable()
    ran = time.perf_counter()
    session.finish()
    finished = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu.dump_stats(str(out / f"{label}.pstats"))
    stats: Any = pstats.Stats(cpu)
    selected = (
        "spatial_mission.py",
        "spatial_local.py",
        "spatial_support.py",
        "spatial_sensing.py",
        "active/planner.py",
        "active/surface_predictive.py",
        "active/gap.py",
        "active/eig.py",
    )
    cpu_functions = [
        {
            "file": Path(key[0]).name,
            "line": key[1],
            "function": key[2],
            "calls": values[1],
            "self_cpu_s": values[2],
            "cumulative_cpu_s": values[3],
        }
        for key, values in stats.stats.items()
        if any(name in key[0].lower().replace("\\", "/") for name in selected)
    ]
    cpu_functions.sort(key=lambda row: row["cumulative_cpu_s"], reverse=True)
    return {
        "backend": "spatial_v1" if spatial else "legacy",
        "run_dir": str(run_dir),
        "seed": settings.run.seed,
        "duration_s": runtime.duration_s,
        "prepare_wall_s": prepared - started,
        "run_wall_s": ran - prepared,
        "finish_wall_s": finished - ran,
        "peak_traced_python_bytes": peak,
        "bundle_bytes": sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file()),
        "rows": _counts(run_dir),
        "cpu_functions": cpu_functions[:30],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new profile directory")
    parser.add_argument("--duration-s", type=float, default=24.0, help="simulated mission duration")
    parser.add_argument("--spatial-only", action="store_true", help="profile only spatial_v1")
    args = parser.parse_args()
    if args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    out: Path = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to reuse profile directory {out}")
    out.mkdir(parents=True)
    rows = []
    modes = (
        (("PROFILE-SPATIAL-DEV", True),)
        if args.spatial_only
        else (
            ("PROFILE-LEGACY-DEV", False),
            ("PROFILE-SPATIAL-DEV", True),
        )
    )
    for label, spatial in modes:
        rows.append(profile(out, label, spatial=spatial, duration_s=args.duration_s))
        (out / "profile.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
