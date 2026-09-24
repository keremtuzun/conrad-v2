"""Profile a fresh spatial development mission from a recorded configuration.

This tool creates a new run under --output and never modifies the source bundle.
Its timings include Python call overhead but exclude final bundle serialization.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import tracemalloc
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch

from conrad.active.planner import MCBRPlanner
from conrad.domains.technical.spatial_mission import SpatialMissionModel2T
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.settings import ConradSettings
from conrad.sim.mission.options import MissionWorldOptions
from conrad.sim.mission.run import prepare
from conrad.sim.mission.sensing import MissionSensorSuite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="completed development bundle")
    parser.add_argument("--output", type=Path, required=True, help="new, absent output directory")
    parser.add_argument("--duration-s", type=float, default=80.0)
    parser.add_argument("--trace-memory", action="store_true", help="trace Python allocations (slows timing)")
    parser.add_argument("--survey-sigma-m", type=float, default=None)
    parser.add_argument("--survey-endpoint-bound-m", type=float, default=None)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or source == output or source in output.parents:
        raise SystemExit("--output must be a new directory outside the source bundle")
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")
    manifest = json.loads((source / "bundle_manifest.json").read_text(encoding="utf-8"))
    inputs = manifest["replay_inputs"]
    options = MissionWorldOptions.model_validate(inputs["mission_world_options"])
    if args.survey_sigma_m is not None or args.survey_endpoint_bound_m is not None:
        if args.survey_sigma_m is None or args.survey_endpoint_bound_m is None:
            raise SystemExit("bounded survey requires both sigma and endpoint bound")
        options = MissionWorldOptions.model_validate(
            {
                **options.model_dump(mode="json"),
                "survey_sigma_m": args.survey_sigma_m,
                "survey_endpoint_bound_m": args.survey_endpoint_bound_m,
            }
        )
    runtime = MissionRuntimeConfig.model_validate(inputs["mission_runtime_config"])
    runtime = runtime.model_copy(update={"duration_s": args.duration_s})
    if options.twin2t_truth_model != "spatial_v1" or runtime.model2t_backend != "spatial_v1":
        raise SystemExit("source must use spatial_v1 truth and belief")

    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "seconds": 0.0})

    def timed(name: str, original: Any) -> Any:
        @wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                totals[name]["calls"] += 1
                totals[name]["seconds"] += time.perf_counter() - start

        return wrapper

    hooks = (
        (MissionSensorSuite, "_spatial_structural", "sensor_support_and_response"),
        (SpatialMissionModel2T, "ingest", "model2t_ingest"),
        (SpatialMissionModel2T, "update_beliefs", "model2t_update"),
        (MCBRPlanner, "plan", "mcbr_plan"),
    )
    from contextlib import ExitStack

    with ExitStack() as stack:
        for cls, attr, name in hooks:
            stack.enter_context(patch.object(cls, attr, timed(name, getattr(cls, attr))))
        session = prepare(
            inputs["scenario_id"],
            ConradSettings.model_validate(inputs["config_resolved"]),
            run_id=output.name,
            runs_root=output.parent,
            stored_world=options,
            stored_runtime=runtime,
            capture=False,
        )
        traced_start = traced_peak = None
        if args.trace_memory:
            tracemalloc.start()
            traced_start, _ = tracemalloc.get_traced_memory()
        ticks = round(runtime.duration_s / runtime.control_period_s)
        start = time.perf_counter()
        cpu_start = time.process_time()
        progress_ticks = max(1, round(10.0 / runtime.control_period_s))
        for tick in range(ticks):
            session.step()
            if (tick + 1) % progress_ticks == 0:
                print(f"profiled {(tick + 1) * runtime.control_period_s:.1f} simulated seconds", flush=True)
        step_seconds = time.perf_counter() - start
        step_cpu_seconds = time.process_time() - cpu_start
        if args.trace_memory:
            _, traced_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        session.finish()
    db = output / "conrad.sqlite"
    with sqlite3.connect(db) as connection:
        belief_count, belief_bytes = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload_json)), 0) FROM belief_revisions"
        ).fetchone()
    result = {
        "source": str(source),
        "run_dir": str(output),
        "duration_s": runtime.duration_s,
        "ticks": ticks,
        "step_wall_seconds": step_seconds,
        "step_cpu_seconds": step_cpu_seconds,
        "seconds_per_simulated_second": step_seconds / runtime.duration_s,
        "profiled_calls": totals,
        "belief_revisions": belief_count,
        "belief_payload_bytes": belief_bytes,
        "python_traced_start_bytes": traced_start,
        "python_traced_peak_bytes": traced_peak,
        "bundle_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
    }
    (output / "reports" / "spatial_profile.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
