"""Run a fresh development-only matched-geometry legacy or spatial mission.

The source must be a completed Spatial V1 development bundle. Its seed,
scenario, geometry, deployment settings and link are reused. The legacy arm
uses its historical scalar structural model and a declared zero-defect or
local-defect truth control; the truth representation is therefore different.
This is an impact audit, never I4/I7 validation or formal evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.settings import ConradSettings
from conrad.sim.mission.options import MissionWorldOptions
from conrad.sim.mission.run import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("legacy", "spatial_v1"), required=True)
    parser.add_argument("--duration-s", type=float, default=120.0)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() or source == output or source in output.parents:
        raise SystemExit("output must be absent and outside the source bundle")
    if args.duration_s <= 0:
        raise SystemExit("duration must be positive")

    inputs = json.loads((source / "bundle_manifest.json").read_text(encoding="utf-8"))["replay_inputs"]
    world_raw = dict(inputs["mission_world_options"])
    runtime_raw = dict(inputs["mission_runtime_config"])
    if world_raw["twin2t_truth_model"] != "spatial_v1" or runtime_raw["model2t_backend"] != "spatial_v1":
        raise SystemExit("source must be a Spatial V1 mission")
    if args.mode == "legacy":
        world_raw["twin2t_truth_model"] = "legacy"
        world_raw["spatial_truth"] = None
        world_raw["spatial_sensor_model"] = None
        # Preserve geometry and seed. This is a truth-representation control,
        # not a claim that the two truth fields have identical local defects.
        local_defect = "LOCAL" in source.name
        world_raw["defect"] = {
            **world_raw["defect"],
            "corrosion_depth_m": 0.008 if local_defect else 0.0,
            "crack_length_m": 0.0,
            "pristine_rest": True,
        }
        runtime_raw["model2t_backend"] = "legacy"
        runtime_raw["model2t_spatial"] = None
    runtime_raw["duration_s"] = args.duration_s
    world = MissionWorldOptions.model_validate(world_raw)
    runtime = MissionRuntimeConfig.model_validate(runtime_raw)
    settings = ConradSettings.model_validate(inputs["config_resolved"])

    print(
        json.dumps(
            {
                "status": "STARTED",
                "mode": args.mode,
                "source_commit": inputs["git_commit"],
                "source_bundle": str(source),
                "output": str(output),
                "duration_s": args.duration_s,
                "seed": settings.run.seed,
                "truth_comparability": "matched geometry and seed; different truth representation",
            }
        ),
        flush=True,
    )
    session = prepare(
        inputs["scenario_id"],
        settings,
        run_id=output.name,
        runs_root=output.parent,
        stored_world=world,
        stored_runtime=runtime,
    )
    primary = session.runtime.shore.arms["primary"]
    build = primary.sender.builder.belief_unit
    offered_wire = {"units_built": 0, "f1_bits": 0, "all_fidelity_bits": 0}

    def count_built(*build_args: Any, **build_kwargs: Any) -> Any:
        made = build(*build_args, **build_kwargs)
        if made is not None:
            options = made[0].unit.fidelity_levels
            offered_wire["units_built"] += 1
            offered_wire["f1_bits"] += next(int(o.size_bits) for o in options if int(o.fidelity) == 1)
            offered_wire["all_fidelity_bits"] += int(options[-1].size_bits)
        return made

    primary.sender.builder.belief_unit = count_built  # type: ignore[method-assign]
    ticks = round(runtime.duration_s / runtime.control_period_s)
    progress_ticks = max(1, round(10.0 / runtime.control_period_s))
    for tick in range(ticks):
        session.step()
        if (tick + 1) % progress_ticks == 0:
            print(
                json.dumps({"status": "RUNNING", "simulated_s": (tick + 1) * runtime.control_period_s}),
                flush=True,
            )
    link_report = session.runtime.shore.harness_report()
    result = session.finish()
    (output / "reports" / "impact_link_diagnostics.json").write_text(
        json.dumps({"harness": link_report, "offered_wire": offered_wire}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": "COMPLETED", "result": result}, default=str), flush=True)


if __name__ == "__main__":
    main()
