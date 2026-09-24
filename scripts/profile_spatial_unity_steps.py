"""Measure short kernel and Unity mission step cost with matching spatial options.

Development timing only. Unity startup and bundle finalization are reported
separately; neither is included in the per-step timing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.settings import ConradSettings
from conrad.sim.mission.options import MissionWorldOptions
from conrad.sim.mission.run import MissionSession, prepare
from conrad.sim.mission.unity_run import UnitySession, prepare_unity
from conrad.sim.mission.unity_world import UnityWorldOptions
from conrad.sim.unity.player import find_player


def _summary(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    return {
        "steps": len(samples),
        "total_wall_s": sum(samples),
        "mean_step_wall_s": sum(samples) / len(samples),
        "p95_step_wall_s": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_step_wall_s": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="completed spatial development bundle")
    parser.add_argument("--output", type=Path, required=True, help="new, absent output directory")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if args.steps <= 0 or out.exists() or source in out.parents:
        raise SystemExit("positive --steps and a new output directory outside source are required")
    player = find_player()
    if player is None:
        raise SystemExit("Unity player unavailable")
    inputs = json.loads((source / "bundle_manifest.json").read_text(encoding="utf-8"))["replay_inputs"]
    options = MissionWorldOptions.model_validate(inputs["mission_world_options"])
    runtime = MissionRuntimeConfig.model_validate(inputs["mission_runtime_config"])
    if options.twin2t_truth_model != "spatial_v1" or runtime.model2t_backend != "spatial_v1":
        raise SystemExit("source must use spatial_v1")
    runtime = runtime.model_copy(update={"duration_s": args.steps * runtime.control_period_s})
    settings = ConradSettings.model_validate(inputs["config_resolved"])
    settings = settings.model_copy(update={"run": settings.run.model_copy(update={"seed": 401})})
    out.mkdir(parents=True)
    results: dict[str, Any] = {"seed": 401, "steps": args.steps, "player": str(player)}
    for backend in ("kernel", "unity"):
        started = time.perf_counter()
        session: MissionSession | UnitySession
        if backend == "kernel":
            session = prepare(
                "GOLDEN-SMOKE",
                settings,
                run_id="KERNEL-STEP-PROFILE-DEV",
                runs_root=out,
                stored_world=options,
                stored_runtime=runtime,
                capture=False,
            )
        else:
            session = prepare_unity(
                "I3-UNITY",
                settings,
                run_id="UNITY-STEP-PROFILE-DEV",
                runs_root=out,
                seed=401,
                uopts=UnityWorldOptions(graphics=True, player_path=str(player)),
                stored_world=options,
                stored_runtime=runtime,
                capture=False,
            )
        prepared = time.perf_counter()
        samples = []
        try:
            for _ in range(args.steps):
                tick_start = time.perf_counter()
                session.step()
                samples.append(time.perf_counter() - tick_start)
            stepped = time.perf_counter()
            session.finish()
        except BaseException:
            if isinstance(session, UnitySession):
                session.abort()
            raise
        finished = time.perf_counter()
        results[backend] = {
            "prepare_wall_s": prepared - started,
            "steps_wall_s": stepped - prepared,
            "finish_wall_s": finished - stepped,
            **_summary(samples),
        }
        (out / "profile.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"{backend} {json.dumps(results[backend])}", flush=True)
    results["unity_to_kernel_mean_step_ratio"] = (
        results["unity"]["mean_step_wall_s"] / results["kernel"]["mean_step_wall_s"]
    )
    (out / "profile.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
