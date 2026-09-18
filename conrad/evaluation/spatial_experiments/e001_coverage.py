"""2S-E001 coverage / occlusion: OCPWE schedules at several target coverages -> Model2S vs baselines.

Metrics per model and coverage level: Brier error on observed / hidden cells, IoU, unsupported confident
hidden reconstruction (Twin2S ``unsupported_confidence`` over never-observed cells), calibration (ECE) and
coverage calibration |V_hat - V_true|.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from conrad.evaluation.spatial_experiments import common as c

EXPERIMENT_ID = "2S-E001"


def run_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    world = c.build_world(seed, config)
    targets = world.group(config.get("target_group", "segments"))
    mcfg = c.model_config(config)
    res = mcfg.grid.base_voxel_m
    out: dict[str, Any] = {}
    for level in config["coverage_levels"]:
        twin = world.twin()
        cond = c.conditions(targets, config.get("conditions", {}), target_coverage=float(level))
        sched, obs = c.observe(world, twin, cond, seed)
        pts = c.eval_points(twin, targets, float(config.get("eval_margin_m", 1.0)), res)
        truth = c.truth_occupancy(twin, pts)
        seen = c.observed_mask(twin, sched, pts, res)
        t_end = sched.views[-1].time_s if sched.views else 0.0
        row: dict[str, Any] = {
            "n_views": len(sched.views),
            "planned_coverage": float(sched.achieved.get("cumulative_coverage", 0.0)),
            "n_observations": len(obs),
        }
        for kind in config["models"]:
            row[kind] = c.map_metrics(c.run_model(kind, world, obs, mcfg, t_end), pts, truth, seen)
        out[f"coverage_{level}"] = row
    return out


def run(config: Mapping[str, Any], seeds: Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    per_seed = {int(s): run_seed(config, int(s)) for s in seeds}
    return c.write_result(EXPERIMENT_ID, config, seeds, per_seed, out_dir)
