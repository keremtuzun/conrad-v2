"""2S-E003 pose uncertainty: estimated pose = true pose + noise with a declared covariance, several sigmas.

The same OCPWE views are used at every sigma (the planner's viewpoint choice happens before the pose
perturbation draws). Compared: UAHSM, UAHSM ignoring pose covariance (ablation), plain occupancy grid.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from conrad.evaluation.spatial_experiments import common as c

EXPERIMENT_ID = "2S-E003"


def run_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    world = c.build_world(seed, config)
    targets = world.group(config.get("target_group", "segments"))
    mcfg = c.model_config(config)
    res = mcfg.grid.base_voxel_m
    rad_per_m = float(config.get("rotation_sigma_rad_per_m", 0.1))
    out: dict[str, Any] = {}
    for sigma in config["pose_sigmas_m"]:
        twin = world.twin()
        cond = c.conditions(
            targets,
            config.get("conditions", {}),
            pose_sigma_m=float(sigma),
            pose_sigma_rad=float(sigma) * rad_per_m,
        )
        sched, obs = c.observe(world, twin, cond, seed)
        pts = c.eval_points(twin, targets, float(config.get("eval_margin_m", 1.0)), res)
        truth = c.truth_occupancy(twin, pts)
        seen = c.observed_mask(twin, sched, pts, res)
        t_end = sched.views[-1].time_s
        row: dict[str, Any] = {"n_views": len(sched.views)}
        for kind in config["models"]:
            row[kind] = c.map_metrics(c.run_model(kind, world, obs, mcfg, t_end), pts, truth, seen)
        out[f"sigma_{sigma}"] = row
    return out


def run(config: Mapping[str, Any], seeds: Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    per_seed = {int(s): run_seed(config, int(s)) for s in seeds}
    return c.write_result(EXPERIMENT_ID, config, seeds, per_seed, out_dir)
