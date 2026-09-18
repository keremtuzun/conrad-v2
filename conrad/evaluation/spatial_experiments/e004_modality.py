"""2S-E004 modality dropout / sonar and RGB degradation.

Conditions change which sensors are active and their degradation (Twin2S degradation keys). RGB carries
no range, so Model2S does not use it for geometry in V1: RGB degradation is expected to change nothing,
and RGB-only must leave the map UNKNOWN. The table reports what actually happened.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from conrad.evaluation.spatial_experiments import common as c

EXPERIMENT_ID = "2S-E004"


def run_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    world = c.build_world(seed, config)
    targets = world.group(config.get("target_group", "segments"))
    mcfg = c.model_config(config)
    res = mcfg.grid.base_voxel_m
    out: dict[str, Any] = {}
    for name, spec in config["variants"].items():
        twin = world.twin()
        cond = c.conditions(
            targets,
            config.get("conditions", {}),
            modalities=tuple(spec["modalities"]),
            degradation=spec.get("degradation", {}),
        )
        sched, obs = c.observe(world, twin, cond, seed)
        pts = c.eval_points(twin, targets, float(config.get("eval_margin_m", 1.0)), res)
        truth = c.truth_occupancy(twin, pts)
        seen = c.observed_mask(twin, sched, pts, res)
        t_end = sched.views[-1].time_s
        row: dict[str, Any] = {"n_views": len(sched.views), "n_observations": len(obs)}
        for kind in config["models"]:
            model = c.run_model(kind, world, obs, mcfg, t_end)
            row[kind] = {
                **c.map_metrics(model, pts, truth, seen),
                "integrated_observations": float(model.diagnostics["integrated"]),
            }
        out[name] = row
    return out


def run(config: Mapping[str, Any], seeds: Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    per_seed = {int(s): run_seed(config, int(s)) for s in seeds}
    return c.write_result(EXPERIMENT_ID, config, seeds, per_seed, out_dir)
