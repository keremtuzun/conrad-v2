"""2S-E002 counterfactual ambiguity: W_A != W_B with O(W_A) ~= O(W_B) (verified by Twin2S).

The schedule looks at the front of the pipeline; ``make_counterfactual_pair`` finds a variant differing
only where the schedule never looks. The model receives world A's observations. In the differing cells
(truth_A != truth_B) it must not confidently pick either world. A follow-up schedule that targets the
changed entity (a discriminating observation) should then lower U_O there.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from conrad.domains.spatial.model import Model2S
from conrad.evaluation.spatial_experiments import common as c
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.ids import IdFactory
from conrad.schemas.timebase import stamp
from conrad.twins.twin2s.counterfactual import make_counterfactual_pair
from conrad.twins.twin2s.evaluation import DEFAULT_CONFIDENCE_THRESHOLD, query_points_between
from conrad.twins.twin2s.ocpwe import run_schedule

EXPERIMENT_ID = "2S-E002"


def ambiguity_metrics(
    model: Model2S, pts: np.ndarray, truth_a: np.ndarray, truth_b: np.ndarray
) -> dict[str, float]:
    if len(pts) == 0:  # no differing cell at the query spacing: reported as n = 0, never a fabricated rate
        nan = float("nan")
        keys = ("confident_claim_rate", "confident_world_a_rate", "confident_world_b_rate", "unknown_rate")
        return {"n_differing_cells": 0.0, **dict.fromkeys(keys, nan)}
    s = model.occupancy_state(pts)
    p = s.probability
    status = model.occupancy_status(pts)
    conf = np.maximum(p, 1 - p) >= DEFAULT_CONFIDENCE_THRESHOLD
    says_occ = p > 0.5
    return {
        "n_differing_cells": float(len(pts)),
        "confident_claim_rate": float(conf.mean()),
        "confident_world_a_rate": float((conf & (says_occ == truth_a)).mean()),
        "confident_world_b_rate": float((conf & (says_occ == truth_b)).mean()),
        "unknown_rate": float(np.mean([st is KnowledgeStatus.UNKNOWN for st in status])),
        "inferred_rate": float(np.mean([st is KnowledgeStatus.INFERRED for st in status])),
        "mean_UO": float(s.uncertainty[:, 3].mean()),
        "mean_abs_p_minus_half": float(np.abs(p - 0.5).mean()),
    }


def run_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    world = c.build_world(seed, config)
    mcfg = c.model_config(config)
    res = mcfg.grid.base_voxel_m
    segs = world.group("segments")
    front = segs[: int(config.get("front_segments", 1))]
    twin = world.twin()
    cond = c.conditions(front, config.get("conditions", {}))
    sched, _ = c.observe(world, twin, cond, seed)
    pair, kind_used, failures = None, None, []
    for kind in config["kinds"]:
        try:
            pair = make_counterfactual_pair(
                world.scenario,
                sched,
                IdFactory(seed + 31),
                np.random.default_rng(seed),
                kind=kind,
                noise_seed=seed,
                max_attempts=int(config.get("max_attempts", 12)),
            )
            kind_used = kind
            break
        except RuntimeError as exc:  # recorded, not swallowed: the kind is reported as not found
            failures.append(f"{kind}: {exc}")
    if pair is None:
        return {"pair_found": 0.0, "failures": failures}
    tw_a, tw_b = world.twin(pair.world_a), world.twin(pair.world_b)
    ids = IdFactory(seed + 13)
    obs = [s.observation for s in run_schedule(tw_a, pair.schedule, world.mission_id, ids.new(), ids.new())]
    spacing = float(config.get("query_spacing_m", res / 2))  # sub-cell probes; the model answers per cell
    lo = np.floor(np.asarray(pair.changed_region_min_m) / res) * res
    hi = np.ceil(np.asarray(pair.changed_region_max_m) / res) * res
    box = query_points_between(lo, hi, spacing)
    ta, tb = c.truth_occupancy(tw_a, box), c.truth_occupancy(tw_b, box)
    diff = ta != tb
    t_end = pair.schedule.views[-1].time_s
    out: dict[str, Any] = {
        "pair_found": 1.0,
        "kind": kind_used,
        "max_observation_difference": pair.report.max_difference,
        "n_views": len(pair.schedule.views),
        "n_box_cells": float(len(box)),
    }
    follow = None
    in_a = {e.entity_id for e in tw_a.world.entities if e.active}
    if not set(pair.changed_entity_ids) <= in_a:
        out["discriminating_followup"] = "skipped: the changed entity exists only in world B"
    else:
        try:
            fcond = c.conditions(pair.changed_entity_ids, config.get("discriminating_conditions", {}))
            follow = c.observe(world, world.twin(pair.world_a), fcond, seed + 1)[1]  # fresh twin of A
        except ValueError as exc:  # OCPWE found no feasible / observable viewpoint; reported, not hidden
            out["discriminating_followup"] = f"not planned: {exc}"
    for kind in config["models"]:
        m = c.run_model(kind, world, obs, mcfg, t_end)
        row = {
            "differing": ambiguity_metrics(m, box[diff], ta[diff], tb[diff]),
            "changed_box": ambiguity_metrics(m, box, ta, tb),
        }
        if follow:
            m.ingest_observations(follow)
            m.update_beliefs(stamp(t_end + 100.0, "SIM"))
            row["after_discriminating_view"] = ambiguity_metrics(m, box[diff], ta[diff], tb[diff])
        out[kind] = row
    return out


def run(config: Mapping[str, Any], seeds: Sequence[int], out_dir: str | Path) -> dict[str, Any]:
    per_seed = {int(s): run_seed(config, int(s)) for s in seeds}
    return c.write_result(EXPERIMENT_ID, config, seeds, per_seed, out_dir)
