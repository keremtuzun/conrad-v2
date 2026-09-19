"""Gate I1 FORMAL path: Twin2S -> Unity -> Unity sensors -> Observation -> geometric encoder -> Model2S.

One I1-UNITY mission on a held-out FINAL-partition world, through the built Unity player. Truth (Twin2S oracle,
Unity truth endpoint) is used only here, to evaluate. Declared tolerances sit next to each assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from unity_gate_support import GATE_RUNS, MISSION_CONFIG, i1_seed, leakage_scan, measured, reset_measured

from conrad.domains.spatial.grid import STATUS_NAMES
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.belief import UpdateKind
from conrad.schemas.frames import WORLD, SpatialSupport
from conrad.schemas.world import Domain
from conrad.sim.mission.unity_run import prepare_unity, replay_unity_run
from conrad.sim.mission.unity_world import range_image_to_model2s, sonar_to_model2s

GATE = "I1"
GEOMETRY_TOL_M = 0.5  # base voxel 0.35 m: half diagonal 0.30 m + pose/range noise
GEOMETRY_MIN_FRACTION = 0.9
FREE_WRONG_MAX = 0.05  # same bound as the surrogate I1 test


@pytest.fixture(scope="module")
def run():
    seed = i1_seed()
    root = GATE_RUNS / GATE
    s = prepare_unity("I1-UNITY", MISSION_CONFIG, runs_root=root, seed=seed)
    try:
        s.run()
        out = s.finish()
    except BaseException:
        s.abort()
        raise
    reset_measured(
        GATE,
        {
            "scenario": "I1-UNITY",
            "seed": seed,
            "partition": "mission/final_test",
            "run_dir": str(Path(out["run_dir"]).relative_to(root.parents[3])),
        },
    )
    return s, Path(out["run_dir"])


def _grid(s):
    return s.runtime.m2s.export_grid().level(0)


def test_robot_moves_through_synthetic_world(run):
    s, _ = run
    tr = np.asarray(s.world.tracker.trajectory)
    path = float(np.sum(np.linalg.norm(np.diff(tr[:, 1:4], axis=0), axis=1)))
    accepted = s.runtime.gateway.accepted
    measured(
        GATE,
        "robot moves through synthetic world",
        true_path_length_m=path,
        duration_s=float(tr[-1, 0]),
        collisions=s.world.tracker.collision_count,
        min_clearance_m=s.world.tracker.min_clearance_m,
        commands_accepted=accepted,
        adapter=s.world.hardware.adapter_name,
    )
    assert path >= 5.0 and s.world.tracker.collision_count == 0 and accepted > 0


def test_persistent_map(run):
    s, _ = run
    revs = [r for r in s.repo.all_revisions() if r.cell.domain is Domain.SPATIAL]
    direct = [r for r in revs if r.update_kind is UpdateKind.DIRECT and r.consumed_evidence_ids]
    max_rev = max(r.revision for r in revs)
    measured(
        GATE,
        "persistent map",
        spatial_revisions=len(revs),
        direct_revisions=len(direct),
        max_revision=max_rev,
        beliefs=len(s.runtime.m2s.export_beliefs()),
    )
    assert direct and max_rev > 0
    assert all(r.cell.spatial_support is not None and r.cell.spatial_support.frame_id == WORLD for r in revs)


def test_geometry_occupied_claims_lie_on_twin2s_surfaces(run):
    s, _ = run
    g = _grid(s)
    occ = (g.status == "OBSERVED") & (g.probability >= s.runtime.m2s.cfg.occupancy.p_occupied)
    d = np.abs(s.world.t2s.world.sdf(g.centers_m[occ]))
    frac = float(np.mean(d <= GEOMETRY_TOL_M)) if occ.any() else 0.0
    measured(
        GATE,
        "geometry",
        occupied_claimed_cells=int(occ.sum()),
        median_abs_sdf_m=float(np.median(d)),
        p90_abs_sdf_m=float(np.percentile(d, 90)),
        fraction_within_tol=frac,
        tol_m=GEOMETRY_TOL_M,
        required_fraction=GEOMETRY_MIN_FRACTION,
        scene_conversion_max_error_m=s.world.scene_report.max_error_m,
    )
    assert occ.sum() >= 20 and frac >= GEOMETRY_MIN_FRACTION


def test_occupancy_free_claims_agree_with_truth(run):
    s, _ = run
    lane = np.asarray(s.world.context.transit_lane)
    rng = np.random.default_rng(3)
    pts = np.concatenate([p + rng.normal(0, 1.0, (200, 3)) for p in lane])
    free = s.runtime.m2s.is_free(pts)
    wrong = float(s.world.t2s.occupancy_truth_at(pts[free]).mean()) if free.any() else 1.0
    measured(
        GATE,
        "occupancy",
        probe_points=len(pts),
        claimed_free=int(free.sum()),
        wrong_fraction=wrong,
        max_wrong=FREE_WRONG_MAX,
    )
    assert free.any() and wrong < FREE_WRONG_MAX


def test_coverage(run):
    s, _ = run
    lane = np.asarray(s.world.context.transit_lane)
    lo, hi = lane.min(0) - 2.0, lane.max(0) + 2.0
    region = SpatialSupport(frame_id=WORLD, center_m=tuple((lo + hi) / 2), half_extent_m=tuple((hi - lo) / 2))
    cov = s.runtime.m2s.coverage(region)
    target = s.world.context.component(s.world.mapping.to_registry[s.world.target]).region(0.3)
    tcov = s.runtime.m2s.coverage(target)
    measured(
        GATE,
        "coverage",
        lane_region_coverage=cov,
        target_segment_coverage=tcov,
        observed_cells=int((_grid(s).status == "OBSERVED").sum()),
    )
    assert cov > 0.0 and tcov > 0.0


def test_observed_inferred_unknown_and_hidden_is_unknown(run):
    s, _ = run
    g = _grid(s)
    counts = {k: int((g.status == k).sum()) for k in STATUS_NAMES}
    world = s.world.t2s.world
    rng = np.random.default_rng(11)
    lo, hi = np.asarray(world.bounds_min), np.asarray(world.bounds_max)
    xy = rng.uniform(lo[:2], hi[:2], (500, 2))
    floor = world.entities[0].primitive.height(xy)  # type: ignore[attr-defined]
    buried = np.c_[xy, floor - 0.8]  # below the seabed: never observable by any sensor
    st = [x.value for x in s.runtime.m2s.occupancy_status(buried)]
    hidden = {k: st.count(k) for k in set(st)}
    measured(GATE, "observed/inferred/unknown", cell_status_counts=counts, hidden_below_seabed=hidden)
    assert counts["OBSERVED"] > 0 and counts["INFERRED"] > 0 and counts["UNKNOWN"] > 0
    assert hidden == {"UNKNOWN": len(buried)}
    assert not s.runtime.m2s.is_free(buried).any()


def test_uncertainty_is_decomposed_and_responds(run):
    s, _ = run
    g = _grid(s)
    u = g.uncertainty
    by = {
        k: u[g.status == k].mean(0).tolist()
        for k in ("OBSERVED", "INFERRED", "UNKNOWN")
        if (g.status == k).any()
    }
    spread = u.std(0).tolist()
    measured(GATE, "uncertainty", channels="UA,UE,UC,UO", mean_by_status=by, std_per_channel=spread)
    assert u.shape[1] == 4
    assert by["UNKNOWN"][3] > by["OBSERVED"][3] + 0.3  # low coverage -> U_O up
    assert sum(x > 1e-6 for x in spread) >= 2  # not constants


def test_provenance_to_raw_unity_frames(run):
    s, run_dir = run
    store = ObjectStore(run_dir / "objects")
    specs = {sp.sensor_id: sp for sp in s.world.context.sensors}
    revs = [
        r
        for r in s.repo.all_revisions()
        if r.cell.domain is Domain.SPATIAL and r.update_kind is UpdateKind.DIRECT
    ]
    ev_ids = {e for r in revs for e in r.consumed_evidence_ids}
    checked, problems = 0, []
    for eid in sorted(ev_ids, key=str):
        ev = s.repo.evidence(eid)
        obs = None if ev is None else s.repo.observation(ev.source_observation_id)
        if obs is None or obs.payload_ref is None:
            problems.append(f"{eid}: evidence/observation missing")
            continue
        src = obs.sensor_context.get("source", {})
        spec = specs[obs.sensor_id]
        raw = np.frombuffer(store.get_bytes(src["raw_payload_digest"]), dtype="<f4").reshape(src["raw_shape"])
        if obs.modality.value == "DEPTH_RANGE":
            again = range_image_to_model2s(raw, spec)
        else:
            again = sonar_to_model2s(raw, float(spec.parameters["max_range_m"]), spec)
        stored = store.get_array(obs.payload_ref)
        if src.get("adapter") != "unity_v2" or not np.array_equal(again, stored, equal_nan=True):
            problems.append(f"{obs.observation_id}: not reproducible from its raw Unity frame")
        checked += 1
    measured(
        GATE,
        "provenance",
        direct_spatial_revisions=len(revs),
        evidence_traced=checked,
        raw_frames_reproduced=checked - len(problems),
        problems=problems[:5],
    )
    assert checked > 0 and not problems


def test_no_twin_truth_leakage_on_runtime_side(run):
    s, run_dir = run
    meta = json.loads((run_dir / "truth" / "truth_record.json").read_text(encoding="utf-8"))["meta"]
    world_ids = set(meta["world_entity_ids"])
    scanned, violations = leakage_scan(run_dir, world_ids, set())
    registry = {c.registry_id for c in s.world.context.design}
    cells = [r.cell for r in s.repo.all_revisions() if r.cell.domain is Domain.SPATIAL]
    reg = {c.registry_entity_id for c in cells if c.registry_entity_id is not None}
    measured(
        GATE,
        "no Twin truth leakage",
        texts_scanned=scanned,
        world_entity_ids=len(world_ids),
        violations=violations[:10],
        registry_ids_on_cells=len(reg),
    )
    assert scanned > 100 and not violations
    assert reg <= registry and not {str(x) for x in reg} & world_ids


def test_bundle_replays_deterministically(run):
    _, run_dir = run
    rep = replay_unity_run(run_dir, GATE_RUNS / "replay" / GATE)
    measured(
        GATE,
        "replay",
        **{
            k: rep[k]
            for k in (
                "events",
                "decisions",
                "revisions",
                "trajectory",
                "equal",
                "verified_files",
                "verified_objects",
                "same_player_binary",
            )
        },
    )
    assert rep["equal"], rep
