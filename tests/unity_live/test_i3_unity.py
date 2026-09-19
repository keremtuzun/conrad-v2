"""Gate I3 FORMAL path: Twin2S + Twin2T -> Unity -> sensors -> Model2S + Model2T (partial infrastructure).

One I3-UNITY mission on a held-out FINAL-partition world. The robot stays on the -Y transit lane; the target
segment's defect patch is on the far side. Structural readings are Twin2T measurements whose visibility the
Twin2S oracle evaluates at the Unity TRUE pose (truth side only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from unity_gate_support import GATE_RUNS, MISSION_CONFIG, i3_seed, leakage_scan, measured, reset_measured

from conrad.schemas.belief import UpdateKind
from conrad.schemas.world import Domain
from conrad.sim.mission.unity_run import prepare_unity, replay_unity_run

GATE = "I3"
HIDDEN_UO_MIN = 0.9  # same bound as the surrogate I3 test


@pytest.fixture(scope="module")
def run():
    seed = i3_seed()
    root = GATE_RUNS / GATE
    s = prepare_unity("I3-UNITY", MISSION_CONFIG, runs_root=root, seed=seed)
    try:
        s.run()
        out = s.finish()
    except BaseException:
        s.abort()
        raise
    reset_measured(
        GATE,
        {
            "scenario": "I3-UNITY",
            "seed": seed,
            "partition": "mission/final_test",
            "run_dir": str(Path(out["run_dir"]).relative_to(root.parents[3])),
        },
    )
    return s, Path(out["run_dir"])


def _revs(s, domain):
    return [r for r in s.repo.all_revisions() if r.cell.domain is domain]


def test_twin2t_twin2s_unity_to_model2s_and_model2t(run):
    s, _ = run
    labels = {x["observation_id"]: x["world_id"] for x in s.world.recorder.series["structural_label"]}
    to_world = {str(r): str(w) for r, w in s.world.mapping.to_world.items()}
    matched = [a for a in s.runtime.perception.structural_log if a["registry_id"] is not None]
    wrong = [a for a in matched if to_world[a["registry_id"]] != labels[a["observation_id"]]]
    s_direct = [r for r in _revs(s, Domain.SPATIAL) if r.update_kind is UpdateKind.DIRECT]
    t_direct = [r for r in _revs(s, Domain.TECHNICAL) if r.update_kind is UpdateKind.DIRECT]
    frames = dict(s.world.hardware.forwarded_counts)
    measured(
        GATE,
        "Twin2T + Twin2S -> Unity -> ECMER -> 2S + 2T",
        structural_observations=len(labels),
        associated=len(matched),
        wrong_associations=len(wrong),
        no_match=s.runtime.perception.stats.structural_no_match,
        unity_frames_forwarded=frames,
        model2s_direct_revisions=len(s_direct),
        model2t_direct_revisions=len(t_direct),
        adapter=s.world.hardware.adapter_name,
    )
    assert frames.get("range_imager", 0) > 0 and frames.get("sonar", 0) > 0
    assert matched and not wrong and s_direct and t_direct


def test_robot_sees_only_partial_infrastructure(run):
    s, _ = run
    patch = s.world.recorder.series["patch_visibility"]
    seen = {x["world_id"] for x in s.world.recorder.series["structural_label"]}
    technical = {str(w) for w in s.world.mapping.to_world.values()}
    max_patch = max(p["visible_fraction"] for p in patch)
    measured(
        GATE,
        "robot sees only partial infrastructure",
        patch_max_visible_fraction=max_patch,
        patch_checks=len(patch),
        components_observed=len(seen & technical),
        components_total=len(technical),
    )
    assert max_patch == 0.0  # truth oracle at the Unity true pose: the far-side defect is never seen
    assert 0 < len(seen & technical) < len(technical)


# Was a strict xfail (GATE I3 = FAIL: a never-observed component carried an INFERRED condition). Cause: the mission
# runtime ran Model2T with TCDP. Fixed by ADR-0009 (no relational propagation in production); verified on the
# surrogate path only (docs/audits/MODEL2T_REPAIR.md iteration 3). Not re-run on Unity by that change.
def test_persistent_technical_belief(run):
    s, _ = run
    revs = _revs(s, Domain.TECHNICAL)
    direct = [r for r in revs if r.update_kind is UpdateKind.DIRECT]
    observed = [
        r for r in direct if (c := r.cell.claim("corrosion_depth_m")) and c.status.value == "OBSERVED"
    ]
    # Hidden = registry components that no structural reading ever saw (truth labels). Since world.py (2026-09-19
    # 16:49) samples the near side of the target segment too, the target itself is partially seen; its far-side
    # defect patch is checked in test_robot_sees_only_partial_infrastructure.
    seen = {x["world_id"] for x in s.world.recorder.series["structural_label"]}
    hidden = {r for r, w in s.world.mapping.to_world.items() if str(w) not in seen}
    heads = {m.world_entity_id: m for m in s.runtime.m2t.export_beliefs() if m.world_entity_id is not None}
    # A registry component without any Model2T belief (e.g. a grouping parent) claims nothing: NO_BELIEF.
    hidden_state = {
        str(r): (
            next(c for c in heads[r].state_summary if c.name == "condition").status.value,
            heads[r].uncertainty.observational,
        )
        if r in heads
        else ("NO_BELIEF", 1.0)
        for r in hidden
    }
    target = s.world.mapping.to_registry[s.world.target]
    tcond = next(c for c in heads[target].state_summary if c.name == "condition")
    per_belief: dict[str, int] = {}
    for r in direct:
        per_belief[str(r.belief_id)] = max(per_belief.get(str(r.belief_id), 0), r.revision)
    measured(
        GATE,
        "persistent technical belief",
        technical_revisions=len(revs),
        direct_revisions=len(direct),
        direct_with_evidence=sum(bool(r.consumed_evidence_ids) for r in direct),
        corrosion_observed_revisions=len(observed),
        max_revision=max(per_belief.values(), default=0),
        hidden_components=len(hidden),
        hidden_component_condition_and_U_O=list(hidden_state.values()),
        target_condition=tcond.status.value,
        target_U_O=heads[target].uncertainty.observational,
        target_near_side_readings=sum(
            1 for x in s.world.recorder.series["structural_label"] if x["world_id"] == str(s.world.target)
        ),
    )
    assert direct and all(r.consumed_evidence_ids for r in direct) and observed
    assert max(per_belief.values()) > 0  # revised in place, not re-created
    believed = [(st, uo) for st, uo in hidden_state.values() if st != "NO_BELIEF"]
    assert believed and all(st == "UNKNOWN" and uo >= HIDDEN_UO_MIN for st, uo in believed)


def test_no_twin_truth_leakage_on_runtime_side(run):
    _, run_dir = run
    meta = json.loads((run_dir / "truth" / "truth_record.json").read_text(encoding="utf-8"))["meta"]
    scanned, violations = leakage_scan(run_dir, set(meta["world_entity_ids"]), set())
    measured(GATE, "leakage", texts_scanned=scanned, violations=violations[:10])
    assert scanned > 100 and not violations


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
