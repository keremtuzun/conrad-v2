"""Gate I6 FORMAL path: one multi-domain mission (Twin2S + Twin2T + Twin2E -> Unity -> 2S + 2T + 2E -> Model1).

Worlds and arms are declared in ``configs/eval/i6_multidomain.yaml`` BEFORE any run: unity_gate final_test worlds
7800014-7800016, each flown twice (strictly sequential, one player at a time):

* ``I6-UNITY-TURBID``: a Twin2E turbidity spike during the lane survey. Unity renders it through the Twin2E optics
  grid (camera transmission); the structural readings and the 2E probe/survey are rendered from Twin2E at the Unity
  TRUE pose on the Python side, exactly as on the kernel path.
* ``I6-UNITY-CLEAR``: the same world without the spike (paired control).

The measurements and the per-criterion pass rules are the surrogate's (``conrad.evaluation.multidomain``, decision
rule in the config): a criterion passes iff it passes on all three worlds. Needs the player rebuilt with
``EcologyScene.cs`` (optics_grid / fouling_cover); an old player refuses the ``optics_grid`` primitive.
"""

from __future__ import annotations

import faulthandler
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from unity_gate_support import GATE_RUNS, GATES, MISSION_CONFIG, leakage_scan, measured, reset_measured

from conrad.evaluation import partitions as P
from conrad.evaluation.multidomain import (
    authority_ok,
    beliefs_ok,
    i6_measurements,
    reasoning_ok,
    run_with_truth,
)
from conrad.settings import REPO_ROOT
from conrad.sim.mission.unity_run import player_identity, prepare_unity, replay_unity_run
from conrad.sim.unity.player import find_player

GATE = "I6"
I6_CONFIG = REPO_ROOT / "configs" / "eval" / "i6_multidomain.yaml"
RESULTS = GATES / GATE / "unity_i6_results.json"
C_BELIEFS = "one mission produces 2S, 2T and 2E beliefs"
C_REASON = "Model1 reasons across all three via the Belief Bus"
C_AUTH = "children remain authoritative within domains"
I6_HARD_TIMEOUT_S = 2 * 3600.0  # 6 sequential 120 s Unity missions


def _player_has_ecology() -> bool:
    """True only when the built player's managed assembly contains the Twin2E scene support (EcologyScene.cs).

    Guards the held-out worlds: a stale player would be launched on a final world only to refuse the optics grid."""
    exe = find_player()
    if exe is None:
        return False
    dll = exe.parent / f"{exe.stem}_Data" / "Managed" / "ConradUnityV2.dll"
    if not dll.is_file():
        return False
    data = dll.read_bytes()  # C# string literals are stored as UTF-16 in the assembly's user-string heap
    return "optics_grid".encode("utf-16-le") in data or b"optics_grid" in data


pytestmark = pytest.mark.skipif(
    not _player_has_ecology(),
    reason="Unity player absent or built before EcologyScene.cs (no optics_grid support): rebuild it with "
    "BuildScript.BuildWindows64Player before recording I6",
)


@pytest.fixture(autouse=True)
def _hard_timeout() -> Iterator[None]:  # overrides the conftest guard for this module only
    faulthandler.dump_traceback_later(I6_HARD_TIMEOUT_S, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _config() -> dict[str, Any]:
    return dict(yaml.safe_load(I6_CONFIG.read_text(encoding="utf-8")))


def _worlds(cfg: dict[str, Any]) -> list[int]:
    seeds = [int(s) for s in cfg["final_worlds"]]
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        allowed = set(P.split("unity_gate", "final_test", "final_evaluation").world_seeds)
    assert set(seeds) <= allowed, "I6 worlds must be unity_gate final_test worlds"
    return seeds


def _fly(seed: int, arm: str, scenario: str, root: Path) -> dict[str, Any]:
    run_id = f"{scenario}-s{seed}"
    s = prepare_unity(scenario, MISSION_CONFIG, run_id=run_id, runs_root=root, seed=seed)
    try:
        truth = run_with_truth(s)
        m = i6_measurements(s, truth)
        camera = _camera_optics(s)
        eco = None if s.world.eco_report is None else s.world.eco_report.model_dump(mode="json")
        updates = len(s.world.optics_updates)
        out = s.finish()
    except BaseException:
        s.abort()
        raise
    return {
        "seed": seed,
        "arm": arm,
        "scenario": scenario,
        "run_dir": str(Path(out["run_dir"]).relative_to(REPO_ROOT)),
        "measured": m,
        "truth": truth,
        "camera": camera,
        "eco_conversion": eco,
        "optics_updates": updates,
        "pass": {
            C_BELIEFS: beliefs_ok(m[C_BELIEFS]),
            C_REASON: reasoning_ok(m[C_REASON], arm),
            C_AUTH: authority_ok(m[C_AUTH]),
        },
    }


def _camera_optics(s: Any) -> dict[str, Any]:
    """Transmission the Unity camera applied (sensor context of the stored RGB observations)."""
    import sqlite3

    con = sqlite3.connect(str(s.run_dir / "conrad.sqlite"))
    try:
        rows = [json.loads(p) for (p,) in con.execute("SELECT payload_json FROM observations")]
    finally:
        con.close()
    rgb = [r for r in rows if r.get("modality") == "RGB"]
    with_grid = [r for r in rgb if (r.get("sensor_context") or {}).get("optics_grid")]
    trans = [float(r["sensor_context"]["mean_transmission"]) for r in with_grid]
    return {
        "rgb_frames": len(rgb),
        "rgb_frames_with_optics_grid": len(with_grid),
        "mean_transmission_min": min(trans) if trans else None,
        "mean_transmission_max": max(trans) if trans else None,
        "mean_transmission_last": trans[-1] if trans else None,
    }


@pytest.fixture(scope="module")
def flights() -> dict[str, Any]:
    cfg = _config()
    seeds = _worlds(cfg)
    root = GATE_RUNS / GATE
    runs = [_fly(seed, arm, scen, root) for seed in seeds for arm, scen in cfg["unity_arms"].items()]
    agree_min = float(cfg["truth_agreement_min"])
    for r in runs:
        a = r["measured"][C_REASON]["gate_truth_agreement"]
        r["pass"][C_REASON] = bool(r["pass"][C_REASON] and a is not None and a >= agree_min)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment_id": cfg["experiment_id"],
                "evidence_kind": "FORMAL",
                "partition_file": "configs/eval/partitions_unity_gates.yaml",
                "partition_digest": P.load_unity_gates()["digest"],
                "config": str(I6_CONFIG.relative_to(REPO_ROOT)),
                "decision_rule": cfg["decision_rule"],
                "worlds": seeds,
                "player": player_identity(),
                "runs": runs,
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    reset_measured(
        GATE,
        {
            "worlds": seeds,
            "arms": cfg["unity_arms"],
            "partition": "configs/eval/partitions_unity_gates.yaml final_test (declared in i6_multidomain.yaml)",
            "results": str(RESULTS.relative_to(REPO_ROOT)),
        },
    )
    return {"runs": runs, "seeds": seeds}


def _per_world(flights: dict[str, Any], criterion: str) -> dict[str, bool]:
    return {
        str(s): all(r["pass"][criterion] for r in flights["runs"] if r["seed"] == s) for s in flights["seeds"]
    }


def _summary(flights: dict[str, Any], criterion: str) -> dict[str, Any]:
    return {f"{r['scenario']}-s{r['seed']}": r["measured"][criterion] for r in flights["runs"]}


def test_one_mission_produces_2s_2t_2e_beliefs(flights):
    per = _per_world(flights, C_BELIEFS)
    measured(GATE, C_BELIEFS, per_world=per, runs=_summary(flights, C_BELIEFS))
    assert all(per.values()), per


def test_model1_reasons_across_all_three_via_the_belief_bus(flights):
    per = _per_world(flights, C_REASON)
    measured(GATE, C_REASON, per_world=per, runs=_summary(flights, C_REASON))
    assert all(per.values()), per


def test_children_remain_authoritative_within_domains(flights):
    per = _per_world(flights, C_AUTH)
    measured(GATE, C_AUTH, per_world=per, runs=_summary(flights, C_AUTH))
    assert all(per.values()), per


def test_twin2e_reaches_the_unity_camera(flights):
    """The optics grid was loaded and refreshed; the camera applied it; turbid water transmits less than clear."""
    rows = {
        f"{r['arm']}-s{r['seed']}": {**r["camera"], "optics_updates": r["optics_updates"]}
        for r in flights["runs"]
    }
    conv = {f"s{r['seed']}": r["eco_conversion"] for r in flights["runs"] if r["arm"] == "TURBID"}
    measured(GATE, "twin2e -> unity", camera=rows, eco_conversion=conv)
    for r in flights["runs"]:
        assert r["optics_updates"] > 0 and r["camera"]["rgb_frames_with_optics_grid"] > 0
    for seed in flights["seeds"]:
        t = next(r for r in flights["runs"] if r["seed"] == seed and r["arm"] == "TURBID")["camera"]
        c = next(r for r in flights["runs"] if r["seed"] == seed and r["arm"] == "CLEAR")["camera"]
        assert t["mean_transmission_last"] < c["mean_transmission_last"]


def test_no_twin_truth_leakage_on_runtime_side(flights):
    rows = {}
    bad = []
    for r in flights["runs"]:
        run_dir = REPO_ROOT / r["run_dir"]
        meta = json.loads((run_dir / "truth" / "truth_record.json").read_text(encoding="utf-8"))["meta"]
        scanned, violations = leakage_scan(run_dir, set(meta["world_entity_ids"]), set())
        rows[f"{r['arm']}-s{r['seed']}"] = {"texts_scanned": scanned, "violations": violations[:10]}
        bad += violations
        assert scanned > 100
    measured(GATE, "leakage", runs=rows)
    assert not bad


def test_bundle_replays_deterministically(flights):
    r = next(x for x in flights["runs"] if x["arm"] == "TURBID")
    rep = replay_unity_run(REPO_ROOT / r["run_dir"], GATE_RUNS / "replay" / GATE)
    measured(
        GATE,
        "replay",
        run=r["run_dir"],
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
            if k in rep
        },
    )
    assert rep["equal"], rep
