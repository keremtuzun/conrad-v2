"""Gate I5 FORMAL path: the Model 1 action matrix plus integrated missions through the built Unity player.

The spec's I5 formal path is "Model1 action-matrix suite on imperfect beliefs + integrated mission", so this
module decides BOTH halves and one recorder command writes all ten criteria:

* the seven action-matrix criteria are belief-level fixtures (M1-ACTION-E001, held-out final split
  7300000-7300009). They are world-independent, so they need no player and are marked ``no_unity_player``:
  they read the stored final artifact, exactly as ``tests/acceptance/test_i5_action_matrix.py`` does.
* the three integrated-mission criteria (ch25 I5 + ch26 Phase 9) fly the eight ``I5-*`` scenarios through
  Unity on the held-out worlds declared in ``configs/eval/i5_unity.yaml`` BEFORE any run, once per arm
  (EGDC, the rule/FSM baseline, the naive act-on-claims baseline), strictly sequentially, one player at a
  time. Scoring is the surrogate's own code
  (``conrad.evaluation.decision_experiments.m1_action_integrated``) with the same thresholds, so the formal
  numbers are comparable with M1-ACTION-E004 line by line.

Nothing here has been executed: the Unity player belongs to another workstream (docs/audits/I5_ACTION_MATRIX.md,
"Iteration 3"). Run it with ``python -m uv run python scripts/record_unity_gate_evidence.py I5``.
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
from conrad.evaluation.decision_experiments.m1_action_integrated import (
    BASELINES,
    PRIMARY,
    SPECS,
    drive_and_score,
    summarize,
    verdicts,
)
from conrad.settings import REPO_ROOT
from conrad.sim.mission.unity_run import player_identity, prepare_unity, replay_unity_run

GATE = "I5"
CONFIG = REPO_ROOT / "configs" / "eval" / "i5_unity.yaml"
RESULTS = GATES / GATE / "unity_i5_results.json"
MATRIX_ARM = "egdc_structured"
MATRIX_CLASSES = ("continue", "request evidence", "replan", "change sensing", "return", "escalate")
C_ACTIONS = "actions exercised correctly inside integrated missions"
C_TRACE = "traceable decisions with low measured UIR"
C_COMPETITIVE = "competitive mission outcomes vs decision baselines"
MIN_PER_CONDITION = 50
# 48 sequential 120 s Unity missions take far longer than the 600 s per-test guard of conftest.py.
I5_HARD_TIMEOUT_S = 6 * 3600.0


@pytest.fixture(autouse=True)
def _hard_timeout() -> Iterator[None]:  # overrides the conftest guard for this module only
    faulthandler.dump_traceback_later(I5_HARD_TIMEOUT_S, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _config() -> dict[str, Any]:
    return dict(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


# ------------------------------------------------------------------ action-matrix half (no player needed)
@pytest.fixture(scope="module")
def matrix() -> dict[str, Any]:
    path = REPO_ROOT / _config()["matrix_artifact"]
    assert path.exists(), f"M1-ACTION-E001 final artifact missing: {path} (run the experiment first)"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["partition"] == "final_test" and data["purpose"] == "final_evaluation"
    return data


def _recall(matrix: dict[str, Any], cls: str) -> dict[str, Any]:
    s = matrix["summary"][MATRIX_ARM]["per_class"][cls]
    floor = float(matrix["acceptance"]["canonical_recall_floor"])
    return {
        "recall": s["recall"],
        "n_canonical": s["n_canonical"],
        "floor": floor,
        "floor_status": "ENGINEERING_ESTIMATE (ch25 leaves the I5 recall bound OPEN)",
        "chosen_as_itself": matrix["summary"][MATRIX_ARM]["confusion_matrix"][cls][cls],
    }


@pytest.mark.no_unity_player
@pytest.mark.parametrize("cls", MATRIX_CLASSES)
def test_action_matrix_class(matrix, cls):
    """The action class is used correctly in its canonical scenarios (belief fixtures, held-out seeds)."""
    m = _recall(matrix, cls)
    measured(GATE, cls, **m)
    assert m["n_canonical"] >= MIN_PER_CONDITION
    assert m["recall"] is not None and m["recall"] >= m["floor"]
    assert m["chosen_as_itself"] >= m["floor"] * m["n_canonical"]


@pytest.mark.no_unity_player
def test_matrix_hard_constraints_inviolable(matrix):
    s = matrix["summary"][MATRIX_ARM]
    hc = s["hard_constraint_violations"]
    measured(
        GATE,
        "hard constraints inviolable",
        chosen_action_audit=hc["chosen_action_audit"],
        injected_invalid_proposals=hc["injected_invalid_proposals"],
        injected_invalid_accepted=hc["injected_invalid_accepted"],
        total=hc["total"],
        uir=s["uir"],
    )
    assert hc["chosen_action_audit"] == 0 and hc["injected_invalid_accepted"] == 0 and hc["total"] == 0
    assert s["uir"]["unsupported_inference_rate"] == 0.0 and s["uir"]["relied_world_claims"] > 0


# ------------------------------------------------------------------ integrated-mission half (Unity)
def _worlds(cfg: dict[str, Any]) -> list[int]:
    seeds = [int(s) for s in cfg["final_worlds"]]
    with P.purpose_scope(P.Purpose.FINAL_EVALUATION):
        allowed = set(P.split(P.I5_UNITY_DOMAIN, "final_test", "final_evaluation").world_seeds)
    assert set(seeds) <= allowed, "I5 Unity worlds must come from partitions_i5_unity.yaml final_test"
    return seeds


def _fly(seed: int, scenario: str, arm: str, cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    run_id = f"I5-UNITY-{scenario}-s{seed}-{arm}"
    scored = root / run_id / "reports" / "i5_score.json"
    if scored.exists() and (root / run_id / "bundle_manifest.json").exists():
        # resume an interrupted session: a finished flight is never re-flown or re-scored
        return json.loads(scored.read_text(encoding="utf-8"))
    s = prepare_unity(scenario, MISSION_CONFIG, run_id=run_id, runs_root=root, seed=seed)
    try:
        row = drive_and_score(s, seed, scenario, arm, cfg)
        out = s.finish()
    except BaseException:
        s.abort()
        raise
    row["run_dir"] = str(Path(out["run_dir"]).relative_to(REPO_ROOT))
    scored.parent.mkdir(parents=True, exist_ok=True)
    scored.write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
    return row


@pytest.fixture(scope="module")
def flights() -> dict[str, Any]:
    cfg = _config()
    seeds = _worlds(cfg)
    scenarios = [str(x) for x in cfg["scenarios"]]
    missing = set(SPECS) - set(scenarios)
    assert set(scenarios) <= set(SPECS), "every declared scenario must be a harness I5 scenario"
    # a scenario the Unity mission path cannot realise must be named, with its reason, in the declaration
    assert missing == set(cfg.get("scenarios_not_on_the_unity_path", {})), missing
    root = GATE_RUNS / GATE
    rows = [
        _fly(seed, scenario, arm, cfg, root)
        for seed in seeds  # strictly sequential: one Unity player at a time
        for scenario in scenarios
        for arm in cfg["arms"]
    ]
    summary = summarize(rows, scenarios)
    verdict = verdicts(summary, cfg, scenarios)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "experiment_id": cfg["experiment_id"],
                "evidence_kind": "FORMAL",
                "evidence_class": "FORMAL (built Unity V2 player, lock-step TCP)",
                "data_status": "SYNTHETIC_ONLY",
                "partition": "final_test",
                "partition_file": "configs/eval/partitions_i5_unity.yaml",
                "partition_digest": P.load_i5_unity()["digest"],
                "config": str(CONFIG.relative_to(REPO_ROOT)),
                "decision_rule": cfg["decision_rule"],
                "worlds": seeds,
                "scenarios": scenarios,
                "arms": list(cfg["arms"]),
                "player": player_identity(),
                **summary,
                "verdicts": verdict,
                "per_run": rows,
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
            "scenarios": scenarios,
            "arms": list(cfg["arms"]),
            "partition": "configs/eval/partitions_i5_unity.yaml final_test (declared in i5_unity.yaml)",
            "matrix_artifact": cfg["matrix_artifact"],
            "results": str(RESULTS.relative_to(REPO_ROOT)),
        },
    )
    return {"rows": rows, "seeds": seeds, "scenarios": scenarios, "summary": summary, "verdicts": verdict}


def test_actions_exercised_correctly_inside_integrated_missions(flights):
    v, e = flights["verdicts"], flights["summary"]["closed_loop"][PRIMARY]
    measured(
        GATE,
        C_ACTIONS,
        per_scenario_correct_rate=v["per_scenario_correct_rate"],
        scenarios_scored=v["scenarios_scored_for_actions"],
        scenarios_not_applicable=v["scenarios_not_applicable"],
        success_floor=v["success_floor"],
        floor_status=v["floor_status"],
        nominal_over_escalations=v["nominal_over_escalations"],
        violations_total=v["violations_total"],
        latency_s_max=e["ALL"]["latency_s_max"],
        warrant_reached=e["ALL"]["warrant_reached"],
        correct=e["ALL"]["correct"],
        n=e["ALL"]["n"],
    )
    assert v["actions_exercised_correctly"], v["per_scenario_correct_rate"]


def test_traceable_decisions_with_low_measured_uir(flights):
    v = flights["verdicts"]
    measured(
        GATE,
        C_TRACE,
        traceable_fraction=v["traceable_fraction"],
        uir=v["uir"],
        uir_max=v["uir_max"],
        relied_world_claims=flights["summary"]["closed_loop"][PRIMARY]["ALL"]["uir"]["relied_world_claims"],
    )
    assert v["traceable_low_uir"], v


def test_competitive_mission_outcomes_vs_decision_baselines(flights):
    v = flights["verdicts"]
    measured(GATE, C_COMPETITIVE, competitive=v["competitive"])
    assert v["competitive_outcomes"], v["competitive"]


def test_every_scenario_and_arm_flew_on_every_world(flights):
    got = {(r["scenario"], r["arm"], r["seed"]) for r in flights["rows"]}
    want = {
        (sc, arm, seed)
        for sc in flights["scenarios"]
        for arm in (PRIMARY, *BASELINES)
        for seed in flights["seeds"]
    }
    measured(GATE, "coverage of the declared grid", flights=len(got), declared=len(want))
    assert got == want


def test_no_twin_truth_leakage_on_runtime_side(flights):
    rows: dict[str, Any] = {}
    bad: list[str] = []
    for r in flights["rows"]:
        run_dir = REPO_ROOT / r["run_dir"]
        meta = json.loads((run_dir / "truth" / "truth_record.json").read_text(encoding="utf-8"))["meta"]
        scanned, violations = leakage_scan(run_dir, set(meta["world_entity_ids"]), set())
        rows[f"{r['scenario']}-s{r['seed']}-{r['arm']}"] = {
            "texts_scanned": scanned,
            "violations": violations[:10],
        }
        bad += violations
        assert scanned > 100
    measured(GATE, "leakage", runs=rows)
    assert not bad


def test_bundle_replays_deterministically(flights):
    r = next(x for x in flights["rows"] if x["arm"] == PRIMARY)
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
