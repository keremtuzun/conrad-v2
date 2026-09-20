"""Gate I6 SURROGATE (python kernel instead of Unity, ADR-0008): one mission, 2S + 2T + 2E, Model1 across all three.

Two layers:

* the retained experiment I6-MULTIDOMAIN-E001 (``scripts/run_i6_multidomain.py --partition final_test``) on the
  worlds declared in ``configs/eval/i6_multidomain.yaml`` (unity_gate final_test 7800014-7800016): the tests re-check
  the stored measurements against the declared decision rule;
* one live mission on a DEVELOPMENT world (unity_gate development), so the harness itself is exercised on every run.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from uuid import UUID

import pytest
import yaml

from conrad.evaluation.multidomain import (
    authority_ok,
    beliefs_ok,
    i6_measurements,
    reasoning_ok,
    run_with_truth,
)
from conrad.evaluation.partitions import split
from conrad.orchestration.mission_config import runtime_config
from conrad.orchestration.multidomain import DEFERRED
from conrad.settings import REPO_ROOT, load_settings
from conrad.sim.mission.run import prepare
from conrad.sim.mission.scenarios import resolve

CONFIG = REPO_ROOT / "configs" / "eval" / "i6_multidomain.yaml"
ARTIFACT = REPO_ROOT / "artifacts" / "experiments" / "I6-MULTIDOMAIN-E001" / "i6_e001.json"
C_BELIEFS = "one mission produces 2S, 2T and 2E beliefs"
C_REASON = "Model1 reasons across all three via the Belief Bus"
C_AUTH = "children remain authoritative within domains"
DEV_WORLD = 7810003  # unity_gate development
DEV_DURATION_S = 50.0


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def e001():
    if not ARTIFACT.exists():
        pytest.fail(f"missing {ARTIFACT}: run scripts/run_i6_multidomain.py --partition final_test")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _runs(e001, arm=None):
    return [r for r in e001["runs"] if arm is None or r["arm"] == arm]


# ------------------------------------------------------------------------------------------ retained experiment
def test_artifact_is_final_split_surrogate(e001, cfg):
    assert e001["partition"] == "final_test" and e001["evidence_class"] == "SURROGATE"
    assert e001["worlds"] == cfg["final_worlds"]
    finals = set(split("unity_gate", "final_test", "final_evaluation").world_seeds)
    assert set(e001["worlds"]) <= finals
    ran = {(r["seed"], r["arm"]) for r in e001["runs"]}
    assert ran == {(s, a) for s in cfg["final_worlds"] for a in cfg["arms"]}


def test_one_mission_produces_2s_2t_2e_beliefs(e001):
    for r in _runs(e001):
        m = r["measured"][C_BELIEFS]
        assert beliefs_ok(m), (r["seed"], r["arm"], m)
    assert e001["summary"][C_BELIEFS]


def test_model1_reasons_across_all_three_via_the_belief_bus(e001):
    agree_min = float(e001["truth_agreement_min"])
    for r in _runs(e001):
        m = r["measured"][C_REASON]
        assert reasoning_ok(m, r["arm"]), (r["seed"], r["arm"], m)
        assert m["gate_truth_agreement"] >= agree_min, (r["seed"], r["arm"], m["gate_truth_agreement"])
    for r in _runs(e001, "TURBID"):
        first = r["measured"][C_REASON]["first_deferral"]
        assert first is not None and all(first["beliefs"][d] for d in ("TECHNICAL", "SPATIAL", "ECOLOGICAL"))
    assert e001["summary"][C_REASON]


def test_children_remain_authoritative_within_domains(e001):
    for r in _runs(e001):
        m = r["measured"][C_AUTH]
        assert authority_ok(m), (r["seed"], r["arm"], m)
        assert m["context_revisions"].get("TECHNICAL", 0) > 0  # 2E reaches 2T, as context only
    assert e001["summary"][C_AUTH]


# ------------------------------------------------------------------------------------------ live development mission
@pytest.fixture(scope="module")
def dev():
    assert DEV_WORLD in split("unity_gate", "development", "design").world_seeds
    s = load_settings("configs/sim/mission_test_small.yaml")
    s = s.model_copy(update={"run": s.run.model_copy(update={"seed": DEV_WORLD})})
    _, runtime_raw = resolve("I6-MULTIDOMAIN-TURBID", dict(s.sim.get("mission", {})))
    rcfg = runtime_config({**runtime_raw, "duration_s": DEV_DURATION_S})  # shorter than the gate missions
    session = prepare(
        "I6-MULTIDOMAIN-TURBID",
        s,
        runs_root=Path(tempfile.mkdtemp(prefix="i6-dev-")),
        stored_runtime=rcfg,
        capture=False,
    )
    try:
        truth = run_with_truth(session)
        yield session, i6_measurements(session, truth)
    finally:
        session.log.close()
        session.engine.dispose()


def test_dev_mission_meets_every_i6_rule(dev):
    _, m = dev
    assert beliefs_ok(m[C_BELIEFS]), m[C_BELIEFS]
    assert reasoning_ok(m[C_REASON], "TURBID"), m[C_REASON]
    assert authority_ok(m[C_AUTH]), m[C_AUTH]


def test_dev_deferral_is_recorded_with_provenance(dev):
    session, _ = dev
    gate = session.runtime.sensing_gate
    deferred = [r for r in gate.records if r["verdict"] == DEFERRED]
    assert deferred
    rec = session.repo.provenance_record(UUID(deferred[0]["provenance_id"]))
    assert rec is not None and rec.operation == "orchestration.sensing_conditions_gate"
    events = [
        e
        for e in session.log.events
        if e.event_type.value == "ACTION_REJECTED" and e.payload.get("reason") == DEFERRED
    ]
    assert len(events) == len(deferred)
    # a deferral is not carried out, so EGDC does not count it as an information attempt
    history = {str(h.decision_id): h for h in session.runtime.deliberation.history}
    assert all(history[r["decision_id"]].executed is False for r in deferred)


def test_multidomain_is_off_by_default():
    from conrad.orchestration.mission_config import MissionRuntimeConfig

    assert MissionRuntimeConfig().multidomain.enabled is False
