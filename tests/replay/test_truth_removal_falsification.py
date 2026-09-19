"""Truth-removal falsification of CC-10: inference + decisions re-run from captured observations only.

A short Python-kernel FLAGSHIP-I4 mission is generated in a temp dir. Its truth and simulator-side entries are
moved out of a copy, the unchanged MissionRuntime is replayed from ``capture/`` in a fresh process under an ``open``
audit hook, and every deployment-side output must be byte-identical. Controls: corrupting truth changes nothing;
altering one captured observation changes the output; an unsealed edit is refused.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

from conrad.settings import load_settings
from conrad.sim.mission.capture import (
    TAPE,
    compare_outputs,
    inference_fingerprint,
    replay_inference,
    verify_capture,
)
from conrad.sim.mission.run import run_scenario

REPO = Path(__file__).resolve().parents[2]
SMALL = "configs/sim/mission_test_small.yaml"


def _script():
    spec = importlib.util.spec_from_file_location(
        "falsify_truth_removal", REPO / "scripts" / "falsify_truth_removal.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


F = _script()


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    s = load_settings(SMALL)
    s = s.model_copy(
        update={"sim": {"mission": {"world": {}, "runtime": {"duration_s": 8.0, "control_period_s": 0.1}}}}
    )
    out = run_scenario("FLAGSHIP-I4", s, runs_root=tmp_path_factory.mktemp("runs"))
    return Path(out["run_dir"])


@pytest.fixture(scope="module")
def stripped(bundle, tmp_path_factory):
    root = tmp_path_factory.mktemp("stripped")
    dst = root / bundle.name
    shutil.copytree(bundle, dst)
    stash = root / "removed"
    stash.mkdir()
    for entry in F.TRUTH_ENTRIES:
        if (dst / entry).exists():
            shutil.move(str(dst / entry), str(stash / entry))
    assert not (dst / "truth").exists() and (stash / "truth" / "truth_record.json").exists()
    return dst, stash


@pytest.fixture(scope="module")
def audited(bundle, stripped, tmp_path_factory):
    dst, stash = stripped
    out = tmp_path_factory.mktemp("play") / "out"
    res = F.run_child(dst, out, [bundle / e for e in F.TRUTH_ENTRIES] + [stash], "t")
    return res, out


def test_bundle_carries_the_observation_capture(bundle):
    check = verify_capture(bundle)
    assert check["problems"] == [] and (bundle / TAPE).exists()


def test_replay_without_truth_is_bit_identical(bundle, audited):
    res, out = audited
    assert res["playback"]["ran"] and res["playback"]["tape_divergence"] is None, res["playback"]
    assert res["playback"]["command_mismatches"] == 0 and res["playback"]["commands_checked"] > 0
    cmp = compare_outputs(bundle, out)
    assert cmp["equal"], cmp
    assert cmp["events"]["bytes_equal"] and cmp["sqlite_tables"]["belief_revisions"]
    assert cmp["mission_files"]["decisions.jsonl"] and cmp["commands"]["equal"]


def test_replay_opens_no_truth_path_and_imports_no_truth_module(audited):
    res, _ = audited
    opens = res["opens"]
    assert opens["truth_paths_opened"] == [], opens["truth_paths_opened"]
    assert res["forbidden"] == [], res["forbidden"]
    assert opens["bundle"]["distinct"] > 0  # the hook saw the captured inputs being read


def test_corrupting_truth_changes_nothing(bundle, audited, tmp_path):
    _, out = audited
    bad = tmp_path / bundle.name
    shutil.copytree(bundle, bad)
    assert F._corrupt_truth(bad)
    rep = replay_inference(bad, tmp_path / "play")
    assert rep["ran"] and rep["tape_divergence"] is None
    assert inference_fingerprint(tmp_path / "play") == inference_fingerprint(out)


def test_altering_one_captured_observation_changes_the_output(bundle, stripped, tmp_path):
    dst, _ = stripped
    alt = tmp_path / dst.name
    shutil.copytree(dst, alt)
    edit = F._rewrite_tape(alt, F._alter_depth)
    rep = replay_inference(alt, tmp_path / "play")
    assert rep["ran"], rep
    cmp = compare_outputs(bundle, tmp_path / "play")
    assert not cmp["equal"], edit
    assert (
        rep["tape_divergence"] is not None
        or rep["command_mismatches"] > 0
        or not cmp["events"]["bytes_equal"]
    )


def test_unsealed_tape_edit_is_refused(stripped, tmp_path):
    dst, _ = stripped
    alt = tmp_path / dst.name
    shutil.copytree(dst, alt)
    F._rewrite_tape(alt, F._alter_depth, reseal=False)
    rep = replay_inference(alt, tmp_path / "play")
    assert not rep["ran"] and any(TAPE in p for p in rep["problems"])


def test_bundle_without_capture_is_not_replayable_from_observations(stripped, tmp_path):
    dst, _ = stripped
    alt = tmp_path / dst.name
    shutil.copytree(dst, alt)
    shutil.rmtree(alt / "capture")
    rep = replay_inference(alt, tmp_path / "play")
    assert not rep["ran"] and rep["problems"]


def _unity_capture_bundle() -> Path | None:
    hits = sorted((REPO / "artifacts" / "runs_capture").glob("I1-UNITY-s5300059-*"))
    hits = [h for h in hits if (h / TAPE).exists() and (h / "bundle_manifest.json").exists()]
    return hits[-1] if hits else None


@pytest.mark.skipif(
    _unity_capture_bundle() is None, reason="no captured I1-UNITY bundle (needs a Unity player run)"
)
def test_unity_i1_bundle_replays_without_truth_or_player(tmp_path):
    src = _unity_capture_bundle()
    assert src is not None
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    stash = tmp_path / "removed"
    stash.mkdir()
    for entry in F.TRUTH_ENTRIES:
        if (dst / entry).exists():
            shutil.move(str(dst / entry), str(stash / entry))
    res = F.run_child(dst, tmp_path / "play", [src / e for e in F.TRUTH_ENTRIES] + [stash], "u")
    if res["playback"].get("same_code") is False and res["playback"].get("tape_divergence"):
        pytest.skip("conrad/ source changed since the Unity bundle was recorded (inconclusive, not a leak)")
    assert res["playback"]["ran"] and res["playback"]["tape_divergence"] is None, res["playback"]
    assert res["opens"]["truth_paths_opened"] == [] and res["forbidden"] == []
    assert compare_outputs(src, tmp_path / "play")["equal"]
