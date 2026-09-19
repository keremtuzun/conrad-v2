"""CC-10 / SS-02 / SS-04 on a real integrated mission bundle (GOLDEN-SMOKE)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from conrad.persistence.replay_store import ReplayIntegrityError
from conrad.sim.mission.replay import compare, replay_run
from tests.acceptance._runs import smoke, smoke_replay


@pytest.fixture(scope="module")
def replayed():
    return Path(smoke()["run_dir"]), smoke_replay()


def test_cc10_replay_reproduces_events_decisions_and_revisions(replayed):
    _, report = replayed
    assert report["equal"], report
    assert report["events"]["original"] > 100 and report["decisions"]["original"] > 3
    assert report["verified_files"] > 10


def test_ss02_same_config_gives_byte_equivalent_manifests(replayed):
    run_dir, report = replayed
    a = json.loads((run_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    b = json.loads((Path(report["replay_dir"]) / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert a == b


def test_ss04_missing_artifact_fails_closed_before_reexecution(tmp_path):
    src = Path(smoke()["run_dir"])
    broken = tmp_path / src.name
    shutil.copytree(src, broken)
    (broken / "mission" / "decisions.jsonl").unlink()
    with pytest.raises(ReplayIntegrityError) as err:
        replay_run(broken, tmp_path / "never")
    assert any("decisions.jsonl" in p for p in err.value.problems)
    assert not (tmp_path / "never").exists()


def test_corrupt_event_log_is_detected(tmp_path):
    src = Path(smoke()["run_dir"])
    broken = tmp_path / src.name
    shutil.copytree(src, broken)
    with (broken / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ReplayIntegrityError, match="corrupt"):
        replay_run(broken, tmp_path / "never")


def test_compare_detects_a_diverging_run(tmp_path):
    src = Path(smoke()["run_dir"])
    other = tmp_path / "other"
    shutil.copytree(src, other)
    lines = (other / "events.jsonl").read_text(encoding="utf-8").splitlines()
    (other / "events.jsonl").write_text("".join(ln + "\n" for ln in lines[:-1]), encoding="utf-8")
    report = compare(src, other)
    assert not report["equal"] and report["events"]["first_difference"] == len(lines) - 1
