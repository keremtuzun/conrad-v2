from __future__ import annotations

import json
import os
import stat

import pytest

from conrad.training.run_dir import (
    Comparability,
    DirtyGitError,
    GitState,
    RunDirectory,
    RunDirectoryError,
    RunPurpose,
    RunSealedError,
    TerminalStatus,
    environment_snapshot,
)

CLEAN = GitState(commit="a" * 40, dirty=False)
DIRTY = GitState(commit="a" * 40, dirty=True, dirty_diff="diff --git a/x b/x\n")
UNKNOWN = GitState(commit=None, dirty=True, available=False)


def create(tmp_path, git: GitState, purpose=RunPurpose.DEVELOPMENT, run_id="r1") -> RunDirectory:
    return RunDirectory.create(
        tmp_path,
        run_id,
        resolved_config={"run": {"seed": 1}, "api_token": "secret"},
        manifests={"m": "d"},
        purpose=purpose,
        clock_ns=lambda: 5,
        git_probe=lambda: git,
        environment={"os": "test"},
    )


def test_layout_and_redaction(tmp_path):
    run = create(tmp_path, CLEAN)
    for name in (
        "config.resolved.yaml",
        "environment.json",
        "git.json",
        "manifests.json",
        "metrics.jsonl",
        "events.jsonl",
        "checkpoints",
        "reports",
        "figures",
        "replay",
        "logs",
        "notes",
    ):
        assert (run.path / name).exists(), name
    assert "secret" not in (run.path / "config.resolved.yaml").read_text()
    assert run.comparability is Comparability.COMPARABLE
    with pytest.raises(RunDirectoryError):
        create(tmp_path, CLEAN)


def test_dirty_development_run_is_noncomparable_with_patch(tmp_path):
    run = create(tmp_path, DIRTY)
    git = json.loads((run.path / "git.json").read_text())
    assert git["status"] == "NONCOMPARABLE" and git["dirty"] is True
    assert (run.path / "dirty_diff.patch").read_text().startswith("diff --git")


@pytest.mark.parametrize("purpose", [RunPurpose.BENCHMARK, RunPurpose.ACCEPTANCE, RunPurpose.PHYSICAL])
@pytest.mark.parametrize("git", [DIRTY, UNKNOWN])
def test_dirty_or_unknown_source_rejected_for_gated_runs(tmp_path, purpose, git):
    with pytest.raises(DirtyGitError):
        create(tmp_path, git, purpose)
    assert not (tmp_path / "r1").exists()
    create(tmp_path, CLEAN, purpose, run_id="ok")


def test_seal_blocks_writes_but_not_notes(tmp_path):
    run = create(tmp_path, CLEAN)
    run.log_metrics({"step": 1, "val_loss": 0.3})
    run.write_artifact("reports", "r.txt", "hello")
    state = run.seal(TerminalStatus.COMPLETED, 9)
    assert "reports/r.txt" in state["files"]
    assert run.verify_seal() == []
    with pytest.raises(RunSealedError):
        run.log_metrics({"step": 2})
    with pytest.raises(RunSealedError):
        RunDirectory.open(run.path).write_artifact("reports", "late.txt", "x")
    run.append_note(10, "kerem", "post-hoc observation")
    assert run.verify_seal() == []
    target = run.path / "reports" / "r.txt"
    os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
    target.write_text("changed")
    assert run.verify_seal() == ["modified after seal: reports/r.txt"]


def test_environment_snapshot_records_lockfile_and_determinism():
    env = environment_snapshot(device="cpu", device_fallback=True)
    assert env["lockfile_digest"] and len(env["lockfile_digest"]) == 64
    assert env["device_fallback"] is True and "deterministic_algorithms" in env["determinism_flags"]
