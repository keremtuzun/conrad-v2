"""Immutable run directory ``artifacts/runs/<run_id>/`` (ch34 Configuration / run directory).

Layout: config.resolved.yaml, environment.json, git.json, manifests.json, metrics.jsonl,
events.jsonl, checkpoints/, reports/, figures/, replay/, logs/, notes/.
After a terminal state the directory is sealed; only ``notes/`` stays append-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import yaml

from conrad.schemas.base import ARCHITECTURE_ID, STACK_ID, ConradModel
from conrad.settings import REPO_ROOT, redact
from conrad.training.determinism import determinism_flags, inspect_compute

SUBDIRS = ("checkpoints", "reports", "figures", "replay", "logs", "notes")
STATE_FILE = "run_state.json"


class RunPurpose(str, Enum):
    DEVELOPMENT = "development"
    BENCHMARK = "benchmark"
    ACCEPTANCE = "acceptance"
    PHYSICAL = "physical"


CLEAN_GIT_REQUIRED = frozenset({RunPurpose.BENCHMARK, RunPurpose.ACCEPTANCE, RunPurpose.PHYSICAL})


class Comparability(str, Enum):
    COMPARABLE = "COMPARABLE"
    NONCOMPARABLE = "NONCOMPARABLE"


class TerminalStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class GitState(ConradModel):
    commit: str | None
    dirty: bool
    dirty_diff: str = ""
    available: bool = True


class DirtyGitError(RuntimeError):
    """A benchmark / acceptance / physical run was started from dirty or unknown tracked source."""


class RunSealedError(RuntimeError):
    """A write was attempted on a run directory that already reached a terminal state."""


class RunDirectoryError(RuntimeError):
    pass


GitProbe = Callable[[], GitState]


def probe_git(repo_root: Path = REPO_ROOT) -> GitState:
    """Read-only git inspection. Unknown state is reported as unavailable + dirty (fail closed)."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True, timeout=30
        ).stdout

    try:
        commit = run("rev-parse", "HEAD").strip()
        dirty = bool(run("status", "--porcelain", "--untracked-files=no").strip())
        diff = run("diff", "HEAD") if dirty else ""
    except (OSError, subprocess.SubprocessError):
        return GitState(commit=None, dirty=True, dirty_diff="", available=False)
    return GitState(commit=commit, dirty=dirty, dirty_diff=diff)


def lockfile_digest(repo_root: Path = REPO_ROOT) -> str | None:
    lock = repo_root / "uv.lock"
    return hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None


def environment_snapshot(
    *,
    device: str = "cpu",
    device_fallback: bool = False,
    precision: str = "float32",
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    compute = inspect_compute()
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "lockfile": "uv.lock",
        "lockfile_digest": lockfile_digest(repo_root),
        "torch_version": str(torch.__version__),
        "cuda_available": compute.cuda_available,
        "cuda_visible_devices": compute.cuda_visible_devices,
        "cuda_runtime": torch.version.cuda,
        "devices": list(compute.cuda_devices) or ["cpu"],
        "device": device,
        "device_fallback": device_fallback,
        "effective_precision": precision,
        "determinism_flags": determinism_flags(),
        "architecture_id": ARCHITECTURE_ID,
        "stack_id": STACK_ID,
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunDirectory:
    def __init__(self, path: Path, *, comparability: Comparability, sealed: bool) -> None:
        self.path = path
        self.comparability = comparability
        self._sealed = sealed

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        runs_root: str | Path,
        run_id: str,
        *,
        resolved_config: Mapping[str, Any],
        manifests: Mapping[str, Any],
        purpose: RunPurpose,
        clock_ns: Callable[[], int],
        git_probe: GitProbe = probe_git,
        environment: Mapping[str, Any] | None = None,
    ) -> RunDirectory:
        git = git_probe()
        if purpose in CLEAN_GIT_REQUIRED and (git.dirty or not git.available or git.commit is None):
            raise DirtyGitError(
                f"{purpose.value} runs reject dirty or unknown tracked source "
                f"(commit={git.commit}, dirty={git.dirty}, git_available={git.available})"
            )
        path = Path(runs_root) / run_id
        if path.exists():
            raise RunDirectoryError(f"run directory already exists and is immutable: {path}")
        path.mkdir(parents=True)
        for sub in SUBDIRS:
            (path / sub).mkdir()
        comparability = Comparability.NONCOMPARABLE if git.dirty else Comparability.COMPARABLE
        (path / "config.resolved.yaml").write_text(
            yaml.safe_dump(
                redact(json.loads(json.dumps(dict(resolved_config), default=str))), sort_keys=True
            ),
            encoding="utf-8",
        )
        _write_json(
            path / "environment.json",
            dict(environment) if environment is not None else environment_snapshot(),
        )
        git_record: dict[str, Any] = {
            "commit": git.commit,
            "dirty": git.dirty,
            "git_available": git.available,
            "purpose": purpose.value,
            "status": comparability.value,
            "dirty_diff": None,
        }
        if git.dirty:
            (path / "dirty_diff.patch").write_text(git.dirty_diff, encoding="utf-8")
            git_record["dirty_diff"] = "dirty_diff.patch"
        _write_json(path / "git.json", git_record)
        _write_json(path / "manifests.json", dict(manifests))
        (path / "metrics.jsonl").touch()
        (path / "events.jsonl").touch()
        run = cls(path, comparability=comparability, sealed=False)
        run.log_event(
            "RUN_CREATED", clock_ns(), {"purpose": purpose.value, "comparability": comparability.value}
        )
        return run

    @classmethod
    def open(cls, path: str | Path) -> RunDirectory:
        p = Path(path)
        if not (p / "git.json").is_file():
            raise RunDirectoryError(f"{p} is not a run directory")
        git = json.loads((p / "git.json").read_text(encoding="utf-8"))
        return cls(p, comparability=Comparability(git["status"]), sealed=(p / STATE_FILE).is_file())

    # ---- writes -------------------------------------------------------------------------------
    @property
    def sealed(self) -> bool:
        return self._sealed or (self.path / STATE_FILE).is_file()

    def _require_open(self) -> None:
        if self.sealed:
            raise RunSealedError(f"run {self.path.name} is sealed; only notes/ accepts appends")

    def _append_jsonl(self, name: str, record: Mapping[str, Any]) -> None:
        self._require_open()
        with (self.path / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True, default=str) + "\n")

    def log_metrics(self, record: Mapping[str, Any]) -> None:
        self._append_jsonl("metrics.jsonl", record)

    def log_event(self, event_type: str, time_ns: int, detail: Mapping[str, Any] | None = None) -> None:
        self._append_jsonl(
            "events.jsonl", {"event_type": event_type, "time_ns": time_ns, "detail": dict(detail or {})}
        )

    def subdir(self, name: str) -> Path:
        if name not in SUBDIRS:
            raise RunDirectoryError(f"unknown run subdirectory {name!r}")
        if name != "notes":
            self._require_open()
        return self.path / name

    def write_artifact(self, subdir: str, name: str, data: bytes | str) -> Path:
        if subdir == "notes":
            raise RunDirectoryError("notes/ is append-only; use append_note()")
        target = self.subdir(subdir) / name
        if target.exists():
            raise RunDirectoryError(f"artifact already exists and is immutable: {target}")
        if isinstance(data, str):
            target.write_text(data, encoding="utf-8")
        else:
            target.write_bytes(data)
        return target

    def append_note(self, time_ns: int, author: str, text: str) -> Path:
        """Allowed before and after sealing. Notes are never edited, only appended."""
        target = self.path / "notes" / "notes.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"time_ns": time_ns, "author": author, "text": text}, sort_keys=True) + "\n"
            )
        return target

    # ---- sealing ------------------------------------------------------------------------------
    def _sealed_files(self) -> list[Path]:
        return sorted(
            p
            for p in self.path.rglob("*")
            if p.is_file() and p.name != STATE_FILE and p.relative_to(self.path).parts[0] != "notes"
        )

    def seal(self, status: TerminalStatus, time_ns: int) -> dict[str, Any]:
        self.log_event("RUN_TERMINAL", time_ns, {"status": status.value})
        files = {p.relative_to(self.path).as_posix(): _sha256(p) for p in self._sealed_files()}
        state = {
            "status": status.value,
            "comparability": self.comparability.value,
            "finished_time_ns": time_ns,
            "files": files,
        }
        _write_json(self.path / STATE_FILE, state)
        for p in [*self._sealed_files(), self.path / STATE_FILE]:
            os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        self._sealed = True
        return state

    def verify_seal(self) -> list[str]:
        """Problems found when re-hashing a sealed run; empty means untouched."""
        state_path = self.path / STATE_FILE
        if not state_path.is_file():
            return ["run is not sealed"]
        recorded: dict[str, str] = json.loads(state_path.read_text(encoding="utf-8"))["files"]
        current = {p.relative_to(self.path).as_posix(): p for p in self._sealed_files()}
        problems = [f"missing after seal: {name}" for name in sorted(set(recorded) - set(current))]
        problems += [f"added after seal: {name}" for name in sorted(set(current) - set(recorded))]
        problems += [
            f"modified after seal: {name}"
            for name in sorted(set(recorded) & set(current))
            if _sha256(current[name]) != recorded[name]
        ]
        return problems
