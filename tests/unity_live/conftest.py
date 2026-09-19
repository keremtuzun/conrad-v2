"""Gate U0 live tests: launch the BUILT Unity player and drive it through UnityRobotHardware over TCP.

Skipped (with an explicit reason) only when the player binary is absent. Build it with
``Unity.exe -batchmode -nographics -quit -projectPath unity/ConradUnityV2 -executeMethod
Conrad.UnityV2.Editor.BuildScript.BuildWindows64Player``.

Each test records the numbers it measured with ``u0_record``; at session end the per-criterion
outcome (from the real pytest reports) and the numbers are written to ``artifacts/gates/U0/``.
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import shutil
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
_STUB = str(REPO / "tests" / "hardware_stub")
if _STUB not in sys.path:
    sys.path.insert(0, _STUB)

from conrad.sim.unity.player import find_player  # noqa: E402

EVIDENCE_DIR = REPO / "artifacts" / "gates" / "U0"
LIVE_TEST_HARD_TIMEOUT_S = 600.0

_RECORDS: dict[str, dict[str, Any]] = {}
_OUTCOMES: dict[str, str] = {}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unity_live: needs the built Unity V2 player (gate U0)")
    config.addinivalue_line(
        "markers", "u0(criterion): the gate U0 criterion a live test provides evidence for"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    player = find_player()
    for item in items:
        if "unity_live" in str(item.fspath):
            item.add_marker(pytest.mark.unity_live)
            if player is None:
                item.add_marker(
                    pytest.mark.skip(
                        reason="Unity player binary absent: build unity/ConradUnityV2/Builds/Win64/ConradSim.exe "
                        "with BuildScript.BuildWindows64Player or set CONRAD_UNITY_PLAYER"
                    )
                )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[None]:
    outcome = yield
    report = outcome.get_result()  # type: ignore[attr-defined]
    marker = item.get_closest_marker("u0")
    if marker is None:
        return
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _OUTCOMES[item.nodeid] = report.outcome
    _RECORDS.setdefault(item.nodeid, {"criterion": marker.args[0], "test": item.name, "measured": {}})


@pytest.fixture(autouse=True)
def _hard_timeout() -> Iterator[None]:
    faulthandler.dump_traceback_later(LIVE_TEST_HARD_TIMEOUT_S, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def u0_record(request: pytest.FixtureRequest) -> Callable[..., None]:
    marker = request.node.get_closest_marker("u0")
    assert marker is not None, "live U0 tests must declare @pytest.mark.u0(<criterion>)"
    rec = _RECORDS.setdefault(
        request.node.nodeid, {"criterion": marker.args[0], "test": request.node.name, "measured": {}}
    )

    def record(**values: Any) -> None:
        rec["measured"].update(values)

    return record


@pytest.fixture(scope="session")
def live_dir() -> Path:
    d = REPO / "artifacts" / "unity" / "u0_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _git_commit() -> str | None:
    """HEAD read from .git files (no git command is executed)."""
    git = REPO / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        loose = git / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip()
        packed = git / "packed-refs"
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    except OSError:
        return None
    return None


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _RECORDS:
        return
    player = find_player()
    unity_root = REPO / "unity" / "ConradUnityV2"
    build_info_path = None if player is None else player.parent / "conrad_build_info.json"
    build_info = (
        json.loads(build_info_path.read_text(encoding="utf-8"))
        if build_info_path and build_info_path.is_file()
        else None
    )
    build_log = unity_root / "Logs" / "u0_build.log"
    log_text = build_log.read_text(encoding="utf-8", errors="replace") if build_log.is_file() else ""
    criteria: dict[str, dict[str, Any]] = {}
    for nodeid, rec in sorted(_RECORDS.items()):
        outcome = _OUTCOMES.get(nodeid, "not-run")
        c = criteria.setdefault(rec["criterion"], {"status": "PASS", "tests": []})
        c["tests"].append({"test": rec["test"], "outcome": outcome, "measured": rec["measured"]})
        if outcome != "passed":
            c["status"] = "FAIL" if outcome == "failed" else "NOT_RUN"
    summary = {
        "gate": "U0",
        "simulation_validity_level": "L1_APPROXIMATE_PHYSICS",
        "parameter_provenance": "every physical parameter SYNTHETIC_ONLY (configs/robot/sim_reference.yaml + labelled variants)",
        "unity_editor_version": None if build_info is None else build_info.get("unity_version"),
        "player": None
        if player is None
        else str(player.relative_to(REPO))
        if player.is_relative_to(REPO)
        else str(player),
        "player_exe_sha256": None if player is None else _sha256(player),
        "build_info": build_info,
        "build_log": "artifacts/gates/U0/unity_build.log (copy of unity/ConradUnityV2/Logs/u0_build.log)",
        "build_log_sha256": _sha256(build_log),
        "build_log_compiler_errors": log_text.count("error CS"),
        "build_log_compiler_warnings": log_text.count("warning CS"),
        "build_log_result_line": next((ln for ln in log_text.splitlines() if "Build Finished" in ln), None),
        "project_version_txt": (unity_root / "ProjectSettings" / "ProjectVersion.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "git_commit": _git_commit(),
        "pytest_exit_status": int(exitstatus),
        "criteria": criteria,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    if build_log.is_file():
        shutil.copyfile(build_log, EVIDENCE_DIR / "unity_build.log")
    (EVIDENCE_DIR / "u0_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
