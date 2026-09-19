"""Process-wide cache of the (expensive) integrated runs shared by acceptance, leakage and replay tests."""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path
from typing import Any

from conrad.sim.mission.run import run_scenario

SMALL = "configs/sim/mission_test_small.yaml"
SMOKE = "configs/sim/golden/golden_smoke.yaml"


@functools.lru_cache(maxsize=1)
def runs_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="conrad-it-"))


@functools.cache
def cached_run(scenario_id: str, config: str = SMALL) -> dict[str, Any]:
    return run_scenario(scenario_id, config, runs_root=runs_root())


def flagship() -> dict[str, Any]:
    return cached_run("FLAGSHIP-I4", SMALL)


def smoke() -> dict[str, Any]:
    return cached_run("GOLDEN-SMOKE", SMOKE)


@functools.lru_cache(maxsize=1)
def smoke_replay() -> dict[str, Any]:
    from conrad.sim.mission.replay import replay_run

    return replay_run(Path(smoke()["run_dir"]), Path(tempfile.mkdtemp(prefix="conrad-r-")))


def short_session(
    scenario_id: str,
    duration_s: float = 24.0,
    runtime: dict[str, Any] | None = None,
    world: dict[str, Any] | None = None,
    steps: int | None = None,
) -> Any:
    """A prepared, stepped (not finished) short mission for white-box integration tests."""
    from conrad.settings import load_settings
    from conrad.sim.mission.run import prepare

    s = load_settings(SMALL)
    rt = {"duration_s": duration_s, "control_period_s": 0.1, **(runtime or {})}
    update: dict[str, Any] = {"sim": {"mission": {"world": world or {}, "runtime": rt}}}
    root = Path(tempfile.mkdtemp(prefix="conrad-s-"))
    session = prepare(scenario_id, s.model_copy(update=update), runs_root=root)
    if steps is None:
        session.run()
    else:
        for _ in range(steps):
            session.step()
    return session
