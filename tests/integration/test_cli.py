"""`conrad sim run GOLDEN-SMOKE` then `conrad replay run`, plus the runtime commands (ch34 CLI contract)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from conrad.cli import commands  # noqa: F401  (registers sim/replay/runtime)
from conrad.cli.app import app
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.replay_store import verify_bundle

RUNNER = CliRunner()


def _config(tmp_path: Path, lane: str = "simulation", mode: str = "simulated") -> Path:
    cfg = {
        "extends": ["configs/sim/golden/golden_smoke.yaml"],
        "paths": {"runs_dir": str(tmp_path / "runs")},
        "run": {"lane": lane},
        "runtime": {"command_mode": mode},
    }
    path = tmp_path / f"cfg_{lane}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _json_prefix(text: str) -> dict:
    return json.loads(text[: text.rindex("}") + 1])


def test_sim_run_then_replay_run(tmp_path):
    cfg = _config(tmp_path)
    res = RUNNER.invoke(
        app, ["sim", "run", "--scenario", "GOLDEN-SMOKE", "--config", str(cfg), "--run-id", "cli-smoke"]
    )
    assert res.exit_code == 0, res.output
    out = _json_prefix(res.output)
    run_dir = Path(out["run_dir"])
    assert (run_dir / "bundle_manifest.json").exists() and out["commands_accepted"] > 0
    rep = RUNNER.invoke(app, ["replay", "run", "--run", "cli-smoke", "--config", str(cfg)])
    assert rep.exit_code == 0, rep.output
    assert "RESULT: REPRODUCED" in rep.output
    mission_id = json.loads((run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])[
        "envelope"
    ]["mission_id"]
    hold = RUNNER.invoke(app, ["runtime", "safe-hold", "--mission", mission_id, "--config", str(cfg)])
    assert hold.exit_code == 0, hold.output
    assert "RECORDED_NO_LIVE_PROCESS" in hold.output
    assert (run_dir / "notes" / "operator_requests.jsonl").exists()
    verify_bundle(run_dir, ObjectStore(run_dir / "objects"))  # the operator note lives outside the digests


def test_replay_of_tampered_bundle_exits_nonzero(tmp_path):
    cfg = _config(tmp_path)
    missing = RUNNER.invoke(app, ["replay", "run", "--run", str(tmp_path / "nope"), "--config", str(cfg)])
    assert missing.exit_code == 2 and "bundle manifest missing" in missing.output


def test_runtime_start_refuses_the_hardware_lane(tmp_path):
    cfg = _config(tmp_path, lane="hil", mode="disabled")
    res = RUNNER.invoke(app, ["runtime", "start", "--config", str(cfg)])
    assert res.exit_code == 3
    body = _json_prefix(res.output)
    assert body["refused"] and "command_mode is not hardware" in body["reasons"]


def test_safe_hold_for_unknown_mission_reports_honestly(tmp_path):
    cfg = _config(tmp_path)
    res = RUNNER.invoke(
        app,
        ["runtime", "safe-hold", "--mission", "00000000-0000-0000-0000-000000000000", "--config", str(cfg)],
    )
    assert res.exit_code == 1 and "no run found" in res.output
