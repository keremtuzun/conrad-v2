"""Build a real run bundle from the Phase-1 fake full system (shared by the console tests)."""

from __future__ import annotations

import json
from pathlib import Path

from conrad.persistence.db import make_engine, migrate
from conrad.persistence.replay_store import REQUIRED_REPLAY_KEYS, write_bundle_manifest
from conrad.persistence.repository import Repository
from conrad.schemas.decision import RejectedCandidate
from conrad.schemas.frames import WORLD, Pose
from tests.fixtures.fake_system import FakeFullSystem

TRUTH_MARKER = 0.918273645


def _inputs() -> dict:
    return {k: f"test-{k}" for k in REQUIRED_REPLAY_KEYS}


def make_bundle(run_dir: Path, seed: int = 21, extras: bool = True) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "run.sqlite"
    migrate(db_path)
    engine = make_engine(db_path)
    system = FakeFullSystem(Repository(engine), seed=seed, log_path=run_dir / "events.jsonl")
    res = system.run()
    engine.dispose()
    if extras:
        plan = res.plans[0]
        assert plan.primary_action is not None
        bad = plan.primary_action.model_copy(
            update={
                "action_id": system.ids.new(),
                "pose": Pose(frame_id=WORLD, position_m=(5.0, -2.5, -10.0)),
            }
        )
        plan = plan.model_copy(
            update={"rejected": (RejectedCandidate(action=bad, reason_codes=("NOT_VISIBLE_FROM_LANE",)),)}
        )
        (run_dir / "mission").mkdir()
        (run_dir / "mission" / "mcbr_plans.json").write_text(
            json.dumps([plan.model_dump(mode="json")]), encoding="utf-8"
        )
        (run_dir / "mission" / "transmissions.json").write_text(
            json.dumps([t.model_dump(mode="json") for t in res.transmissions]), encoding="utf-8"
        )
        (run_dir / "truth").mkdir()
        (run_dir / "truth" / "truth_record.json").write_text(
            json.dumps(
                {
                    "hidden_severity": TRUTH_MARKER,
                    "pose_track": [
                        {"time_ns": 0, "position_m": [0.0, -3.0, -10.0]},
                        {"time_ns": 1, "position_m": list(res.final_position)},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "reports").mkdir()
        (run_dir / "reports" / "metrics.json").write_text(
            json.dumps({"commands_accepted": len(res.commands), "decisions": len(res.decisions)}),
            encoding="utf-8",
        )
        (run_dir / "config.resolved.yaml").write_text("database: run.sqlite\nseed: 21\n", encoding="utf-8")
    write_bundle_manifest(run_dir, run_dir.name, _inputs(), [])
    return run_dir
