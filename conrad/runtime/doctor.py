"""`conrad doctor`: material runtime prerequisites, human and machine readable (ch34).

Exit is nonzero on a missing contract, incompatible checkpoint, invalid migration, unavailable
required adapter, insufficient disk, invalid RobotConfig or forbidden command_mode. Secrets are
reported by NAME only.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from conrad.persistence import db
from conrad.persistence.object_store import ObjectStore
from conrad.robotics.hardware.config import load_robot_config, validate_for_lane
from conrad.schemas.base import ARCHITECTURE_ID, SCHEMA_VERSION, STACK_ID, ConradModel
from conrad.schemas.frames import FramedPoint, Transform, quat_from_euler, transform_point
from conrad.settings import REPO_ROOT, CommandMode, ConradSettings, ExecutionLane, load_settings

ADAPTERS = {
    "sim_kernel": "conrad.sim.kernel",
    "unity": "conrad.adapters.unity",
    "physical": None,  # supplied by the hardware owner (EXT-HW-01); never importable by default
}


class Check(ConradModel):
    name: str
    status: str  # OK | WARN | FAIL
    detail: str


class DoctorReport(ConradModel):
    architecture_id: str = ARCHITECTURE_ID
    stack_id: str = STACK_ID
    schema_version: str = SCHEMA_VERSION
    config: str
    lane: str
    command_mode: str
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(c.status != "FAIL" for c in self.checks)


def _run(name: str, fn: Callable[[], str], out: list[Check]) -> None:
    try:
        detail = fn()
        status = "WARN" if detail.startswith("WARN:") else "OK"
        out.append(Check(name=name, status=status, detail=detail.removeprefix("WARN:").strip()))
    except Exception as exc:
        out.append(Check(name=name, status="FAIL", detail=f"{type(exc).__name__}: {exc}"))


def run_doctor(config_path: str, settings: ConradSettings | None = None) -> DoctorReport:
    checks: list[Check] = []
    loaded: dict[str, Any] = {}

    def config() -> str:
        loaded["s"] = settings or load_settings(config_path)
        s: ConradSettings = loaded["s"]
        return f"valid; digest {s.config_digest()[:12]}"

    _run("config", config, checks)
    if "s" not in loaded:
        return DoctorReport(config=config_path, lane="unknown", command_mode="unknown", checks=tuple(checks))
    s: ConradSettings = loaded["s"]
    lane, rt = s.run.lane, s.runtime

    def python() -> str:
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError(f"Python 3.11.x required, found {sys.version.split()[0]}")
        return sys.version.split()[0]

    def lock() -> str:
        lockfile = REPO_ROOT / "uv.lock"
        if not lockfile.exists():
            raise RuntimeError("uv.lock missing")
        for mod in (
            "pydantic",
            "numpy",
            "scipy",
            "torch",
            "cv2",
            "yaml",
            "sqlalchemy",
            "alembic",
            "typer",
            "structlog",
            "zmq",
        ):
            importlib.import_module(mod)
        return "uv.lock sha256 " + hashlib.sha256(lockfile.read_bytes()).hexdigest()[:12]

    def identity() -> str:
        if s.run.architecture_id != ARCHITECTURE_ID or s.run.stack_id != STACK_ID:
            raise RuntimeError("architecture/stack identity mismatch")
        return f"{ARCHITECTURE_ID} / {STACK_ID}"

    def directories() -> str:
        for rel in (s.paths.artifacts_dir, s.paths.runs_dir, s.paths.object_store):
            s.resolve(rel).mkdir(parents=True, exist_ok=True)
        return "artifact directories present"

    def disk() -> str:
        free = shutil.disk_usage(s.resolve(s.paths.artifacts_dir)).free
        if free < rt.min_free_disk_bytes:
            raise RuntimeError(f"free disk {free} < required {rt.min_free_disk_bytes}")
        return f"{free // (1024 * 1024)} MiB free"

    def object_store() -> str:
        ObjectStore(s.resolve(s.paths.object_store)).self_test()
        return "write/read hash verified"

    def migrations() -> str:
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / "probe.sqlite"
            head = db.migrate(scratch)
            engine = db.make_engine(scratch)
            db.check_migration_state(engine)
            db.integrity_check(engine)
            engine.dispose()
        target = s.resolve(s.paths.database)
        if not target.exists():
            if lane in (ExecutionLane.HIL, ExecutionLane.PHYSICAL):
                raise RuntimeError(f"database {target} does not exist; run `conrad db migrate`")
            return f"WARN: migration head {head} applies cleanly; runtime database not created yet (run `conrad db migrate`)"
        engine = db.make_engine(target)
        try:
            db.check_migration_state(engine)
            db.integrity_check(engine)
        finally:
            engine.dispose()
        return f"database at head {head}, integrity ok"

    def frames() -> str:
        t = Transform(
            parent_frame="A",
            child_frame="B",
            translation_m=(1, 2, 3),
            rotation_wxyz=quat_from_euler(0.1, 0.2, 0.3),
        )
        p = FramedPoint(frame_id="B", xyz_m=(0.5, -0.25, 2.0))
        back = transform_point(transform_point(p, t), t.inverse())
        if max(abs(a - b) for a, b in zip(back.xyz_m, p.xyz_m, strict=True)) > 1e-9:
            raise RuntimeError("frame contract fixture failed")
        return "T_A_from_B round trip ok"

    def robot_config() -> str:
        cfg = load_robot_config(rt.robot_config)
        problems = validate_for_lane(cfg, lane)
        if problems:
            raise RuntimeError(f"{len(problems)} problem(s), first: {problems[0]}")
        return f"{cfg.config_name} digest {cfg.content_digest()[:12]} valid for lane {lane.value}"

    def command_mode() -> str:
        if rt.command_mode is CommandMode.HARDWARE:
            missing = []
            if not rt.hardware_enable:
                missing.append("local hardware enable")
            if not rt.hil_evidence_ref:
                missing.append("HIL evidence reference")
            if lane is not ExecutionLane.PHYSICAL and lane is not ExecutionLane.HIL:
                missing.append("hil/physical lane")
            if missing:
                raise RuntimeError("command_mode hardware forbidden: missing " + ", ".join(missing))
        return f"command_mode {rt.command_mode.value} permitted in lane {lane.value}"

    def adapters() -> str:
        for name in rt.required_adapters:
            if name not in ADAPTERS:
                raise RuntimeError(f"unknown adapter {name!r}")
            module = ADAPTERS[name]
            if module is None:
                raise RuntimeError(f"adapter {name!r} is not installed (external hardware driver, EXT-HW-01)")
            importlib.import_module(module)
        return "available: " + ", ".join(rt.required_adapters)

    def checkpoints() -> str:
        listed = s.model.get("checkpoints", [])
        if not listed:
            return "no checkpoints configured (analytic operators in use)"
        ckpt = importlib.import_module("conrad.training.checkpoint")
        for path in listed:
            ckpt.verify_checkpoint_metadata(s.resolve(path))
        return f"{len(listed)} checkpoint(s) compatible"

    def manifests() -> str:
        if not s.data.manifest_ids:
            return "no dataset manifests required by this config"
        manifest = importlib.import_module("conrad.data.manifest")
        for mid in s.data.manifest_ids:
            problems = manifest.verify_manifest_by_id(mid)
            if problems:
                raise RuntimeError(f"manifest {mid}: {problems[0]}")
        return f"{len(s.data.manifest_ids)} manifest(s) verified"

    def secrets() -> str:
        missing = [name for name in rt.required_secrets if not os.environ.get(name)]
        if missing:
            raise RuntimeError("missing secret names: " + ", ".join(missing))
        return f"{len(rt.required_secrets)} required secret name(s) present"

    for name, fn in (
        ("python", python),
        ("dependencies", lock),
        ("identity", identity),
        ("directories", directories),
        ("disk", disk),
        ("object_store", object_store),
        ("migrations", migrations),
        ("frame_contract", frames),
        ("robot_config", robot_config),
        ("command_mode", command_mode),
        ("adapters", adapters),
        ("checkpoints", checkpoints),
        ("dataset_manifests", manifests),
        ("secrets", secrets),
    ):
        _run(name, fn, checks)
    return DoctorReport(
        config=config_path, lane=lane.value, command_mode=rt.command_mode.value, checks=tuple(checks)
    )
