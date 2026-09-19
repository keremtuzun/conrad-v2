"""Observation capture and truth-free inference replay (CC-10 falsification harness).

Recording (truth side, used by ``run.prepare`` and ``unity_run.prepare_unity``): the deployment runtime sees the
robot ONLY through ``RobotHardwareInterface``. ``RecordingHardware`` wraps the world's hardware object and writes
every call the runtime makes, with its arguments and its return value, to ``capture/rhi_tape.jsonl.gz``. The driver
writes tick markers and the events it injects into the shared event log. The mission context and the RobotConfig the
runtime loaded are stored next to the tape. Nothing on the tape is read from the truth world by the runtime: it is
exactly what crossed the hardware boundary.

Playback (``replay_inference``): rebuild the unchanged ``MissionRuntime`` from the bundle's stored configuration,
the captured mission context and ``PlaybackHardware``, which serves the tape in order and refuses any call the
tape does not hold (``TapeDivergence``). No Twin, no world, no simulator and no Unity player are constructed, so the
inference and decision planes are re-run from captured observations only. This module must not import
``conrad.twins``, ``conrad.sim.kernel``, ``conrad.sim.mission.world`` or anything else that holds truth; the
falsification script checks that in a fresh process.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib
import io
import json
import pickle
import shutil
import sqlite3
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import IO, Any

from pydantic import BaseModel

from conrad.orchestration.artifacts import write_mission_artifacts
from conrad.orchestration.mission import MissionRuntime
from conrad.orchestration.mission_config import runtime_config
from conrad.orchestration.mission_context import MissionContext
from conrad.persistence.db import make_engine, migrate
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.repository import Repository
from conrad.robotics.hardware.interface import RobotHardwareInterface
from conrad.runtime.event_log import EventLog, RunContext, event_signature, read_events
from conrad.schemas.events import EventType
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Observation
from conrad.schemas.robot import (
    AllocatedCommand,
    BatteryState,
    CommandAck,
    DepthSample,
    ImuSample,
    RobotCapabilities,
    RobotConfig,
    SystemHealth,
    ThrusterState,
)
from conrad.settings import ConradSettings

CAPTURE_DIR = "capture"
TAPE = f"{CAPTURE_DIR}/rhi_tape.jsonl.gz"
CONTEXT = f"{CAPTURE_DIR}/mission_context.json"
ROBOT = f"{CAPTURE_DIR}/robot_config.json"
META = f"{CAPTURE_DIR}/capture_meta.json"
TAPE_VERSION = "rhi-tape-1"
ATTRIBUTES = ("adapter_name", "is_physical")


class TapeDivergence(RuntimeError):
    """The runtime asked the hardware for something the tape does not hold at this point."""


# ------------------------------------------------------------------------------------------ value codec
def encode(v: Any) -> Any:
    """Tagged JSON encoding that decodes to an equal value (pydantic models, tuples, enums, containers)."""
    if v is None or isinstance(v, bool | int | float | str):
        return v
    if isinstance(v, BaseModel):
        cls = type(v)
        return {"__m__": f"{cls.__module__}:{cls.__qualname__}", "v": v.model_dump(mode="json")}
    if isinstance(v, Enum):
        ecls = type(v)
        return {"__e__": f"{ecls.__module__}:{ecls.__qualname__}", "v": v.value}
    if isinstance(v, tuple):
        return {"__t__": [encode(x) for x in v]}
    if isinstance(v, list):
        return [encode(x) for x in v]
    if isinstance(v, dict):
        return {"__d__": [[encode(k), encode(x)] for k, x in v.items()]}
    raise TypeError(f"cannot capture a value of type {type(v).__name__}")


def _exact(value: Any, encoded: Any) -> bool:
    """True when ``decode(encoded)`` reproduces ``value`` bit for bit (checked on the JSON form)."""
    try:
        return bool(encode(decode(encoded)) == encoded and _same(decode(encoded), value))
    except (TypeError, ValueError):
        return False


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, BaseModel) and isinstance(b, BaseModel):
        return type(a) is type(b) and a.model_dump(mode="json") == b.model_dump(mode="json")
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b) and type(a) is type(b)


def _resolve(path: str) -> Any:
    module, _, qual = path.partition(":")
    obj: Any = importlib.import_module(module)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def decode(v: Any) -> Any:
    if isinstance(v, list):
        return [decode(x) for x in v]
    if not isinstance(v, dict):
        return v
    if "__m__" in v:
        return _resolve(v["__m__"]).model_validate(v["v"])
    if "__e__" in v:
        return _resolve(v["__e__"])(v["v"])
    if "__t__" in v:
        return tuple(decode(x) for x in v["__t__"])
    if "__d__" in v:
        return {decode(k): decode(x) for k, x in v["__d__"]}
    raise ValueError(f"unknown tape encoding keys {sorted(v)}")


def _line(rec: dict[str, Any]) -> str:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), allow_nan=True)


# ------------------------------------------------------------------------------------------ recording
def code_digest() -> str:
    """sha256 over every ``conrad/**/*.py`` (path + bytes): tells code drift apart from an input dependence."""
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


class Tape:
    """Append-only gzip JSONL writer; deterministic bytes (gzip mtime 0, no file name)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._raw: IO[bytes] = path.open("wb")
        self._gz = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self._text = io.TextIOWrapper(self._gz, encoding="utf-8", newline="\n")
        self.records = 0
        self.closed = False

    def write(self, rec: dict[str, Any]) -> None:
        self._text.write(_line(rec) + "\n")
        self.records += 1

    def marker(self, kind: str, **fields: Any) -> None:
        self.write({"k": kind, **fields})

    def close(self) -> None:
        if not self.closed:
            self._text.close()
            self._raw.close()
            self.closed = True


class RecordingHardware(RobotHardwareInterface):
    """Transparent proxy: forwards to ``inner`` and records every call the runtime makes."""

    def __init__(self, inner: RobotHardwareInterface, tape: Tape) -> None:
        self._inner = inner
        self._tape = tape
        self.calls_after_close = 0
        self.adapter_name = inner.adapter_name
        self.is_physical = inner.is_physical
        extras = sorted(n for n in ("get_payload_observations",) if callable(getattr(inner, n, None)))
        self._extras = extras
        tape.marker(
            "header",
            version=TAPE_VERSION,
            inner=f"{type(inner).__module__}:{type(inner).__qualname__}",
            attributes={a: getattr(inner, a) for a in ATTRIBUTES},
            extra_methods=extras,
        )

    def _call(self, name: str, *args: Any) -> Any:
        result = getattr(self._inner, name)(*args)
        if self._tape.closed:
            # After ``finish`` the capture is sealed; harnesses may still query the runtime (scoring, reports).
            # Those calls are outside the replayed interval: pass them through, count them, never crash.
            self.calls_after_close += 1
            return result
        rec: dict[str, Any] = {"k": "call", "n": name, "a": [encode(a) for a in args], "r": encode(result)}
        if not _exact(result, rec["r"]):
            # A validator that is not idempotent at the last ulp (e.g. Pose re-normalizes its quaternion) would
            # make the JSON decode differ from what the runtime saw; keep the exact object too.
            rec["p"] = base64.b64encode(pickle.dumps(result, protocol=4)).decode("ascii")
        self._tape.write(rec)
        return result

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name not in self.__dict__.get("_extras", ()):
            raise AttributeError(name)
        return lambda *args: self._call(name, *args)

    def capabilities(self) -> RobotCapabilities:
        return self._call("capabilities")

    def robot_config_digest(self) -> str:
        return self._call("robot_config_digest")

    def clock_domain(self) -> str:
        return self._call("clock_domain")

    def now_ns(self) -> int:
        return self._call("now_ns")

    def get_imu(self) -> ImuSample | None:
        return self._call("get_imu")

    def get_depth(self) -> DepthSample | None:
        return self._call("get_depth")

    def get_camera(self) -> Observation | None:
        return self._call("get_camera")

    def get_sonar(self) -> Observation | None:
        return self._call("get_sonar")

    def get_thruster_state(self) -> tuple[ThrusterState, ...]:
        return self._call("get_thruster_state")

    def get_power_state(self) -> BatteryState | None:
        return self._call("get_power_state")

    def get_health(self) -> SystemHealth:
        return self._call("get_health")

    def send(self, command: AllocatedCommand) -> CommandAck:
        return self._call("send", command)


def record_driver_event(
    tape: Tape | None, event_type: EventType, module: str, trace: Any, payload: Any
) -> None:
    if tape is not None:
        tape.marker(
            "driver_emit",
            event_type=event_type.value,
            module=module,
            trace_id=str(trace),
            payload=json.loads(json.dumps(payload, default=str)),
        )


def write_capture_files(
    run_dir: Path, tape: Tape, context: MissionContext, robot: RobotConfig, run_uuid: Any, producer: str
) -> None:
    tape.close()
    (run_dir / CONTEXT).write_text(context.model_dump_json(), encoding="utf-8")
    (run_dir / ROBOT).write_text(robot.model_dump_json(), encoding="utf-8")
    meta = {
        "tape_version": TAPE_VERSION,
        "records": tape.records,
        "run_uuid": str(run_uuid),
        "producer": producer,
        "robot_config_digest": robot.content_digest(),
        "code_digest": code_digest(),
        "note": "Everything the deployment runtime received through RobotHardwareInterface, in call order.",
    }
    (run_dir / META).write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")


# ------------------------------------------------------------------------------------------ playback
def read_tape(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class PlaybackHardware(RobotHardwareInterface):
    """Serves the tape in order. A call whose name differs from the tape raises ``TapeDivergence``."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records or records[0].get("k") != "header":
            raise TapeDivergence("tape has no header")
        head = records[0]
        if head.get("version") != TAPE_VERSION:
            raise TapeDivergence(f"unsupported tape version {head.get('version')!r}")
        self._records = records
        self._pos = 1
        self._extras = tuple(head.get("extra_methods", ()))
        attrs = head.get("attributes", {})
        self.adapter_name = str(attrs.get("adapter_name", "playback"))
        self.is_physical = bool(attrs.get("is_physical", False))
        self.command_mismatches: list[int] = []
        self.commands_checked = 0

    @property
    def position(self) -> int:
        return self._pos

    def peek(self) -> dict[str, Any] | None:
        return self._records[self._pos] if self._pos < len(self._records) else None

    def take_marker(self) -> dict[str, Any]:
        rec = self.peek()
        if rec is None or rec["k"] == "call":
            got = "end of tape" if rec is None else f"call {rec['n']}"
            raise TapeDivergence(f"tape record {self._pos}: driver expected a marker, tape has {got}")
        self._pos += 1
        return rec

    def _call(self, name: str, *args: Any) -> Any:
        rec = self.peek()
        if rec is None or rec["k"] != "call" or rec["n"] != name:
            have = (
                "end of tape" if rec is None else (rec["n"] if rec["k"] == "call" else f"marker {rec['k']}")
            )
            raise TapeDivergence(f"tape record {self._pos}: runtime called {name}, tape has {have}")
        self._pos += 1
        if name == "send":
            self.commands_checked += 1
            if [encode(a) for a in args] != rec["a"]:
                self.command_mismatches.append(self._pos - 1)
        if "p" in rec:
            return pickle.loads(base64.b64decode(rec["p"]))
        return decode(rec["r"])

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name not in self.__dict__.get("_extras", ()):
            raise AttributeError(name)
        return lambda *args: self._call(name, *args)

    def capabilities(self) -> RobotCapabilities:
        return self._call("capabilities")

    def robot_config_digest(self) -> str:
        return self._call("robot_config_digest")

    def clock_domain(self) -> str:
        return self._call("clock_domain")

    def now_ns(self) -> int:
        return self._call("now_ns")

    def get_imu(self) -> ImuSample | None:
        return self._call("get_imu")

    def get_depth(self) -> DepthSample | None:
        return self._call("get_depth")

    def get_camera(self) -> Observation | None:
        return self._call("get_camera")

    def get_sonar(self) -> Observation | None:
        return self._call("get_sonar")

    def get_thruster_state(self) -> tuple[ThrusterState, ...]:
        return self._call("get_thruster_state")

    def get_power_state(self) -> BatteryState | None:
        return self._call("get_power_state")

    def get_health(self) -> SystemHealth:
        return self._call("get_health")

    def send(self, command: AllocatedCommand) -> CommandAck:
        return self._call("send", command)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_capture(bundle: Path) -> dict[str, Any]:
    """Digest-check every non-truth file replay needs; list manifest files that are absent (e.g. removed truth)."""
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    files: dict[str, str] = manifest["files"]
    problems: list[str] = []
    for rel in (TAPE, CONTEXT, ROBOT, META):
        if rel not in files:
            problems.append(f"bundle has no {rel} (recorded before observation capture existed)")
    absent = sorted(rel for rel in files if not (bundle / rel).exists())
    needed = [
        rel
        for rel in files
        if rel.startswith((f"{CAPTURE_DIR}/", "objects/")) or rel == "config.resolved.yaml"
    ]
    for rel in needed:
        path = bundle / rel
        if not path.exists():
            problems.append(f"required file missing: {rel}")
        elif _sha256(path) != files[rel]:
            problems.append(f"required file corrupt: {rel}")
    return {"manifest": manifest, "problems": problems, "absent": absent, "verified": len(needed)}


def replay_inference(bundle: Path, out_dir: Path) -> dict[str, Any]:
    """Re-run inference + decision planes from the captured observations only. Returns a playback report."""
    check = verify_capture(bundle)
    if check["problems"]:
        return {"ran": False, "problems": check["problems"], "absent_manifest_files": check["absent"]}
    inputs = check["manifest"]["replay_inputs"]
    meta = json.loads((bundle / META).read_text(encoding="utf-8"))
    settings = ConradSettings.model_validate(inputs["config_resolved"])
    if settings.config_digest() != inputs["config_digest"]:
        return {"ran": False, "problems": ["stored configuration does not match its recorded digest"]}
    rcfg = runtime_config(inputs["mission_runtime_config"])
    context = MissionContext.model_validate_json((bundle / CONTEXT).read_text(encoding="utf-8"))
    robot = RobotConfig.model_validate_json((bundle / ROBOT).read_text(encoding="utf-8"))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    robot_path = out_dir / "robot_config.captured.json"
    robot_path.write_text(robot.model_dump_json(), encoding="utf-8")
    settings = settings.model_copy(
        update={"runtime": settings.runtime.model_copy(update={"robot_config": str(robot_path)})}
    )
    shutil.copytree(bundle / "objects", out_dir / "objects")
    seed = int(inputs["scenario_seed"])
    ids = IdFactory(seed)
    run_uuid = ids.child("run").new()
    if str(run_uuid) != meta["run_uuid"]:
        return {"ran": False, "problems": [f"run UUID {run_uuid} differs from captured {meta['run_uuid']}"]}
    hw = PlaybackHardware(list(read_tape(bundle / TAPE)))
    migrate(out_dir / "conrad.sqlite")
    engine = make_engine(out_dir / "conrad.sqlite")
    divergence: str | None = None
    ticks = 0
    log: EventLog | None = None
    try:
        ctx = RunContext(
            run_uuid,
            context.mission_id,
            None,
            ids.child("events"),
            hw.now_ns,
            hw.clock_domain(),
            meta["producer"],
        )
        log = EventLog(ctx, out_dir / "events.jsonl")
        rt = MissionRuntime(
            hw,
            context,
            Repository(engine),
            ObjectStore(out_dir / "objects"),
            log,
            settings,
            ids.child("runtime"),
            rcfg,
            seed,
            operator_inbox=bundle / "notes" / "operator_requests.jsonl",
        )
        _expect(hw, "start")
        rt.start()
        while True:
            rec = hw.take_marker()
            if rec["k"] == "driver_emit":
                log.emit(EventType(rec["event_type"]), rec["module"], rt.s.run_id, rec["payload"])
            elif rec["k"] == "tick":
                rt.tick()
                ticks += 1
            elif rec["k"] == "finish":
                rt.finish()
                break
            else:
                raise TapeDivergence(f"unknown marker {rec['k']!r}")
        log.close()
        _expect(hw, "artifacts")
        write_mission_artifacts(rt, out_dir)
        if hw.peek() is not None:
            raise TapeDivergence(f"tape record {hw.position}: runtime finished but the tape continues")
    except TapeDivergence as exc:
        divergence = str(exc)
    finally:
        if log is not None:
            log.close()
        engine.dispose()
    now_code = code_digest()
    return {
        "ran": True,
        "code_digest_recorded": meta.get("code_digest"),
        "code_digest_now": now_code,
        "same_code": meta.get("code_digest") == now_code,
        "ticks": ticks,
        "tape_records": meta["records"],
        "tape_divergence": divergence,
        "commands_checked": hw.commands_checked,
        "command_mismatches": len(hw.command_mismatches),
        "verified_files": check["verified"],
        "absent_manifest_files": check["absent"],
    }


def _expect(hw: PlaybackHardware, kind: str) -> None:
    rec = hw.take_marker()
    if rec["k"] != kind:
        raise TapeDivergence(f"expected marker {kind!r}, tape has {rec['k']!r}")


# ------------------------------------------------------------------------------------------ comparison
def _file_digest(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _table_digests(db: Path) -> dict[str, str]:
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        out = {}
        for name in names:
            h = hashlib.sha256()
            for row in con.execute(f'SELECT * FROM "{name}" ORDER BY rowid'):
                h.update(repr(row).encode())
            out[name] = h.hexdigest()
        return out
    finally:
        con.close()


def _commands(events_path: Path) -> list[str]:
    return [
        json.dumps(e.payload, sort_keys=True, default=str)
        for e in read_events(events_path)
        if e.event_type is EventType.COMMAND_SENT
    ]


def _first_diff(a: list[Any], b: list[Any]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_outputs(original: Path, replayed: Path) -> dict[str, Any]:
    """Byte and signature comparison of every deployment-side output (events, SQLite, mission/*, commands)."""
    out: dict[str, Any] = {}
    ev_a, ev_b = original / "events.jsonl", replayed / "events.jsonl"
    sig_a = event_signature(list(read_events(ev_a)))
    sig_b = event_signature(list(read_events(ev_b))) if ev_b.exists() else []
    idx = _first_diff(sig_a, sig_b)
    out["events"] = {
        "original": len(sig_a),
        "replayed": len(sig_b),
        "signature_equal": idx is None,
        "first_difference": idx,
        "bytes_equal": _file_digest(ev_a) == _file_digest(ev_b),
    }
    ta = _table_digests(original / "conrad.sqlite")
    tb = _table_digests(replayed / "conrad.sqlite") if (replayed / "conrad.sqlite").exists() else {}
    out["sqlite_tables"] = {k: ta.get(k) == tb.get(k) for k in sorted(set(ta) | set(tb))}
    mission = (
        sorted(p.name for p in (original / "mission").glob("*")) if (original / "mission").exists() else []
    )
    out["mission_files"] = {
        n: _file_digest(original / "mission" / n) == _file_digest(replayed / "mission" / n) for n in mission
    }
    ca, cb = _commands(ev_a), (_commands(ev_b) if ev_b.exists() else [])
    out["commands"] = {"original": len(ca), "replayed": len(cb), "equal": ca == cb}
    out["equal"] = bool(
        out["events"]["bytes_equal"]
        and all(out["sqlite_tables"].values())
        and all(out["mission_files"].values())
        and out["commands"]["equal"]
    )
    return out


def inference_fingerprint(run_dir: Path) -> str:
    """One digest over all deployment-side outputs, for quick equality checks in tests."""
    h = hashlib.sha256()
    h.update((_file_digest(run_dir / "events.jsonl") or "-").encode())
    for name, dig in sorted(_table_digests(run_dir / "conrad.sqlite").items()):
        h.update(f"{name}:{dig}".encode())
    for p in sorted((run_dir / "mission").glob("*")):
        h.update(f"{p.name}:{_file_digest(p)}".encode())
    return h.hexdigest()
