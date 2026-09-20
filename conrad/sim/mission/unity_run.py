"""Unity backend of the integrated mission driver (gates I1-I3, formal path) and of NAV-001..NAV-006. TRUTH SIDE.

* ``run_unity_scenario`` / ``prepare_unity``: ``I1-UNITY`` and ``I3-UNITY`` through ``UnityMissionWorld`` and the
  unchanged ``MissionRuntime``; writes the same run bundle as ``conrad.sim.mission.run`` plus ``unity/`` (player
  inputs + log) and ``backend = "unity"`` in the replay inputs.
* ``replay_unity_run``: verify every digest (fail closed), re-execute from the stored seed/config against a fresh
  player, compare event/decision/revision signatures (exact) and the Unity true trajectory (declared tolerance).
* ``run_unity_nav`` / ``replay_unity_nav`` (``I2-UNITY-NAV``): NAV-001..NAV-006 geometries as Unity colliders;
  estimator -> planner -> trajectory -> controller -> allocator -> safety -> CommandGateway ->
  UnityRobotHardware. Unity truth is read on this side for metrics only.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from conrad.adapters.unity import FaultInjectionRequest
from conrad.adapters.unity import FaultType as UnityFaultType
from conrad.evaluation.nav_benchmarks.runner import ROBOT_CONFIG as NAV_ROBOT_CONFIG
from conrad.evaluation.nav_benchmarks.runner import _goal, _metrics
from conrad.evaluation.nav_benchmarks.scenarios import (
    ScenarioGeometry,
    SyntheticPositionFixRenderer,
    geometry_of,
    load_scenario,
)
from conrad.orchestration.artifacts import write_mission_artifacts
from conrad.orchestration.evaluation import evaluate_run_dir
from conrad.orchestration.mission import MissionRuntime
from conrad.orchestration.mission_config import MissionRuntimeConfig, runtime_config
from conrad.persistence.db import make_engine, migrate
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.replay_store import ReplayIntegrityError, verify_bundle, write_bundle_manifest
from conrad.persistence.repository import Repository
from conrad.robotics.hardware.config import load_robot_config
from conrad.robotics.navigation import GoalStatus, NavigationStack, NavigationStackConfig
from conrad.runtime.command_gateway import CommandGateway
from conrad.runtime.event_log import EventLog, RunContext
from conrad.schemas.base import ARCHITECTURE_ID, STACK_ID
from conrad.schemas.events import EventType
from conrad.schemas.frames import WORLD, Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import RobotConfig
from conrad.schemas.timebase import TimeStamp
from conrad.settings import (
    REPO_ROOT,
    CommandMode,
    ConradSettings,
    ExecutionLane,
    RuntimeSettings,
    snapshot_yaml,
)
from conrad.sim.mission.capture import TAPE, RecordingHardware, Tape, record_driver_event, write_capture_files
from conrad.sim.mission.options import MissionWorldOptions, world_options
from conrad.sim.mission.replay import compare
from conrad.sim.mission.run import settings_for
from conrad.sim.mission.scenarios import resolve
from conrad.sim.mission.unity_world import (
    UnityMissionWorld,
    UnityWorldOptions,
    neutral_density,
    write_robot_config,
)
from conrad.sim.unity.player import UnityPlayerSession, find_player, scenario_document
from conrad.sim.unity.scene import BoxPrimitive, CapsulePrimitive, SceneGeometry

PRODUCER = "conrad-mission-unity-0.1"
DRIVER = "conrad.sim.mission.unity_driver"
BACKEND = "unity"
TRAJECTORY_TOLERANCE_M = 1e-9  # declared replay tolerance: lock-step Unity is deterministic (U0: 0.0)

# Unity scenario IDs: (kernel scenario whose definition they reuse, extra overrides). Durations follow the
# surrogate I1/I3 tests (24 s / 30 s) lengthened for I1 so the robot covers the transit lane.
UNITY_SCENARIOS: dict[str, dict[str, Any]] = {
    "I1-UNITY": {
        "base": "I1-SPATIAL",
        "world": {"ecological_enabled": False},
        "runtime": {"model2e_enabled": False, "duration_s": 60.0, "control_period_s": 0.1},
    },
    "I3-UNITY": {
        "base": "I3-STRUCTURAL",
        "world": {"ecological_enabled": False},
        "runtime": {"model2e_enabled": False, "duration_s": 30.0, "control_period_s": 0.1},
    },
    # Gate I4 (closed active inspection): the FLAGSHIP-I4 world and the ACTIVE-MCBR-E004 budgets. The planner under
    # comparison is set per run through the stored runtime config (tests/unity_live/test_i4_unity.py).
    "I4-UNITY": {
        "base": "FLAGSHIP-I4",
        "world": {"ecological_enabled": False},
        "runtime": {
            "model2e_enabled": False,
            "duration_s": 100.0,
            "control_period_s": 0.1,
            "max_plans_per_need": 4,
        },
    },
}
# Gate I6 (multi-domain): the kernel I6 scenarios on Unity with Twin2E (optics grid + biofouling colours, Python-side
# Twin2E payload channels). Same worlds, arms and runtime as the surrogate (configs/eval/i6_multidomain.yaml).
UNITY_SCENARIOS["I6-UNITY-TURBID"] = {
    "base": "I6-MULTIDOMAIN-TURBID",
    "world": {"ecological_enabled": True},
    "runtime": {"model2e_enabled": True, "control_period_s": 0.1},
}
UNITY_SCENARIOS["I6-UNITY-CLEAR"] = {
    "base": "I6-MULTIDOMAIN-CLEAR",
    "world": {"ecological_enabled": True},
    "runtime": {"model2e_enabled": True, "control_period_s": 0.1},
}
# Flagship integrated Unity mission (docs/audits/FLAGSHIP_UNITY.md). The kernel scenario carries the three
# declared stressors; here only the Unity control period is set. Twin2E stays ON (production default
# model2e_enabled), so this is the full 2S + 2T + 2E stack with the frozen production planner.
UNITY_SCENARIOS["FLAGSHIP-UNITY"] = {
    "base": "FLAGSHIP-UNITY",
    "world": {},
    "runtime": {"control_period_s": 0.1},
}
NAV_UNITY_IDS = tuple(f"NAV-00{i}" for i in range(1, 7))
UNITY_SCENARIO_IDS = (*UNITY_SCENARIOS, "I2-UNITY-NAV")


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve_unity(scenario_id: str, mission_section: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if scenario_id not in UNITY_SCENARIOS:
        raise KeyError(f"unknown Unity mission scenario {scenario_id!r}; known: {sorted(UNITY_SCENARIOS)}")
    spec = UNITY_SCENARIOS[scenario_id]
    world, runtime = resolve(str(spec["base"]), mission_section)
    return _merge(world, spec["world"]), _merge(runtime, spec["runtime"])


def git_commit() -> str:
    """Read-only lookup of the checked-out commit (no git command is executed)."""
    git = REPO_ROOT / ".git"
    head = git / "HEAD"
    if not head.exists():
        return "UNAVAILABLE_NO_GIT_METADATA"
    ref = head.read_text(encoding="utf-8").strip()
    if not ref.startswith("ref: "):
        return ref
    loose = git / ref[5:]
    if loose.exists():
        return loose.read_text(encoding="utf-8").strip()
    packed = git / "packed-refs"
    if packed.exists():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref[5:]):
                return line.split(" ", 1)[0]
    return "UNAVAILABLE_UNRESOLVED_REF"


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def player_identity() -> dict[str, Any]:
    player = find_player()
    info = None if player is None else player.parent / "conrad_build_info.json"
    return {
        "player": None if player is None else player.name,
        "player_exe_sha256": None if player is None else _sha256(player),
        "build_info": json.loads(info.read_text(encoding="utf-8")) if info and info.is_file() else None,
    }


# ====================================================================================== integrated missions
@dataclass
class UnitySession:
    """One prepared Unity mission: truth world + unchanged deployment runtime, stepped by the driver."""

    scenario_id: str
    settings: ConradSettings
    run_name: str
    run_dir: Path
    world: UnityMissionWorld
    runtime: MissionRuntime
    log: EventLog
    repo: Repository
    engine: Any
    rcfg: MissionRuntimeConfig
    wopts: MissionWorldOptions
    uopts: UnityWorldOptions
    run_uuid: Any
    inspection_s: float | None = None
    finished: bool = False
    tape: Tape | None = None

    def step(self) -> None:
        rt, world = self.runtime, self.world
        if (
            self.inspection_s is None
            and rt.executive.active is not None
            and rt.executive.active.purpose == "INSPECT"
        ):
            self.inspection_s = world.t_s
        for fired in world.due_faults(self.inspection_s):
            record_driver_event(self.tape, EventType.FAULT_INJECTED, DRIVER, rt.s.run_id, fired)
            self.log.emit(EventType.FAULT_INJECTED, DRIVER, rt.s.run_id, fired)
        if self.tape is not None:
            self.tape.marker("tick")
        rt.tick()
        world.advance(self.rcfg.control_period_s)

    def run(self) -> None:
        for _ in range(round(self.rcfg.duration_s / self.rcfg.control_period_s)):
            self.step()

    def finish(self) -> dict[str, Any]:
        rt, world = self.runtime, self.world
        if self.tape is not None:
            self.tape.marker("finish")
        rt.finish()
        self.log.close()
        world.recorder.snapshot_target("end", world.t2t, world.target, world.t_s)
        world.recorder.meta["inspection_goal_started_s"] = self.inspection_s
        world.recorder.trajectory = list(world.tracker.trajectory)
        world.recorder.series["unity_geometric_capture"] = list(world.hardware.capture_log)
        world.recorder.meta["unity_forwarded_frames"] = dict(world.hardware.forwarded_counts)
        world.recorder.meta["unity_validity_level"] = world.hardware.validity_level.value
        if world.t2e is not None:  # gate I6: every Twin2E optics refresh sent to Unity (truth side)
            world.recorder.series["unity_optics_updates"] = list(world.optics_updates)
        truth = world.truth_access()
        world.recorder.finish(truth, self.run_dir)
        if self.tape is not None:
            self.tape.marker("artifacts")
        runtime_metrics = write_mission_artifacts(rt, self.run_dir)
        if self.tape is not None:
            write_capture_files(
                self.run_dir, self.tape, world.context, rt.robot_config, self.run_uuid, PRODUCER
            )
        exit_code = world.close()
        self.engine.dispose()
        report = evaluate_run_dir(self.run_dir)
        report["runtime"] = runtime_metrics
        report["unity"] = {
            "player_exit_code": exit_code,
            "scene_digest": world.scene_digest,
            "scene_conversion": world.scene_report.model_dump(mode="json"),
            "forwarded_frames": dict(world.hardware.forwarded_counts),
            "out_of_order_packets": world.hardware.out_of_order_packets,
            "eco_conversion": None if world.eco_report is None else world.eco_report.model_dump(mode="json"),
            "optics_updates": len(world.optics_updates),
        }
        (self.run_dir / "reports").mkdir(exist_ok=True)
        (self.run_dir / "reports" / "metrics.json").write_text(
            json.dumps(report, indent=1, default=str), encoding="utf-8"
        )
        inputs = unity_replay_inputs(self)
        engine = make_engine(self.run_dir / "conrad.sqlite")
        try:
            digests = Repository(engine).referenced_digests(rt.s.run_id)
        finally:
            engine.dispose()
        write_bundle_manifest(self.run_dir, self.run_name, inputs, digests)
        self.finished = True
        return {"run_id": self.run_name, "run_dir": str(self.run_dir), "report": report}

    def abort(self) -> None:
        """Stop the player and release the database without writing a bundle (error paths)."""
        if not self.finished:
            self.world.close()
            self.log.close()
            self.engine.dispose()
            if self.tape is not None:
                self.tape.close()


def unity_replay_inputs(s: UnitySession) -> dict[str, Any]:
    world = s.world
    sensors = [x.model_dump(mode="json") for x in world.context.sensors]
    return {
        "backend": BACKEND,
        "scenario_id": s.scenario_id,
        "run_name": s.run_name,
        "scenario_seed": s.settings.run.seed,
        "scenario_version": world.scenario.scenario_version,
        "twin_versions": {
            "twin2s": world.scenario.scenario_version,
            "twin2t": world.t2t.config.generator_version,
            "twin2e": None if world.t2e is None else world.scenario.scenario_version,
        },
        "model_versions": {
            "model2t": "model2t-analytic-0.1.0",
            "planner": s.rcfg.planner,
            "producer": PRODUCER,
        },
        "robot_config_digest": world.robot_config.content_digest(),
        "sensor_configuration": hashlib.sha256(json.dumps(sensors, sort_keys=True).encode()).hexdigest(),
        "config_digest": s.settings.config_digest(),
        "config_resolved": json.loads(s.settings.canonical_json()),
        "mission_world_options": s.wopts.model_dump(mode="json"),
        "mission_runtime_config": s.rcfg.model_dump(mode="json"),
        "unity_world_options": s.uopts.model_dump(mode="json"),
        "unity": {
            **player_identity(),
            "scene_digest": world.scene_digest,
            "robot_config_path": world.robot_config_path,
            "validity_level": "L1_APPROXIMATE_PHYSICS",
        },
        "git_commit": git_commit(),
        "seeds": {"run": s.settings.run.seed},
        "architecture_id": ARCHITECTURE_ID,
        "stack_id": STACK_ID,
    }


def prepare_unity(
    scenario_id: str,
    config: str | Path | ConradSettings,
    run_id: str | None = None,
    runs_root: Path | None = None,
    seed: int | None = None,
    uopts: UnityWorldOptions | None = None,
    capture: bool = True,
    stored_world: MissionWorldOptions | None = None,
    stored_runtime: MissionRuntimeConfig | None = None,
) -> UnitySession:
    settings = settings_for(config)
    if seed is not None:
        settings = settings.model_copy(update={"run": settings.run.model_copy(update={"seed": seed})})
    u = uopts or UnityWorldOptions()
    world_raw, runtime_raw = resolve_unity(scenario_id, dict(settings.sim.get("mission", {})))
    wopts = stored_world or world_options(world_raw)
    rcfg = stored_runtime or runtime_config(runtime_raw)
    steps = rcfg.control_period_s * 1e9 / u.physics_dt_ns
    if abs(steps - round(steps)) > 1e-9 or round(steps) < 1:
        raise ValueError(
            f"control_period_s {rcfg.control_period_s} must be a whole number of Unity physics steps "
            f"({u.physics_dt_ns} ns)"
        )
    seed_ = settings.run.seed
    run_name = run_id or f"{scenario_id}-s{seed_}-{settings.config_digest()[:8]}"
    root = runs_root or settings.resolve(settings.paths.runs_dir)
    run_dir = root / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    ids = IdFactory(seed_)
    run_uuid = ids.child("run").new()
    world = UnityMissionWorld.build(seed_, scenario_id, wopts, run_uuid, run_dir, u)
    try:
        # The runtime loads the RobotConfig the Unity vehicle was built from (same digest, checked at start).
        settings = settings.model_copy(
            update={"runtime": settings.runtime.model_copy(update={"robot_config": world.robot_config_path})}
        )
        (run_dir / "config.resolved.yaml").write_text(snapshot_yaml(settings), encoding="utf-8")
        db_path = run_dir / "conrad.sqlite"
        migrate(db_path)
        engine = make_engine(db_path)
        repo = Repository(engine)
        tape = Tape(run_dir / TAPE) if capture else None
        hw = world.hardware if tape is None else RecordingHardware(world.hardware, tape)
        ctx = RunContext(
            run_uuid,
            world.context.mission_id,
            None,
            ids.child("events"),
            hw.now_ns,
            hw.clock_domain(),
            PRODUCER,
        )
        log = EventLog(ctx, run_dir / "events.jsonl")
        runtime = MissionRuntime(
            hw,
            world.context,
            repo,
            world.store,
            log,
            settings,
            ids.child("runtime"),
            rcfg,
            seed_,
            operator_inbox=run_dir / "notes" / "operator_requests.jsonl",
        )
        world.hardware.set_estimated_pose_provider(runtime.estimated_pose)
        world.advance(0.1)
        if tape is not None:
            tape.marker("start")
        runtime.start()
    except BaseException:
        world.close()
        raise
    return UnitySession(
        scenario_id,
        settings,
        run_name,
        run_dir,
        world,
        runtime,
        log,
        repo,
        engine,
        rcfg,
        wopts,
        u,
        run_uuid,
        tape=tape,
    )


def run_unity_scenario(
    scenario_id: str,
    config: str | Path | ConradSettings,
    run_id: str | None = None,
    runs_root: Path | None = None,
    seed: int | None = None,
    stored_world: MissionWorldOptions | None = None,
    stored_runtime: MissionRuntimeConfig | None = None,
) -> dict[str, Any]:
    """Run ``I1-UNITY`` / ``I3-UNITY`` (or ``I2-UNITY-NAV``: all six NAV benchmarks) and store the bundle."""
    if scenario_id == "I2-UNITY-NAV":
        settings = settings_for(config)
        s = settings.run.seed if seed is None else seed
        root = (runs_root or settings.resolve(settings.paths.runs_dir)) / (run_id or f"I2-UNITY-NAV-s{s}")
        results = {b: run_unity_nav(b, s, root / b) for b in NAV_UNITY_IDS}
        return {"run_id": root.name, "run_dir": str(root), "report": results}
    session = prepare_unity(
        scenario_id, config, run_id, runs_root, seed, stored_world=stored_world, stored_runtime=stored_runtime
    )
    try:
        session.run()
        return session.finish()
    except BaseException:
        session.abort()
        raise


def _trajectory(run_dir: Path) -> np.ndarray:
    truth = json.loads((run_dir / "truth" / "truth_record.json").read_text(encoding="utf-8"))
    return np.asarray(truth["trajectory"], dtype=np.float64)


def replay_unity_run(run_dir: Path, scratch: Path | None = None) -> dict[str, Any]:
    """Raises ReplayIntegrityError when the bundle does not verify. Returns the comparison report."""
    manifest = verify_bundle(run_dir, ObjectStore(run_dir / "objects"))
    inputs = manifest.replay_inputs
    if inputs.get("backend") != BACKEND:
        raise ReplayIntegrityError([f"not a Unity bundle (backend={inputs.get('backend')!r})"])
    for key in ("scenario_id", "config_resolved", "run_name"):
        if key not in inputs:
            raise ReplayIntegrityError([f"replay input missing: {key}"])
    settings = ConradSettings.model_validate(inputs["config_resolved"])
    if settings.config_digest() != inputs["config_digest"]:
        raise ReplayIntegrityError(["stored configuration does not match its recorded digest"])
    try:
        wopts = world_options(inputs["mission_world_options"]) if "mission_world_options" in inputs else None
        rcfg = (
            runtime_config(inputs["mission_runtime_config"]) if "mission_runtime_config" in inputs else None
        )
    except ValueError as exc:
        raise ReplayIntegrityError([f"stored mission options no longer validate: {exc}"]) from exc
    now = player_identity()["player_exe_sha256"]
    root = scratch or Path(tempfile.mkdtemp(prefix="conrad-unity-replay-"))
    out = run_unity_scenario(
        str(inputs["scenario_id"]),
        settings,
        run_id=str(inputs["run_name"]),
        runs_root=root,
        stored_world=wopts,
        stored_runtime=rcfg,
    )
    replayed = Path(out["run_dir"])
    report = compare(run_dir, replayed)
    a, b = _trajectory(run_dir), _trajectory(replayed)
    diff = float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else math.inf
    report["trajectory"] = {
        "samples": [len(a), len(b)],
        "max_abs_difference": diff,
        "tolerance": TRAJECTORY_TOLERANCE_M,
        "equal": diff <= TRAJECTORY_TOLERANCE_M,
    }
    report["equal"] = bool(report["equal"] and report["trajectory"]["equal"])
    report["replay_dir"] = str(replayed)
    report["verified_files"] = len(manifest.files)
    report["verified_objects"] = len(manifest.object_digests)
    report["same_player_binary"] = now == inputs.get("unity", {}).get("player_exe_sha256")
    return report


# ====================================================================================== NAV-001..006 on Unity
def unity_nav_robot_config(battery_capacity_j: float | None = None) -> tuple[RobotConfig, str]:
    """``sim_reference`` with AHRS orientation from the Unity IMU (the kernel benchmark's IMU provides it).

    ``battery_capacity_j`` (fault cases only) replaces the SYNTHETIC_ONLY battery capacity."""
    base = load_robot_config(NAV_ROBOT_CONFIG)
    doc = base.model_dump(mode="json")
    for s in doc["sensors"]:
        if s["modality"] == "IMU":
            s["parameters"] = {"report_orientation": True, "gyro_noise_std": 0.002}
    doc["config_name"] = f"{base.config_name}_unity_nav"
    if battery_capacity_j is not None:
        doc["battery"]["capacity_j"] = {
            "value": float(battery_capacity_j),
            "units": "J",
            "source": "SYNTHETIC_ONLY",
        }
        doc["config_name"] += f"_battery{round(battery_capacity_j)}J"
    robot = RobotConfig.model_validate(doc)
    return robot, write_robot_config(robot)


def nav_scene(scn: dict[str, Any], geo: ScenarioGeometry) -> SceneGeometry:
    """NAV obstacles as colliders (replaces the player's default world; an empty list leaves open water)."""
    prims: list[BoxPrimitive | CapsulePrimitive] = []
    for k, ob in enumerate(scn.get("obstacles", [])):
        if ob["type"] == "vertical_cylinder":
            cx, cy = (float(v) for v in ob["center_xy"])
            prims.append(
                CapsulePrimitive(
                    id=f"cyl{k}", p0_m=(cx, cy, -80.0), p1_m=(cx, cy, 20.0), radius_m=ob["radius_m"]
                )
            )
        elif ob["type"] == "pipe":
            assert geo.pipeline is not None
            pts = geo.pipeline
            prims += [
                CapsulePrimitive(
                    id=f"pipe{k}-{i}",
                    p0_m=tuple(float(v) for v in pts[i]),
                    p1_m=tuple(float(v) for v in pts[i + 1]),
                    radius_m=float(ob["radius_m"]),
                )
                for i in range(len(pts) - 1)
            ]
        elif ob["type"] == "seabed":
            z = float(ob["z_m"])
            prims.append(
                BoxPrimitive(id=f"seabed{k}", center_m=(0.0, 0.0, z - 1.0), size_m=(400.0, 400.0, 2.0))
            )
        else:
            raise ValueError(f"unknown obstacle type {ob['type']!r}")
    return SceneGeometry(primitives=tuple(prims))


def nav_environment(scn: dict[str, Any], robot: RobotConfig) -> dict[str, Any]:
    gusts = []
    for f in scn.get("faults", []):
        if f["type"] != "CURRENT_GUST":
            raise ValueError(f"NAV fault {f['type']} is not realised on the Unity path")
        gusts.append(
            {"start_s": float(f["t_s"]), "duration_s": float(f["duration_s"]), "velocity_mps": f["vector"]}
        )
    current: dict[str, Any] = {"kind": "constant", "velocity_mps": [float(v) for v in scn["current_mps"]]}
    if gusts:
        current["gusts"] = gusts
    return {
        "water_density_kgm3": neutral_density(robot),
        "surface_z_m": 0.0,
        "turbidity": 0.0,
        "current": current,
    }


def _files_digest(folder: Path) -> dict[str, str]:
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "nav_manifest.json"
    }


@dataclass(frozen=True)
class NavFault:
    """A fault injected on the hardware-adapter side of a Unity NAV run (gate I2 "faults reach defined safe
    states"). ``kind`` is a Unity ``FaultType`` (sent through the bridge) or ``FIX_OUTAGE`` (the synthetic USBL-like
    fix stops from ``t_s`` on)."""

    kind: str
    t_s: float
    target: str | None = None
    magnitude: float = 1.0

    def as_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "t_s": self.t_s, "target": self.target, "magnitude": self.magnitude}


FIX_OUTAGE = "FIX_OUTAGE"


def run_unity_nav(
    benchmark_id: str,
    seed: int,
    bundle_dir: str | Path,
    fix_bias_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    uopts: UnityWorldOptions | None = None,
    faults: tuple[NavFault, ...] = (),
    duration_s: float | None = None,
    battery_capacity_j: float | None = None,
) -> dict[str, Any]:
    """One NAV benchmark through Unity. The stack sees only ``UnityRobotHardware``; ``fix_bias_m`` (evaluation
    counterfactual) biases the synthetic USBL-like fix to show that control acts on the ESTIMATED state.

    ``faults`` are injected at their time on the adapter side; with faults the bundle also holds ``safety.json``
    (per-step safety state, reasons, the authorized thruster commands and whether the RHI accepted them)."""
    if benchmark_id not in NAV_UNITY_IDS:
        raise KeyError(f"{benchmark_id} is not part of the Unity NAV set {NAV_UNITY_IDS}")
    u = uopts or UnityWorldOptions()
    out = Path(bundle_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    scn = load_scenario(benchmark_id)
    robot, robot_path = unity_nav_robot_config(battery_capacity_j)
    ids = IdFactory(seed=seed)
    unknown = [f.kind for f in faults if f.kind != FIX_OUTAGE and f.kind not in UnityFaultType.__members__]
    if unknown:
        raise ValueError(f"unsupported NAV fault kinds {unknown}")
    run_s = float(scn["duration_s"]) if duration_s is None else float(duration_s)
    mission_id, run_id = ids.new(), ids.new()
    geo = geometry_of(scn)
    dt = float(scn["control_period_s"])
    dt_ns = round(dt * 1e9)
    if dt_ns % u.physics_dt_ns:
        raise ValueError(f"control period {dt} s is not a whole number of Unity physics steps")
    fixes = scn["fixes"]
    renderer = SyntheticPositionFixRenderer(
        mission_id,
        run_id,
        ids,
        np.random.default_rng([seed, 17]),
        fixes["period_s"],
        fixes["sigma_m"],
        [*fixes["outages"], *([f.t_s, math.inf] for f in faults if f.kind == FIX_OUTAGE)],
    )
    start = np.asarray(scn["start"], dtype=np.float64)
    scenario = scenario_document(
        ids.child("unity_scenario"),
        robot,
        seed=seed,
        initial_position_m=(float(start[0]), float(start[1]), float(start[2])),
        environment=nav_environment(scn, robot),
        scenario_version=f"unity-nav:{benchmark_id}",
    )
    scene = nav_scene(scn, geo)
    inflation = float(scn.get("planner_inflation_m", 0.5))
    tracker_sdf = None if geo.empty else geo.sdf
    radius = 0.5 * max(robot.dimensions_m.value)  # type: ignore[arg-type]
    store = ObjectStore(out / "objects")
    with UnityPlayerSession(
        robot,
        out / "unity",
        scenario,
        experiment_overrides={"physics_dt_ns": u.physics_dt_ns, "experiment_name": benchmark_id},
        player_path=u.player_path,
        startup_timeout_s=u.startup_timeout_s,
        request_timeout_ms=u.request_timeout_ms,
    ) as player:
        hw = player.hardware(ids.child("unity_rhi"), mission_id, run_id, payload_store=store)
        hw.connect()
        hw.configure_scene(scene.to_json())
        hw.reset(seed)
        truth = player.truth()
        truth.connect(ids.child("unity_truth").new().hex)
        stack = NavigationStack(
            hw,
            robot,
            ids,
            mission_id,
            run_id,
            Pose(frame_id=WORLD, position_m=(float(start[0]), float(start[1]), float(start[2]))),
            config=NavigationStackConfig(control_period_s=dt),
            is_free=None if geo.empty else (lambda p: geo.sdf(p) > inflation),
            local_distance=None if geo.empty else geo.sdf,
        )
        gateway = CommandGateway(
            hw,
            robot,
            RuntimeSettings(command_mode=CommandMode.SIMULATED),
            ExecutionLane.SIMULATION,
            mission_id,
            run_id,
            state_age_s=stack.state_age_s,
        )
        goal, final = _goal(scn, ids)
        hw.step(100_000_000)  # let the first IMU/depth samples arrive (kernel runner: advance(0.1))
        traj = stack.set_goal(goal)
        route = np.array([p.pose.position_m for p in traj.points]) if traj else final[None, :]
        bias = np.asarray(fix_bias_m, dtype=np.float64)
        log_t, log_true, log_est, commands = [], [], [], []
        refused, collisions, in_collision, min_clear = 0, 0, False, math.inf
        arrived_s: float | None = None
        fixes_applied = 0
        pending = sorted((f for f in faults if f.kind != FIX_OUTAGE), key=lambda f: f.t_s)
        fault_acks: list[dict[str, Any]] = []
        safety_log: list[list[Any]] = []
        s = truth.get_ground_truth()
        for _ in range(round(run_s / dt)):
            now = hw.now_ns()
            if s.sim_time_ns != now:
                raise RuntimeError(f"truth endpoint at {s.sim_time_ns} ns, control at {now} ns")
            while pending and now >= round(pending[0].t_s * 1e9):
                f = pending.pop(0)
                fack = hw.inject_fault(
                    FaultInjectionRequest(
                        fault_id=f"{benchmark_id}-{f.kind}-{len(fault_acks)}",
                        fault_type=UnityFaultType(f.kind),
                        target=f.target,
                        start_time_ns=now,
                        magnitude=f.magnitude,
                    )
                )
                if not fack.accepted:
                    raise RuntimeError(f"Unity refused fault {f}: {fack.reason_codes}")
                fault_acks.append({**f.as_json(), "scheduled_time_ns": fack.scheduled_time_ns})
            biased = s.pose.model_copy(
                update={"position_m": tuple(float(v) for v in np.asarray(s.pose.position_m) + bias)}
            )
            fix = renderer(
                "sonar", biased, stack.estimator.get_state().pose, TimeStamp(time_ns=now, clock_domain="SIM")
            )
            if fix is not None:
                fixes_applied += int(stack.add_position_fix(fix))
            res = stack.step()
            if res.decision.authorized:
                ack = gateway.submit(res.command)
                commands.append([now, bool(ack.accepted), list(ack.reason_codes)])
            else:
                refused += 1
            if faults:
                a = res.assessment
                sent = dict(res.command.thruster_commands) if res.decision.authorized else None
                executed = bool(commands[-1][1]) if res.decision.authorized else None
                safety_log.append([now, a.state.value, list(a.reason_codes), sent, executed])
            hw.step(dt_ns)
            s = truth.get_ground_truth()
            p = np.asarray(s.pose.position_m, dtype=np.float64)
            if tracker_sdf is not None:
                clearance = float(tracker_sdf(p[None, :])[0]) - radius
                min_clear = min(min_clear, clearance)
                if clearance < 0 and not in_collision:
                    collisions, in_collision = collisions + 1, True
                elif clearance > 0.02:
                    in_collision = False
            if arrived_s is None and stack.status in (GoalStatus.ARRIVED, GoalStatus.COMPLETE):
                arrived_s = s.sim_time_ns / 1e9
            log_t.append(s.sim_time_ns / 1e9)
            log_true.append(p)
            log_est.append(np.asarray(stack.estimator.get_state().pose.position_m, dtype=np.float64))
        power = hw.get_power_state()
        validity = hw.validity_level.value
        hw.close()
    tt, tr, est = np.array(log_t), np.array(log_true), np.array(log_est)
    metrics = _metrics(scn, tt, tr, est, route, final, geo)
    ok = scn["success"]
    checks = {
        "goal_accepted": traj is not None,
        "final_position": metrics["final_position_error_m"] <= ok["final_tolerance_m"],
        "collisions": collisions <= ok["max_collisions"],
    }
    if "standoff_rms_max_m" in ok:
        checks["standoff"] = metrics["standoff_rms_error_m"] <= ok["standoff_rms_max_m"]
    if "station_rms_max_m" in ok:
        checks["station_keeping"] = metrics["station_rms_error_m"] <= ok["station_rms_max_m"]
    result = {
        "backend": BACKEND,
        "benchmark_id": benchmark_id,
        "description": scn["description"],
        "seed": seed,
        "fix_bias_m": list(bias),
        "simulation_validity_level": validity,
        "success": all(checks.values()),
        "checks": checks,
        **metrics,
        "final_true_position_m": [float(v) for v in tr[-1]],
        "final_estimated_position_m": [float(v) for v in est[-1]],
        "time_to_goal_s": arrived_s,
        "goal_status": stack.status.value,
        "energy_j": None if power is None or power.energy_used_j is None else round(power.energy_used_j, 1),
        "collisions": collisions,
        "min_clearance_m": None if tracker_sdf is None else round(min_clear, 3),
        "commands_refused_by_supervisor": refused,
        "gateway_accepted": gateway.accepted,
        "gateway_rejected": gateway.rejected,
        "position_fixes_delivered": renderer.delivered,
        "position_fixes_accepted": fixes_applied,
        "faults": fault_acks + [f.as_json() for f in faults if f.kind == FIX_OUTAGE],
        "duration_s": run_s,
        "estimator_inputs": "UnityRobotHardware readings + synthetic USBL-like fix; no truth client in the stack",
        "robot_config": robot_path,
        "robot_config_digest": robot.content_digest(),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "trajectory.json").write_text(
        json.dumps({"t_s": tt.tolist(), "true_m": tr.tolist(), "estimated_m": est.tolist()}), encoding="utf-8"
    )
    (out / "commands.json").write_text(json.dumps(commands), encoding="utf-8")
    if faults:
        (out / "safety.json").write_text(json.dumps(safety_log), encoding="utf-8")
    inputs = {
        "backend": BACKEND,
        "benchmark_id": benchmark_id,
        "seed": seed,
        "fix_bias_m": list(bias),
        "faults": [f.as_json() for f in faults],
        "duration_s": run_s,
        "battery_capacity_j": battery_capacity_j,
        "scenario": scn,
        "unity_world_options": u.model_dump(mode="json"),
        "robot_config_digest": robot.content_digest(),
        "unity": player_identity(),
        "git_commit": git_commit(),
    }
    manifest = {"inputs": inputs, "files": _files_digest(out)}
    (out / "nav_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return result


def replay_unity_nav(bundle_dir: str | Path, scratch: Path | None = None) -> dict[str, Any]:
    """Verify the NAV bundle digests, re-run with the stored inputs and compare (declared tolerance)."""
    src = Path(bundle_dir)
    manifest = json.loads((src / "nav_manifest.json").read_text(encoding="utf-8"))
    problems = [
        f"file {'missing' if not (src / rel).is_file() else 'corrupt'}: {rel}"
        for rel, digest in manifest["files"].items()
        if _sha256(src / rel) != digest
    ]
    if problems:
        raise ReplayIntegrityError(problems)
    inp = manifest["inputs"]
    if load_scenario(inp["benchmark_id"]) != inp["scenario"]:
        raise ReplayIntegrityError(["NAV scenario definition changed since the run"])
    root = scratch or Path(tempfile.mkdtemp(prefix="conrad-unity-nav-replay-"))
    dst = root / src.name
    run_unity_nav(
        inp["benchmark_id"],
        int(inp["seed"]),
        dst,
        tuple(inp["fix_bias_m"]),
        faults=tuple(NavFault(**f) for f in inp.get("faults", [])),
        duration_s=inp.get("duration_s"),
        battery_capacity_j=inp.get("battery_capacity_j"),
    )
    a = json.loads((src / "trajectory.json").read_text(encoding="utf-8"))
    b = json.loads((dst / "trajectory.json").read_text(encoding="utf-8"))
    ta, tb = np.asarray(a["true_m"]), np.asarray(b["true_m"])
    diff = float(np.max(np.abs(ta - tb))) if ta.shape == tb.shape else math.inf
    ca = json.loads((src / "commands.json").read_text(encoding="utf-8"))
    cb = json.loads((dst / "commands.json").read_text(encoding="utf-8"))
    return {
        "verified_files": len(manifest["files"]),
        "trajectory_max_abs_difference": diff,
        "tolerance": TRAJECTORY_TOLERANCE_M,
        "commands_equal": ca == cb,
        "commands": [len(ca), len(cb)],
        "equal": diff <= TRAJECTORY_TOLERANCE_M and ca == cb,
        "replay_dir": str(dst),
        "same_player_binary": player_identity()["player_exe_sha256"] == inp["unity"]["player_exe_sha256"],
    }


__all__ = [
    "NAV_UNITY_IDS",
    "UNITY_SCENARIOS",
    "UNITY_SCENARIO_IDS",
    "UnitySession",
    "prepare_unity",
    "replay_unity_nav",
    "replay_unity_run",
    "run_unity_nav",
    "run_unity_scenario",
]
