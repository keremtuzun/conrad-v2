"""run_scenario: truth-side driver of one integrated mission, writing a complete, replayable run bundle.

``artifacts/runs/<run_id>/``: config.resolved.yaml, events.jsonl, conrad.sqlite (revisions, provenance,
commands), objects/ (payloads), mission/ (deployment artifacts), truth/ (evaluation only), reports/metrics.json
and bundle_manifest.json (replay inputs + digests).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from conrad.domains.technical.spatial_mission import MODEL_VERSION as SPATIAL_MODEL_VERSION
from conrad.orchestration.artifacts import write_mission_artifacts
from conrad.orchestration.association import SPATIAL_ASSOCIATION_VERSION
from conrad.orchestration.evaluation import evaluate_run_dir
from conrad.orchestration.mission import MissionRuntime
from conrad.orchestration.mission_config import MissionRuntimeConfig, runtime_config
from conrad.persistence.db import make_engine, migrate
from conrad.persistence.replay_store import write_bundle_manifest
from conrad.persistence.repository import Repository
from conrad.runtime.event_log import EventLog, RunContext
from conrad.schemas.base import ARCHITECTURE_ID, STACK_ID
from conrad.schemas.events import EventType
from conrad.schemas.ids import IdFactory
from conrad.schemas.structural_sensor import DETECTABILITY_VERSION, StructuralSensorModelV2
from conrad.schemas.structural_support import STRUCTURAL_SUPPORT_VERSION
from conrad.settings import REPO_ROOT, ConradSettings, load_settings, snapshot_yaml
from conrad.sim.mission.capture import TAPE, RecordingHardware, Tape, record_driver_event, write_capture_files
from conrad.sim.mission.obstacles import add_lane_obstacle
from conrad.sim.mission.options import MissionWorldOptions, world_options
from conrad.sim.mission.scenarios import resolve
from conrad.sim.mission.spatial_registration import REGISTRATION_VERSION
from conrad.sim.mission.spatial_support import VISIBILITY_CERTIFICATE_VERSION
from conrad.sim.mission.world import MissionWorld
from conrad.twins.twin2t.spatial_field import EVOLUTION_VERSION, TRUTH_VERSION

PRODUCER = "conrad-mission-0.1"
DRIVER = "conrad.sim.mission.driver"


def spatial_version_contract(sensor: StructuralSensorModelV2) -> dict[str, str]:
    """One version declaration shared by kernel, Unity, and replay."""
    return {
        "truth": TRUTH_VERSION,
        "evolution": EVOLUTION_VERSION,
        "sensor": sensor.version,
        "sensor_config_digest": sensor.digest,
        "support": STRUCTURAL_SUPPORT_VERSION,
        "registration": REGISTRATION_VERSION,
        "visibility_certificate": VISIBILITY_CERTIFICATE_VERSION,
        "model2t": SPATIAL_MODEL_VERSION,
        "detectability": DETECTABILITY_VERSION,
        "spatial_association": SPATIAL_ASSOCIATION_VERSION,
    }


def _git_commit() -> str:
    """Read-only lookup of the checked-out commit (no git command is executed)."""
    git_dir = REPO_ROOT / ".git"
    if git_dir.is_file():
        pointer = git_dir.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir: "):
            return "UNAVAILABLE_INVALID_GIT_POINTER"
        named = Path(pointer[8:])
        git_dir = named if named.is_absolute() else REPO_ROOT / named
    head = git_dir / "HEAD"
    if not head.exists():
        return "UNAVAILABLE_NO_GIT_METADATA"
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref: "):
        common_file = git_dir / "commondir"
        common = git_dir
        if common_file.exists():
            named = Path(common_file.read_text(encoding="utf-8").strip())
            common = named if named.is_absolute() else git_dir / named
        for base in (git_dir, common):
            target = base / ref[5:]
            if target.exists():
                return target.read_text(encoding="utf-8").strip()
        packed = common / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref[5:]:
                    return parts[0]
        return "UNAVAILABLE_UNRESOLVED_REF"
    return ref


def settings_for(config: str | Path | ConradSettings) -> ConradSettings:
    return config if isinstance(config, ConradSettings) else load_settings(config)


def default_run_id(scenario_id: str, settings: ConradSettings) -> str:
    return f"{scenario_id}-s{settings.run.seed}-{settings.config_digest()[:8]}"


@dataclass
class MissionSession:
    """One prepared mission: truth world + deployment runtime, stepped by the driver."""

    scenario_id: str
    settings: ConradSettings
    run_name: str
    run_dir: Path
    world: MissionWorld
    runtime: MissionRuntime
    log: EventLog
    repo: Repository
    engine: Any
    rcfg: MissionRuntimeConfig
    wopts: MissionWorldOptions
    inspection_s: float | None = None
    tape: Tape | None = None
    run_uuid: Any = None

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
        world.recorder.finish(world.hardware.truth_access(), self.run_dir)
        if self.tape is not None:
            self.tape.marker("artifacts")
        runtime_metrics = write_mission_artifacts(rt, self.run_dir)
        if self.tape is not None:
            write_capture_files(
                self.run_dir, self.tape, world.context, rt.robot_config, self.run_uuid, PRODUCER
            )
        self.engine.dispose()
        report = evaluate_run_dir(self.run_dir)
        report["runtime"] = runtime_metrics
        (self.run_dir / "reports").mkdir(exist_ok=True)
        (self.run_dir / "reports" / "metrics.json").write_text(
            json.dumps(report, indent=1, default=str), encoding="utf-8"
        )
        inputs = replay_inputs(
            self.scenario_id,
            self.settings,
            world,
            self.rcfg.model_dump(mode="json"),
            self.wopts.model_dump(mode="json"),
            self.run_name,
        )
        engine = make_engine(self.run_dir / "conrad.sqlite")
        try:
            digests = Repository(engine).referenced_digests(rt.s.run_id)
        finally:
            engine.dispose()
        write_bundle_manifest(self.run_dir, self.run_name, inputs, digests)
        return {"run_id": self.run_name, "run_dir": str(self.run_dir), "report": report}


def prepare(
    scenario_id: str,
    config: str | Path | ConradSettings,
    run_id: str | None = None,
    runs_root: Path | None = None,
    stored_world: MissionWorldOptions | None = None,
    stored_runtime: MissionRuntimeConfig | None = None,
    capture: bool = True,
) -> MissionSession:
    """``stored_world`` / ``stored_runtime`` (replay) replace the options resolved from the current scenario table."""
    settings = settings_for(config)
    seed = settings.run.seed
    world_raw, runtime_raw = resolve(scenario_id, dict(settings.sim.get("mission", {})))
    wopts = stored_world or world_options(world_raw)
    rcfg = stored_runtime or runtime_config(runtime_raw)
    validate_spatial_mission_selection(wopts, rcfg)
    ratio = rcfg.control_period_s / wopts.physics_dt_s
    if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
        raise ValueError(
            f"control_period_s {rcfg.control_period_s} must be an integer multiple of physics_dt_s "
            f"{wopts.physics_dt_s}; otherwise simulated time drifts from the control clock"
        )
    run_name = run_id or default_run_id(scenario_id, settings)
    root = runs_root or settings.resolve(settings.paths.runs_dir)
    run_dir = root / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "config.resolved.yaml").write_text(snapshot_yaml(settings), encoding="utf-8")
    ids = IdFactory(seed)
    run_uuid = ids.child("run").new()
    db_path = run_dir / "conrad.sqlite"
    migrate(db_path)
    engine = make_engine(db_path)
    repo = Repository(engine)
    world = MissionWorld.build(seed, scenario_id, wopts, run_uuid, run_dir / "objects")
    if wopts.lane_obstacle is not None:
        add_lane_obstacle(world, wopts.lane_obstacle)
    tape = Tape(run_dir / TAPE) if capture else None
    hw = world.hardware if tape is None else RecordingHardware(world.hardware, tape)
    ctx = RunContext(
        run_uuid, world.context.mission_id, None, ids.child("events"), hw.now_ns, hw.clock_domain(), PRODUCER
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
        seed,
        operator_inbox=run_dir / "notes" / "operator_requests.jsonl",
    )
    world.hardware.set_estimated_pose_provider(runtime.estimated_pose)
    world.advance(0.1)
    if tape is not None:
        tape.marker("start")
    runtime.start()
    return MissionSession(
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
        None,
        tape,
        run_uuid,
    )


def validate_spatial_mission_selection(wopts: MissionWorldOptions, rcfg: MissionRuntimeConfig) -> None:
    """Truth-side launch check; matching declarations are independently stored on both sides."""
    spatial_truth = wopts.twin2t_truth_model == "spatial_v1"
    spatial_belief = rcfg.model2t_backend == "spatial_v1"
    if spatial_truth != spatial_belief:
        raise ValueError("spatial mission requires matching Twin2T truth and Model2T backend selections")
    if not spatial_truth:
        return
    assert wopts.spatial_truth is not None and wopts.spatial_sensor_model is not None
    settings = rcfg.model2t_spatial
    if settings is None:
        raise ValueError("spatial Model2T settings missing")
    if (
        int(settings["axial_cells"]) != wopts.spatial_truth.axial_cells
        or int(settings["sectors"]) != wopts.spatial_truth.sectors
    ):
        raise ValueError("spatial truth and belief grid declarations differ")
    if StructuralSensorModelV2.model_validate(settings["sensor"]).digest != wopts.spatial_sensor_model.digest:
        raise ValueError("spatial truth and belief sensor configurations differ")


def run_scenario(
    scenario_id: str,
    config: str | Path | ConradSettings,
    run_id: str | None = None,
    runs_root: Path | None = None,
    stored_world: MissionWorldOptions | None = None,
    stored_runtime: MissionRuntimeConfig | None = None,
) -> dict[str, Any]:
    session = prepare(scenario_id, config, run_id, runs_root, stored_world, stored_runtime)
    session.run()
    return session.finish()


def replay_inputs(
    scenario_id: str,
    settings: ConradSettings,
    world: MissionWorld,
    runtime_cfg: dict[str, Any],
    world_cfg: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    sensors = [s.model_dump(mode="json") for s in world.context.sensors]
    spatial = runtime_cfg["model2t_backend"] == "spatial_v1"
    if spatial:
        sensor = StructuralSensorModelV2.model_validate(runtime_cfg["model2t_spatial"]["sensor"])
        spatial_versions = spatial_version_contract(sensor)
    else:
        spatial_versions = None
    return {
        "scenario_id": scenario_id,
        "run_name": run_name,
        "scenario_seed": settings.run.seed,
        "scenario_version": world.scenario.scenario_version,
        "twin_versions": {
            "twin2s": world.scenario.scenario_version,
            "twin2t": world.t2t.config.generator_version,
            "twin2e": "twin2e" if world.t2e is None else type(world.t2e).__name__,
        },
        "model_versions": {
            "model2t": SPATIAL_MODEL_VERSION if spatial else "model2t-analytic-0.1.0",
            "planner": runtime_cfg["planner"],
            "producer": PRODUCER,
        },
        "robot_config_digest": world.hardware.robot_config_digest(),
        "sensor_configuration": hashlib.sha256(json.dumps(sensors, sort_keys=True).encode()).hexdigest(),
        "config_digest": settings.config_digest(),
        "config_resolved": json.loads(settings.canonical_json()),
        "mission_world_options": world_cfg,
        "mission_runtime_config": runtime_cfg,
        "spatial_versions": spatial_versions,
        "git_commit": _git_commit(),
        "seeds": {"run": settings.run.seed},
        "architecture_id": ARCHITECTURE_ID,
        "stack_id": STACK_ID,
    }


def load_resolved_settings(run_dir: Path) -> ConradSettings:
    data = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    return ConradSettings.model_validate(data)
