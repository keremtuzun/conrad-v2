"""Launch the built Unity V2 headless player and connect to it (lock-step, loopback TCP).

``UnityPlayerSession`` writes the three inputs the player reads (RobotConfig export, shared Scenario,
ExperimentConfig), starts ``ConradSim.exe -batchmode`` as a child process bound to free loopback ports,
waits until the control endpoint accepts, and hands out a connected ``UnityRobotHardware`` (control
plane) and, when enabled, a ``UnityTruthClient`` (truth plane, training/evaluation only).

The player is found at, in order: the explicit ``player_path``, ``$CONRAD_UNITY_PLAYER``, then
``unity/ConradUnityV2/Builds/Win64/ConradSim.exe`` (built by ``BuildScript.BuildWindows64Player``).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import UUID

from conrad.adapters.unity.conversion import PayloadStore, UnityBridgeConfig
from conrad.adapters.unity.hardware import UnityRobotHardware
from conrad.adapters.unity.transport import TcpBridgeTransport, UnityBridgeError
from conrad.schemas.frames import WORLD, Pose, Quat, Vec3
from conrad.schemas.ids import IdFactory
from conrad.schemas.robot import RobotConfig
from conrad.schemas.world import RobotSpec, Scenario
from conrad.sim.unity.robot_export import robot_config_to_unity
from conrad.sim.unity.truth import UnityTruthClient

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAYER = REPO_ROOT / "unity" / "ConradUnityV2" / "Builds" / "Win64" / "ConradSim.exe"
EXPERIMENT_FORMAT = "conrad.unity.experiment.v1"

EXIT_REFUSED_INPUTS = 3  # ConradHeadless.ExitRefusedInputs


class UnityPlayerError(UnityBridgeError):
    """The player could not be started, refused its inputs, or exited early."""


def find_player(player_path: str | Path | None = None) -> Path | None:
    for candidate in (player_path, os.environ.get("CONRAD_UNITY_PLAYER"), DEFAULT_PLAYER):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def scenario_document(
    ids: IdFactory,
    robot: RobotConfig,
    *,
    seed: int,
    initial_position_m: Vec3,
    initial_orientation_wxyz: Quat = (1.0, 0.0, 0.0, 0.0),
    environment: dict[str, Any],
    scenario_version: str = "unity-u0",
) -> dict[str, Any]:
    """A real ``conrad.schemas.world.Scenario`` (no world entities) dumped for Unity."""
    if "water_density_kgm3" not in environment:
        raise ValueError("environment.water_density_kgm3 is required (Unity has no default)")
    scenario = Scenario(
        scenario_id=ids.new(),
        scenario_version=scenario_version,
        seed=seed,
        world_entities=(),
        environment=environment,
        robots=(
            RobotSpec(
                robot_id=robot.robot_id,
                robot_config_ref=f"{robot.config_name}@{robot.config_version}",
                initial_pose=Pose(
                    frame_id=WORLD, position_m=initial_position_m, orientation_wxyz=initial_orientation_wxyz
                ),
            ),
        ),
    )
    return scenario.model_dump(mode="json")


def experiment_document(
    *,
    seed: int,
    control_endpoint: str,
    truth_endpoint: str | None,
    physics_dt_ns: int = 5_000_000,
    clock_domain: str = "SIM",
    sensor_rendering: dict[str, Any] | None = None,
    power: dict[str, Any] | None = None,
    replay_log_path: str | None = None,
    thruster_noise_fraction: float = 0.0,
    experiment_name: str = "unity_u0",
) -> dict[str, Any]:
    """``conrad.unity.experiment.v1``. Every rendering/power value here is SYNTHETIC_ONLY."""
    doc: dict[str, Any] = {
        "format": EXPERIMENT_FORMAT,
        "experiment_name": experiment_name,
        "seed": seed,
        "clock_domain": clock_domain,
        "physics_dt_ns": physics_dt_ns,
        "lock_step": True,
        "validation_report_ref": None,
        "truth_endpoint_enabled": truth_endpoint is not None,
        "thruster_noise_fraction": thruster_noise_fraction,
        "power": power
        or {
            "source": "SYNTHETIC_ONLY",
            "compute_w": 15.0,
            "sensors_w": 5.0,
            "comms_w": 2.0,
            "thruster_coefficient": 0.8,
        },
        "sensor_rendering": sensor_rendering
        or {
            "camera": {"source": "SYNTHETIC_ONLY", "width_px": 64, "height_px": 48, "vertical_fov_deg": 60.0},
            "sonar": {
                "source": "SYNTHETIC_ONLY",
                "beams": 16,
                "bins": 64,
                "range_max_m": 20.0,
                "horizontal_fov_rad": 1.0,
                "attenuation_per_m": 0.02,
                "false_return_prob": 0.0,
            },
        },
        "bridge": {
            "control_endpoint": control_endpoint,
            "stream_endpoint": None,
            "truth_endpoint": truth_endpoint,
        },
        "faults": [],
    }
    if replay_log_path is not None:
        doc["replay_log_path"] = replay_log_path
        doc["replay_truth_every_n_steps"] = 1
    return doc


@dataclass
class UnityPlayerSession:
    """One player process. Use as a context manager; the process is always terminated on exit."""

    robot: RobotConfig
    run_dir: Path
    scenario: dict[str, Any]
    experiment_overrides: dict[str, Any] = field(default_factory=dict)
    player_path: str | Path | None = None
    graphics: bool = True
    startup_timeout_s: float = 60.0
    request_timeout_ms: int = 10_000
    enable_truth: bool = True
    robot_document: dict[str, Any] | None = None  # override (e.g. to prove Unity refuses OPEN parameters)
    process: subprocess.Popen[bytes] | None = None
    control_endpoint: str = ""
    truth_endpoint: str | None = None

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "player.log"

    def start(self) -> UnityPlayerSession:
        player = find_player(self.player_path)
        if player is None:
            raise UnityPlayerError("no Unity player binary; build it with BuildScript.BuildWindows64Player")
        port = free_loopback_port()
        self.control_endpoint = f"tcp://127.0.0.1:{port}"
        self.truth_endpoint = f"tcp://127.0.0.1:{free_loopback_port()}" if self.enable_truth else None
        seed = int(self.scenario["seed"])
        experiment = experiment_document(
            seed=seed, control_endpoint=self.control_endpoint, truth_endpoint=self.truth_endpoint
        )
        experiment.update(self.experiment_overrides)
        robot_doc = (
            self.robot_document if self.robot_document is not None else robot_config_to_unity(self.robot)
        )
        paths = {
            "-conradRobotConfig": self.run_dir / "robot_config.json",
            "-conradScenario": self.run_dir / "scenario.json",
            "-conradExperiment": self.run_dir / "experiment.json",
        }
        for path, doc in zip(paths.values(), (robot_doc, self.scenario, experiment), strict=True):
            path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
        args = [str(player), "-batchmode", "-logFile", str(self.log_path)]
        if not self.graphics:
            args.append("-nographics")
        for flag, path in paths.items():
            args += [flag, str(path)]
        args += ["-conradParentPid", str(os.getpid()), "-conradIdleExitS", "120"]
        args += ["-screen-width", "320", "-screen-height", "240", "-screen-fullscreen", "0"]
        self.args = args
        self.process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self

    def wait_ready(self) -> None:
        """Block until the control endpoint accepts, or raise with the player's own error lines."""
        deadline = time.monotonic() + self.startup_timeout_s
        port = int(self.control_endpoint.rsplit(":", 1)[1])
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                code = None if self.process is None else self.process.returncode
                raise UnityPlayerError(f"player exited with code {code}: {self.log_errors()}")
            if "[Conrad] bridge up" in self.log_text():
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                        return
                except OSError:
                    pass
            time.sleep(0.1)
        raise UnityPlayerError(f"player not ready after {self.startup_timeout_s} s: {self.log_errors()}")

    def wait_exit(self, timeout_s: float) -> int:
        assert self.process is not None
        return self.process.wait(timeout=timeout_s)

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def log_errors(self) -> str:
        lines = [ln for ln in self.log_text().splitlines() if "[Conrad]" in ln or "Exception" in ln]
        return " | ".join(lines[-8:])

    def bridge_config(self, **over: Any) -> UnityBridgeConfig:
        base: dict[str, Any] = {
            "transport": "tcp",
            "control_endpoint": self.control_endpoint,
            "request_timeout_ms": self.request_timeout_ms,
            "connect_timeout_ms": 5000,
            "minimum_validity_level": "L1_APPROXIMATE_PHYSICS",
        }
        base.update(over)
        return UnityBridgeConfig.model_validate(base)

    def hardware(
        self,
        ids: IdFactory,
        mission_id: UUID,
        run_id: UUID,
        payload_store: PayloadStore | None = None,
        **config_over: Any,
    ) -> UnityRobotHardware:
        return UnityRobotHardware(
            self.bridge_config(**config_over),
            self.robot,
            ids,
            mission_id,
            run_id,
            payload_store=payload_store,
        )

    def truth(self, clock_domain: str = "SIM") -> UnityTruthClient:
        """TRUTH PLANE ONLY (evaluation of the simulator itself)."""
        if self.truth_endpoint is None:
            raise UnityPlayerError("truth endpoint disabled for this session")
        transport = TcpBridgeTransport(self.truth_endpoint, None, ("127.0.0.1",), self.request_timeout_ms)
        return UnityTruthClient(transport, self.robot.content_digest(), clock_domain)

    def stop(self) -> int | None:
        if self.process is None:
            return None
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        return self.process.returncode

    def __enter__(self) -> UnityPlayerSession:
        self.start()
        try:
            self.wait_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.stop()
