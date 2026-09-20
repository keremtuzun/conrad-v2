"""UnityMissionWorld: the shared scenario and twins of ``MissionWorld`` with Unity V2 as the vehicle simulator.

TRUTH PLANE. Formal execution path of gates I1/I3 (ADR-0008):

    SharedScenario -> Twin2S / Twin2T -> Unity (static colliders from the Twin2S primitives) -> Unity sensors
      -> Observation -> ECMER / geometric encoder -> Model2S (+ Model2T for structural payloads)

What comes from where:

* Vehicle physics, IMU, pressure depth, power, thrusters, the range imager (DEPTH_RANGE), the imaging sonar and
  the RGB camera are real Unity sensors driven through ``UnityRobotHardware`` (lock-step TCP).
* Structural inspection readings come from Twin2T, their visibility from the Twin2S oracle evaluated at the
  Unity-reported TRUE pose (truth endpoint, read on this side only). The USBL-like position fix is the Unity true
  pose plus noise. Both reuse ``MissionSensorSuite`` exactly as the kernel path does.
* Unity range/sonar frames are converted into the declared Model2S payload formats by pure functions below
  (``range_image_to_model2s``, ``sonar_to_model2s``). Model2S semantics are unchanged; the raw wire payload stays
  in the object store and every derived observation names its digest (provenance to the raw Unity frame).

The deployment side (``MissionRuntime``) receives an RHI object and the same mission context as on the kernel
path; it cannot tell which simulator it is attached to except through declared capabilities.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import numpy as np
import yaml
from pydantic import Field

from conrad.adapters.unity.conversion import PayloadStore, UnityBridgeConfig
from conrad.adapters.unity.hardware import UnityRobotHardware
from conrad.orchestration.mission_context import MissionContext
from conrad.persistence.object_store import ObjectStore
from conrad.robotics.hardware.config import load_robot_config
from conrad.schemas.base import ConradModel
from conrad.schemas.frames import Pose, quat_from_euler
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.robot import RobotCapabilities, RobotConfig
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Scenario, SensorSpec
from conrad.settings import REPO_ROOT
from conrad.sim.kernel import TruthAccess
from conrad.sim.mission.obstacles import add_lane_obstacle
from conrad.sim.mission.options import MissionWorldOptions
from conrad.sim.mission.registry import RegistryMapping
from conrad.sim.mission.sensing import MissionSensorSuite
from conrad.sim.mission.truth import MissionTruthRecorder
from conrad.sim.mission.world import ROBOT_CONFIG, MissionWorld, _eco_event
from conrad.sim.unity.eco_scene import EcoConversionReport, EcoSceneOptions, eco_scene, optics_update
from conrad.sim.unity.player import UnityPlayerSession, scenario_document
from conrad.sim.unity.truth import TruthVehicleState, UnityTruthClient
from conrad.sim.unity.twin_scene import SceneConversionReport, TwinSceneOptions, twin_scene
from conrad.twins.twin2e import Twin2E
from conrad.twins.twin2s.twin import Twin2S
from conrad.twins.twin2t import Twin2T

UNITY_PAYLOAD_SENSORS = (
    "structural_inspection",
    "structural_inspection_aux",
    "depth_range_imager",
    "imaging_sonar",
    "rgb_camera",
    "usbl_position_fix",
)
# Declared only when the mission has an ecological twin (gate I6): Twin2E probe and survey, rendered on the Python side
# at the Unity TRUE pose by the same MissionSensorSuite as on the kernel path.
UNITY_ECO_SENSORS = ("environmental_probe", "ecological_survey")
RANGE_SENSOR, SONAR_SENSOR, CAMERA_SENSOR = "range_imager", "sonar", "camera"
FIX_OUTAGE = "FIX_OUTAGE"
"""The only ``FaultSpec.type`` the Unity mission path realises (an outage of the synthetic USBL-like fix)."""
CONVERTER_VERSION = "unity_to_model2s_v1"
UNITY_ROBOT_DIR = "artifacts/unity/robot"


class UnityConversionError(ValueError):
    """A Unity frame does not match the sensor configuration Model2S was given."""


class UnityWorldOptions(ConradModel):
    """Truth-side Unity settings. Every value is a SYNTHETIC_ONLY simulation setting."""

    physics_dt_ns: int = Field(default=5_000_000, gt=0)
    camera_rate_hz: float = Field(default=1.0, gt=0)
    sonar_noise_std: float = Field(
        default=0.005, ge=0, description="intensity-proxy noise; < detect threshold"
    )
    sonar_attenuation_per_m: float = Field(default=0.02, ge=0)
    sonar_bin_oversampling: int = Field(default=2, ge=1, description="Unity bins per Model2S bin (rebinned)")
    water_density_kgm3: float | None = Field(default=None, description="None = neutral for the RobotConfig")
    request_timeout_ms: int = Field(default=30_000, gt=0)
    startup_timeout_s: float = Field(default=90.0, gt=0)
    graphics: bool = True
    player_path: str | None = None
    scene: TwinSceneOptions = TwinSceneOptions()
    eco: EcoSceneOptions = EcoSceneOptions()


# ------------------------------------------------------------------------------------ RobotConfig variant
def _sourced(value: Any, units: str) -> dict[str, Any]:
    return {"value": value, "units": units, "source": "SYNTHETIC_ONLY"}


def _mount(spec: SensorSpec) -> list[float]:
    return [*spec.mount_pose.position_m, *spec.mount_pose.orientation_wxyz]


def unity_robot_config(
    base: RobotConfig,
    geometric: tuple[SensorSpec, ...],
    geometric_period_s: float,
    opts: UnityWorldOptions,
) -> RobotConfig:
    """``base`` plus the Unity range imager, the Twin2S-declared sonar geometry and AHRS orientation.

    The pressure sensor keeps the base config (no bias): Unity reports the true depth below its surface at the
    scenario's ``surface_z_m``, and the deployment estimator takes the surface height from the mission context
    (``MissionContext.water_surface_z_m``). The former -surface_z_m bias workaround is gone (I2 repair).
    """
    by_mod = {s.modality: s for s in geometric}
    if set(by_mod) != {"DEPTH_RANGE", "SONAR"}:
        raise ValueError(f"expected one DEPTH_RANGE and one SONAR sensor spec, got {sorted(by_mod)}")
    rng_spec, son = by_mod["DEPTH_RANGE"], by_mod["SONAR"]
    doc = base.model_dump(mode="json")
    rate = 1.0 / geometric_period_s
    sensors: list[dict[str, Any]] = []
    for s in doc["sensors"]:
        s = dict(s)
        if s["modality"] == "IMU":
            s["parameters"] = {"report_orientation": True, "gyro_noise_std": 0.002}
        elif s["modality"] == "RGB":
            s["rate_hz"] = _sourced(opts.camera_rate_hz, "Hz")
            s["parameters"] = {"width_px": 64, "height_px": 48, "vertical_fov_deg": 60.0}
        elif s["modality"] == "SONAR":
            p = son.parameters
            rmin, rmax, nr = float(p["min_range_m"]), float(p["max_range_m"]), int(p["n_range_bins"])
            s["sensor_name"] = SONAR_SENSOR
            s["frame_id"] = son.frame_id
            s["mount_pose_body"] = _sourced(_mount(son), "m,quat_wxyz")
            s["rate_hz"] = _sourced(rate, "Hz")
            s["noise_std"] = _sourced(opts.sonar_noise_std, "native")
            s["parameters"] = {
                "beams": int(p["n_beams"]),
                "bins": math.ceil(opts.sonar_bin_oversampling * nr * rmax / (rmax - rmin)),
                "range_max_m": rmax,
                "horizontal_fov_rad": math.radians(float(p["hfov_deg"])),
                "vertical_fov_rad": math.radians(float(p["vfov_deg"])),
                "elevation_rays": int(p["n_elevation_rays"]),
                "attenuation_per_m": opts.sonar_attenuation_per_m,
                "false_return_prob": 0.0,
            }
        sensors.append(s)
    p = rng_spec.parameters
    sensors.append(
        {
            "sensor_name": RANGE_SENSOR,
            "modality": "DEPTH_RANGE",
            "frame_id": rng_spec.frame_id,
            "mount_pose_body": _sourced(_mount(rng_spec), "m,quat_wxyz"),
            "rate_hz": _sourced(rate, "Hz"),
            "noise_std": _sourced(float(p["range_noise_sigma_m"]), "m"),
            "bias": _sourced(0.0, "m"),
            "drift_per_s": _sourced(0.0, "m/s"),
            "latency_s": _sourced(0.01, "s"),
            "parameters": {
                "width_px": int(p["width_px"]),
                "height_px": int(p["height_px"]),
                "hfov_deg": float(p["hfov_deg"]),
                "max_range_m": float(p["max_range_m"]),
                "min_range_m": float(p.get("min_range_m", 0.0)),
                "dropout_prob": float(p.get("dropout_prob", 0.0)),
            },
        }
    )
    doc["sensors"] = sensors
    doc["config_name"] = f"{base.config_name}_unity_mission"
    return RobotConfig.model_validate(doc)


def write_robot_config(robot: RobotConfig) -> str:
    """Content-addressed YAML under ``artifacts/unity/robot``; returns the repo-relative path for settings."""
    rel = f"{UNITY_ROBOT_DIR}/{robot.config_name}-{robot.content_digest()[:12]}.yaml"
    path = REPO_ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# GENERATED by conrad.sim.mission.unity_world (SYNTHETIC_ONLY Unity variant). Do not edit.\n"
    text += yaml.safe_dump(robot.model_dump(mode="json"), sort_keys=True)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    if load_robot_config(rel).content_digest() != robot.content_digest():
        raise RuntimeError("RobotConfig YAML round trip changed its digest")
    return rel


def neutral_density(robot: RobotConfig) -> float:
    return float(robot.mass_kg.value) / float(robot.displaced_volume_m3.value)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------ frame converters
def range_image_to_model2s(raw: np.ndarray, spec: SensorSpec) -> np.ndarray:
    """``range_f32_hw_v1`` already follows the declared pinhole model (row-major, +X boresight, u right,
    v down, NaN = no return), so the conversion is a shape check and a copy."""
    p = spec.parameters
    shape = (int(p["height_px"]), int(p["width_px"]))
    if raw.shape != shape:
        raise UnityConversionError(f"range frame {raw.shape} != declared {shape}")
    return np.ascontiguousarray(raw, dtype=np.float32)


def sonar_to_model2s(raw: np.ndarray, range_max_m: float, spec: SensorSpec) -> np.ndarray:
    """Unity (beams x bins over [0, range_max], beam 0 = rightmost) -> Model2S (range_bins x beams over
    [min_range, max_range], beam 0 = leftmost). Intensities of Unity bins are summed into the Model2S bin that
    contains the Unity bin centre; bins outside [min_range, max_range) are dropped."""
    p = spec.parameters
    nb, nr = int(p["n_beams"]), int(p["n_range_bins"])
    rmin, rmax = float(p["min_range_m"]), float(p["max_range_m"])
    if raw.ndim != 2 or raw.shape[0] != nb:
        raise UnityConversionError(f"sonar frame {raw.shape} does not have {nb} beams")
    if abs(range_max_m - rmax) > 1e-6:
        raise UnityConversionError(f"sonar range_max {range_max_m} != declared max_range_m {rmax}")
    bins = raw.shape[1]
    centres = (np.arange(bins) + 0.5) * range_max_m / bins
    keep = (centres >= rmin) & (centres < rmax)
    idx = np.clip(((centres[keep] - rmin) / (rmax - rmin) * nr).astype(np.int64), 0, nr - 1)
    out = np.zeros((nr, nb), dtype=np.float64)
    np.add.at(out, idx, np.asarray(raw, dtype=np.float64)[::-1, keep].T)
    return out.astype(np.float32)


def raw_array(store: ObjectStore, raw: Observation) -> np.ndarray:
    """The raw Unity wire payload (little-endian binary, shape/dtype recorded in the PayloadRef)."""
    assert raw.payload_ref is not None
    ref = raw.payload_ref
    data = store.get_bytes(ref)
    return np.frombuffer(data, dtype=np.dtype(ref.dtype or "<f4").newbyteorder("<")).reshape(ref.shape)


def derive_model2s_array(store: ObjectStore, raw: Observation, spec: SensorSpec) -> np.ndarray:
    """Re-derivable conversion (used at run time and by the provenance audit)."""
    arr = raw_array(store, raw)
    if raw.modality is Modality.DEPTH_RANGE:
        return range_image_to_model2s(arr, spec)
    if raw.modality is Modality.SONAR:
        return sonar_to_model2s(arr, float(raw.sensor_context["range_max_m"]), spec)
    raise UnityConversionError(f"{raw.modality.value} has no Model2S conversion")


# ------------------------------------------------------------------------------------ truth tracking
@dataclass
class UnityTruthTracker:
    """Evaluation-only vehicle truth from the Unity truth endpoint (collisions against the Twin2S SDF)."""

    t2s: Twin2S
    vehicle_radius_m: float
    collision_count: int = 0
    min_clearance_m: float = float("inf")
    energy_used_j: float = 0.0
    trajectory: list[list[float]] = field(default_factory=list)
    _in_collision: bool = False
    _last_traj_s: float = -1e9

    def update(self, s: TruthVehicleState) -> None:
        p = np.asarray(s.pose.position_m, dtype=np.float64)
        clearance = float(self.t2s.world.sdf(p[None, :])[0]) - self.vehicle_radius_m
        self.min_clearance_m = min(self.min_clearance_m, clearance)
        if clearance < 0 and not self._in_collision:
            self.collision_count += 1
            self._in_collision = True
        elif clearance > 0.02:
            self._in_collision = False
        t = s.sim_time_ns / 1e9
        if t - self._last_traj_s + 1e-9 >= 1.0:
            self._last_traj_s = t
            self.trajectory.append([round(t, 3), *(float(v) for v in p), *s.pose.orientation_wxyz])


# ------------------------------------------------------------------------------------ hardware
class UnityMissionHardware(UnityRobotHardware):
    """``UnityRobotHardware`` plus the declared payload sensors (TRUTH-SIDE wiring, like ``MissionHardware``).

    The runtime calls only RobotHardwareInterface methods and ``get_payload_observations``.
    """

    def __init__(
        self,
        config: UnityBridgeConfig,
        robot_config: RobotConfig,
        ids: IdFactory,
        mission_id: UUID,
        run_id: UUID,
        payload_store: PayloadStore | None = None,
    ) -> None:
        super().__init__(config, robot_config, ids, mission_id, run_id, payload_store=payload_store)
        self._estimated: Callable[[], Pose | None] = lambda: None
        self._suite: MissionSensorSuite | None = None
        self._truth: UnityTruthClient | None = None
        self._specs: dict[str, SensorSpec] = {}
        self._object_store: ObjectStore | None = None
        self._derived_ids: IdFactory | None = None
        self._forwarded: dict[str, int] = {}
        self._truth_history: dict[int, TruthVehicleState] = {}
        self.forwarded_counts: dict[str, int] = {}
        self.capture_log: list[dict[str, Any]] = []
        self.ecological = False  # set by UnityMissionWorld.build when the mission has Twin2E

    def attach(
        self,
        suite: MissionSensorSuite,
        truth: UnityTruthClient,
        specs: dict[str, SensorSpec],
        store: ObjectStore,
        ids: IdFactory,
    ) -> None:
        self._suite, self._truth, self._specs = suite, truth, specs
        self._object_store, self._derived_ids = store, ids

    def set_estimated_pose_provider(self, provider: Callable[[], Pose | None]) -> None:
        self._estimated = provider

    def capabilities(self) -> RobotCapabilities:
        declared = UNITY_PAYLOAD_SENSORS + (UNITY_ECO_SENSORS if self.ecological else ())
        return super().capabilities().model_copy(update={"environmental_sensors": declared})

    def true_state(self) -> TruthVehicleState:
        """TRUTH: evaluation / payload rendering only; never handed to the runtime."""
        assert self._truth is not None
        s = self._truth.get_ground_truth()
        if s.sim_time_ns != self.now_ns():
            raise RuntimeError(f"truth endpoint at {s.sim_time_ns} ns, control at {self.now_ns()} ns")
        self._truth_history[s.sim_time_ns] = s
        while len(self._truth_history) > 64:
            self._truth_history.pop(next(iter(self._truth_history)))
        return s

    def truth_at(self, time_ns: int) -> TruthVehicleState | None:
        """TRUTH at a past control instant (exact when an acquisition falls on a control-step boundary)."""
        return self._truth_history.get(time_ns)

    def get_payload_observations(self) -> list[Observation]:
        """Payload readings due now. Each carries only the ESTIMATED pose."""
        if self._suite is None:
            return []
        now = self.now_ns()
        truth = self.true_state()
        est = self._estimated()
        stamp = TimeStamp(time_ns=now, clock_domain=self.clock_domain())
        out = self._suite.collect(now / 1e9, stamp, truth.pose, est)
        out += self._unity_frames(est, truth)
        return out

    def _unity_frames(self, est: Pose | None, truth: TruthVehicleState) -> list[Observation]:
        out: list[Observation] = []
        getters = (
            (RANGE_SENSOR, self.get_range_image),
            (SONAR_SENSOR, self.get_sonar),
            (CAMERA_SENSOR, self.get_camera),
        )
        for name, getter in getters:
            raw = getter()
            if raw is None or self._forwarded.get(name) == raw.timestamp.time_ns:
                continue
            self._forwarded[name] = raw.timestamp.time_ns
            self.forwarded_counts[name] = self.forwarded_counts.get(name, 0) + 1
            if raw.modality is Modality.RGB:
                out.append(raw.model_copy(update={"robot_pose_estimate": est}))
                continue
            derived = self._derive(raw, est)
            at = self.truth_at(raw.timestamp.time_ns)
            pose = (at or truth).pose
            self.capture_log.append(
                {
                    "sensor": name,
                    "acquisition_ns": raw.timestamp.time_ns,
                    "true_pose_time_ns": (at or truth).sim_time_ns,
                    "observation_id": str(derived.observation_id),
                    "raw_payload_digest": None if raw.payload_ref is None else raw.payload_ref.digest,
                    "true_position_m": list(pose.position_m),
                    "true_orientation_wxyz": list(pose.orientation_wxyz),
                }
            )
            out.append(derived)
        return out

    def _derive(self, raw: Observation, est: Pose | None) -> Observation:
        assert self._object_store is not None and self._derived_ids is not None
        name = RANGE_SENSOR if raw.modality is Modality.DEPTH_RANGE else SONAR_SENSOR
        spec = self._specs[name]
        arr = derive_model2s_array(self._object_store, raw, spec)
        ref = self._object_store.put_array(arr)
        assert raw.payload_ref is not None
        enc = {
            "layout": "HxW" if raw.modality is Modality.DEPTH_RANGE else "range_bins x beams",
            "units": "m" if raw.modality is Modality.DEPTH_RANGE else "relative",
            "missing": "NaN",
        }
        return Observation(
            observation_id=self._derived_ids.new(),
            mission_id=raw.mission_id,
            run_id=raw.run_id,
            trace_id=raw.trace_id,
            sensor_id=spec.sensor_id,
            modality=raw.modality,
            timestamp=raw.timestamp,
            sensor_frame=spec.frame_id,
            robot_pose_estimate=est,
            payload_ref=ref,
            sensor_health=raw.sensor_health,
            calibration_ref=spec.calibration_ref,
            sensor_context={
                "settings": dict(spec.parameters),
                "self_reported_health": raw.sensor_health.value,
                "encoding": enc,
                "source": {
                    "adapter": raw.sensor_context.get("adapter"),
                    "simulator_validity_level": raw.sensor_context.get("simulator_validity_level"),
                    "unity_sensor": name,
                    "raw_observation_id": str(raw.observation_id),
                    "raw_payload_digest": raw.payload_ref.digest,
                    "raw_layout": raw.sensor_context.get("layout"),
                    "raw_shape": raw.sensor_context.get("shape"),
                    "delivery_time_ns": raw.sensor_context.get("delivery_time_ns"),
                    "converter": CONVERTER_VERSION,
                },
            },
        )


# ------------------------------------------------------------------------------------ world
def _neutral_suite(suite: MissionSensorSuite, keep_t2e: bool = False) -> MissionSensorSuite:
    """The kernel path's payload suite without its Twin2S geometric channel (Unity renders geometry).

    With ``keep_t2e`` (gate I6) the Twin2E channels stay exactly as on the kernel: structural reading quality from
    ``observability_modifiers`` and the environmental probe / ecological survey."""
    opts = suite.opts.model_copy(update={"geometric_period_s": 1e12})
    sensors = dataclasses.replace(suite.sensors, geometric=())
    return dataclasses.replace(suite, opts=opts, sensors=sensors, t2e=suite.t2e if keep_t2e else None)


@dataclass
class UnityMissionWorld:
    """Same fields the mission driver reads from ``MissionWorld`` (duck-typed), backed by Unity."""

    seed: int
    scenario_id: str
    opts: MissionWorldOptions
    uopts: UnityWorldOptions
    scenario: Scenario
    t2s: Twin2S
    t2t: Twin2T
    t2e: Twin2E | None
    hardware: UnityMissionHardware
    context: MissionContext
    mapping: RegistryMapping
    target: UUID
    recorder: MissionTruthRecorder
    store: ObjectStore
    player: UnityPlayerSession
    truth_client: UnityTruthClient
    tracker: UnityTruthTracker
    robot_config: RobotConfig
    robot_config_path: str
    scene_report: SceneConversionReport
    scene_digest: str
    suite: MissionSensorSuite
    eco_report: EcoConversionReport | None = None
    optics_updates: list[dict[str, Any]] = field(default_factory=list)
    _pending_eco: list[Any] = field(default_factory=list)
    _pending_faults: list[Any] = field(default_factory=list)
    _next_optics_s: float = 0.0

    @classmethod
    def build(
        cls,
        seed: int,
        scenario_id: str,
        options: MissionWorldOptions,
        run_id: UUID,
        run_dir: Path,
        uopts: UnityWorldOptions | None = None,
    ) -> UnityMissionWorld:
        u = uopts or UnityWorldOptions()
        # FIX_OUTAGE is realised on this path (the synthetic USBL-like fix is rendered in Python, exactly as on
        # the kernel). Unity-side faults (thruster, sensor, power, leak) would have to go through the bridge
        # INJECT_FAULT and are still refused here; on the NAV path they are supported by ``run_unity_nav``.
        unsupported = sorted({f.type for f in options.faults if f.type != FIX_OUTAGE})
        if unsupported:
            raise ValueError(
                f"the Unity mission path realises only {FIX_OUTAGE} faults, not {unsupported}; Unity-side "
                "faults go through run_unity_nav's bridge injection"
            )
        if options.eco_events and not options.ecological_enabled:
            raise ValueError("ecological events need the ecological twin (ecological_enabled=true)")
        base = MissionWorld.build(seed, scenario_id, options, run_id, run_dir / "objects")
        if options.lane_obstacle is not None:
            # Gate I5 blocked-route scenario: one unregistered box on the transit lane, appended to the Twin2S
            # world BEFORE the Unity scene is converted, so the player renders it and the geometric sensors see
            # it exactly as on the kernel path (conrad.sim.mission.run.prepare does the same call).
            add_lane_obstacle(base, options.lane_obstacle)
        ctx = base.context
        geometric = tuple(s for s in ctx.sensors if s.modality in ("DEPTH_RANGE", "SONAR"))
        surface = float(base.scenario.environment["water_surface_z_m"])
        robot = unity_robot_config(load_robot_config(ROBOT_CONFIG), geometric, options.geometric_period_s, u)
        rel = write_robot_config(robot)
        scene, report = twin_scene(base.t2s.world, u.scene)
        scene_json = scene.to_json()
        eco_report: EcoConversionReport | None = None
        if (
            base.t2e is not None
        ):  # gate I6: Twin2E optics grid + biofouling colours (measured conversion error)
            scene_json, _, eco_report = eco_scene(base.t2e, base.t2s.world, scene_json, u.eco)
        environment: dict[str, Any] = {
            "water_density_kgm3": u.water_density_kgm3 or neutral_density(robot),
            "surface_z_m": surface,
            "turbidity": 0.0,  # uniform camera proxy; the optics grid replaces it when Twin2E is present
        }
        if any(
            options.current_mps
        ):  # same constant current the kernel path gives SimKernel (Conrad WORLD, m/s)
            environment["current"] = {
                "kind": "constant",
                "velocity_mps": [float(v) for v in options.current_mps],
            }
        scene_digest = hashlib.sha256(json.dumps(scene_json, sort_keys=True).encode()).hexdigest()
        lane = ctx.transit_lane
        yaw = math.atan2(lane[1][1] - lane[0][1], lane[1][0] - lane[0][0])
        scenario_doc = scenario_document(
            IdFactory(seed).child("unity_scenario"),
            robot,
            seed=seed,
            initial_position_m=lane[0],
            initial_orientation_wxyz=quat_from_euler(0.0, 0.0, yaw),
            environment=environment,
            scenario_version=f"unity-mission:{base.scenario.scenario_version}",
        )
        player = UnityPlayerSession(
            robot,
            run_dir / "unity",
            scenario_doc,
            experiment_overrides={"physics_dt_ns": u.physics_dt_ns, "experiment_name": scenario_id},
            player_path=u.player_path,
            graphics=u.graphics,
            startup_timeout_s=u.startup_timeout_s,
            request_timeout_ms=u.request_timeout_ms,
        )
        player.__enter__()
        try:
            ids = IdFactory(seed)
            hw = UnityMissionHardware(
                player.bridge_config(), robot, ids.child("unity_rhi"), ctx.mission_id, run_id, base.store
            )
            hw.connect()
            ack = hw.configure_scene(scene_json)
            hw.reset(seed)
            truth = player.truth()
            truth.connect(ids.child("unity_truth").new().hex)
            suite = base.hardware.suite
            assert suite is not None
            specs = {
                RANGE_SENSOR: next(s for s in geometric if s.modality == "DEPTH_RANGE"),
                SONAR_SENSOR: next(s for s in geometric if s.modality == "SONAR"),
            }
            hw.ecological = base.t2e is not None
            neutral = _neutral_suite(suite, keep_t2e=hw.ecological)
            hw.attach(neutral, truth, specs, base.store, ids.child("unity_payload"))
            tracker = UnityTruthTracker(base.t2s, 0.5 * max(robot.dimensions_m.value))  # type: ignore[arg-type]
            base.recorder.meta.update(
                {
                    "backend": "unity",
                    "unity_scene_digest": scene_digest,
                    "unity_scene_primitives": ack.primitive_count,
                    "unity_scene_conversion": report.model_dump(mode="json"),
                    "unity_robot_config": rel,
                    "unity_robot_config_digest": robot.content_digest(),
                    "unity_eco_conversion": None
                    if eco_report is None
                    else eco_report.model_dump(mode="json"),
                }
            )
        except BaseException:
            player.stop()
            raise
        world = cls(
            seed,
            scenario_id,
            options,
            u,
            base.scenario,
            base.t2s,
            base.t2t,
            base.t2e,
            hw,
            ctx,
            base.mapping,
            base.target,
            base.recorder,
            base.store,
            player,
            truth,
            tracker,
            robot,
            rel,
            report,
            scene_digest,
            neutral,
            eco_report,
            _pending_eco=sorted(options.eco_events, key=lambda e: e.t_s),
            _pending_faults=sorted(options.faults, key=lambda f: f.t_s),
            _next_optics_s=u.eco.update_period_s,
        )
        world.tracker.update(hw.true_state())
        return world

    # ------------------------------------------------------------------ time
    @property
    def t_s(self) -> float:
        return self.hardware.now_ns() / 1e9

    def due_faults(self, inspection_started_s: float | None = None) -> list[dict[str, Any]]:
        """Scheduled FIX_OUTAGE faults and ecological events whose time has come (as ``MissionWorld.due_faults``).

        Only ``FIX_OUTAGE`` is realised here; every other fault type is refused in ``build``."""
        fired: list[dict[str, Any]] = []
        keep = []
        for f in self._pending_faults:
            base = 0.0 if f.trigger == "TIME" else inspection_started_s
            if base is None or base + f.t_s > self.t_s + 1e-9:
                keep.append(f)
                continue
            self.suite.dynamic_fix_outages.append((self.t_s, self.t_s + (f.duration_s or 30.0)))
            fired.append({"kind": "FAULT", "fired_at_s": round(self.t_s, 3), **f.model_dump(mode="json")})
        self._pending_faults = keep
        while self._pending_eco and self._pending_eco[0].t_s <= self.t_s + 1e-9:
            ev = self._pending_eco.pop(0)
            if self.t2e is not None:
                self.t2e.step(1e-3, [_eco_event(ev.event_type, ev.parameters, self.t_s)])
                self._next_optics_s = self.t_s  # the field changed: refresh the Unity optics grid now
            fired.append({"kind": "ECOLOGICAL_EVENT", "event_type": ev.event_type, "t_s": ev.t_s})
        return fired

    def _refresh_optics(self) -> None:
        """Send the current Twin2E optics grid to Unity (incremental CONFIGURE_SCENE, replace=false)."""
        assert self.t2e is not None
        body, grid = optics_update(self.t2e, self.t2s.world, self.uopts.eco)
        ack = self.hardware.configure_scene(body)
        cells = grid.values()
        self.optics_updates.append(
            {
                "t_s": round(self.t_s, 6),
                "twin2e_t_s": round(self.t2e.t_s, 6),
                "unity_scene_digest": ack.scene_digest,
                "attenuation_min_per_m": float(cells.min()),
                "attenuation_max_per_m": float(cells.max()),
            }
        )
        self._next_optics_s = self.t_s + self.uopts.eco.update_period_s

    def advance(self, dt_s: float) -> None:
        step = self.uopts.physics_dt_ns
        n = round(dt_s * 1e9 / step)
        if n < 1 or abs(n * step - dt_s * 1e9) > 1.0:
            raise ValueError(f"dt {dt_s} s is not a whole number of Unity physics steps ({step} ns)")
        self.hardware.step(n * step)
        self.t2t.step(dt_s)
        if self.t2e is not None:
            lag = self.t_s - self.t2e.t_s
            if lag > 1e-6:
                self.t2e.step(lag)
            if self.t_s + 1e-9 >= self._next_optics_s:
                self._refresh_optics()
        self.tracker.update(self.hardware.true_state())

    def truth_access(self) -> TruthAccess:
        """Duck-typed evaluation summary for ``MissionTruthRecorder.finish`` (collisions, clearance, energy)."""
        power = self.hardware.get_power_state()
        self.tracker.energy_used_j = (
            0.0 if power is None or power.energy_used_j is None else power.energy_used_j
        )
        return cast(TruthAccess, self.tracker)

    def close(self) -> int | None:
        """Stop the player (idempotent). Returns its exit code."""
        try:
            self.hardware.close()
        finally:
            code = self.player.stop()
        return code
