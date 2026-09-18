"""Twin2E -> ECMER-E -> Model2E episode harness for the 2E experiments. EVALUATION PLANE.

Truth (Twin2E) is used only to generate observations and to score beliefs; Model2E receives
Evidence built from Observations with an ESTIMATED pose, plus a mission asset registry built from
the scenario's world entities and Twin2S positions (permitted mission context, not truth state).

SYNTHETIC_ONLY.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from conrad.domains.ecological import EcologicalEncoder, Model2E, Model2EConfig, baseline_config
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence
from conrad.schemas.world import Scenario, SensorSpec
from conrad.twins.base import SensingContext
from conrad.twins.twin2e import Twin2E, Twin2EConfig, build_small_scenario, load_config

REPO = Path(__file__).resolve().parents[3]
SMALL_TWIN_CFG = REPO / "configs" / "sim" / "twin2e_test_small.yaml"
SESSILE_TYPES = ("pipeline_segment", "coral_patch")


def twin_config(config: Mapping[str, Any]) -> Twin2EConfig:
    return load_config(REPO / str(config.get("twin_config", SMALL_TWIN_CFG.relative_to(REPO))))


def env(**values: tuple[Any, str]) -> dict[str, dict[str, Any]]:
    return {k: {"value": v, "units": u} for k, (v, u) in values.items()}


def make_scenario(seed: int, tcfg: Twin2EConfig, config: Mapping[str, Any], **kw: Any) -> Scenario:
    ids = IdFactory(seed).child("twin2e")
    return build_small_scenario(
        ids,
        seed,
        tcfg,
        n_structures=int(config.get("n_structures", 3)),
        n_benthic=int(config.get("n_benthic", 3)),
        n_mobile=int(config.get("n_mobile", 1)),
        **kw,
    )


def make_twin(scenario: Scenario, tcfg: Twin2EConfig, seed: int) -> Twin2E:
    store = ObjectStore(tempfile.mkdtemp(prefix="eco_exp_"))
    twin = Twin2E(IdFactory(seed).child("twin2e-run"), store, tcfg)
    twin.initialize(scenario)
    return twin


def registry(scenario: Scenario) -> list[dict[str, Any]]:
    pos = scenario.spatial_state["entities"]
    return [
        {
            "registry_entity_id": e.id,
            "entity_type": e.entity_type,
            "position_m": pos[str(e.id)]["position_m"],
            "frame_id": e.reference_frame,
        }
        for e in scenario.world_entities
        if str(e.id) in pos
    ]


def model_config(config: Mapping[str, Any]) -> Model2EConfig:
    """Experiment-level Model2E overrides (``model2e:`` block), e.g. process noise matched to the
    twin's LOGGED ecology time acceleration."""
    return Model2EConfig.model_validate(dict(config.get("model2e", {})))


def make_models(
    variants: Sequence[str], seed: int, scenario: Scenario, base: Model2EConfig | None = None
) -> dict[str, Model2E]:
    out = {}
    for name in variants:
        m = Model2E(baseline_config(name, base), IdFactory(seed).child(f"model2e/{name}"))
        m.initialize({"asset_registry": registry(scenario), "clock_domain": "SIM"})
        out[name] = m
    return out


@dataclass
class Sensor:
    spec: SensorSpec
    position: np.ndarray
    period_steps: int
    phase: int
    index: int


@dataclass
class Episode:
    twin: Twin2E
    rng: np.random.Generator
    seed: int
    pose_sigma_m: float = 0.3
    ids: IdFactory = field(init=False)
    encoder: EcologicalEncoder = field(init=False)

    def __post_init__(self) -> None:
        self.ids = IdFactory(self.seed).child("episode")
        self.encoder = EcologicalEncoder(
            Model2EConfig(), IdFactory(self.seed).child("ecmer-e"), self.pose_sigma_m
        )
        self.mission_id, self.run_id = self.ids.new(), self.ids.new()

    def mooring(
        self, index: int, position: np.ndarray, fields: Sequence[str], period: int, phase: int
    ) -> Sensor:
        spec = SensorSpec(
            sensor_id=self.ids.new(),
            modality="ENVIRONMENTAL",
            frame_id=f"env_{index}",
            mount_pose=Pose(frame_id="ROBOT", position_m=(0.0, 0.0, 0.0)),
            rate_hz=1.0 / 900.0,
            parameters={"fields": list(fields)},
        )
        return Sensor(spec, np.asarray(position, dtype=np.float64), period, phase, index)

    def survey_sensor(self, radius_m: float) -> SensorSpec:
        return SensorSpec(
            sensor_id=self.ids.new(),
            modality="STRUCTURED",
            frame_id="survey_cam",
            mount_pose=Pose(frame_id="ROBOT", position_m=(0.0, 0.0, 0.0)),
            rate_hz=1.0 / 900.0,
            parameters={"level": "E0_ABSTRACT", "footprint_radius_m": radius_m},
        )

    def sense(self, spec: SensorSpec, true_pos: np.ndarray, pose_noise: bool) -> list[Evidence]:
        true_pose = Pose(
            frame_id="WORLD", position_m=(float(true_pos[0]), float(true_pos[1]), float(true_pos[2]))
        )
        est = true_pos + (self.rng.normal(0.0, self.pose_sigma_m, 3) if pose_noise else 0.0)
        est_pose = Pose(frame_id="WORLD", position_m=(float(est[0]), float(est[1]), float(est[2])))
        ctx = SensingContext(
            mission_id=self.mission_id,
            run_id=self.run_id,
            trace_id=self.ids.new(),
            sensor=spec,
            true_pose=true_pose,
            estimated_pose=est_pose,
            timestamp=self.twin.now(),
            degradation={},
        )
        out = []
        for sample in self.twin.generate_observation(ctx):
            ev, _ = self.encoder.encode(sample.observation, sample.observation.timestamp.time_ns + 1_000_000)
            out.append(ev)
        return out


def run_steps(
    twin: Twin2E,
    n_steps: int,
    dt_s: float,
    step_fn: Callable[[int], None],
) -> None:
    """step_fn(i) runs at the twin's current time (observe, update); then the twin advances."""
    for i in range(n_steps):
        step_fn(i)
        twin.step(dt_s)
    step_fn(n_steps)


def survey_waypoints(scenario: Scenario, altitude_m: float = 2.5) -> list[np.ndarray]:
    """Mission plan: hover above each registered sessile asset in turn (registry positions only)."""
    pos = scenario.spatial_state["entities"]
    return [
        np.asarray(pos[str(e.id)]["position_m"], dtype=np.float64) + np.array([0.0, 0.0, altitude_m])
        for e in scenario.world_entities
        if e.entity_type in SESSILE_TYPES and str(e.id) in pos
    ]
