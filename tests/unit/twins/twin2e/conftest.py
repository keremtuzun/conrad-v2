from __future__ import annotations

from pathlib import Path

import pytest

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import SensorSpec
from conrad.twins.base import SensingContext
from conrad.twins.twin2e import Twin2E, Twin2EConfig, build_small_scenario, load_config

REPO = Path(__file__).resolve().parents[4]
SMALL_CFG = REPO / "configs" / "sim" / "twin2e_test_small.yaml"


@pytest.fixture
def cfg() -> Twin2EConfig:
    return load_config(SMALL_CFG)


@pytest.fixture
def store(tmp_path: Path) -> ObjectStore:
    return ObjectStore(tmp_path / "objects")


@pytest.fixture
def make_twin(cfg, store):
    def _make(seed: int = 3, config: Twin2EConfig | None = None, scenario=None, **kw):
        ids = IdFactory(seed)
        c = config or cfg
        scen = scenario if scenario is not None else build_small_scenario(ids, seed, c, **kw)
        tw = Twin2E(IdFactory(seed + 1000), store, c)
        tw.initialize(scen)
        return tw

    return _make


def sensing_ctx(
    twin: Twin2E, modality: str, position, params=None, degradation=None, ts=None
) -> SensingContext:
    ids = IdFactory(99)
    pose = Pose(frame_id="WORLD", position_m=tuple(float(v) for v in position))
    sensor = SensorSpec(
        sensor_id=ids.new(),
        modality=modality,
        frame_id="sensor",
        mount_pose=Pose(frame_id="ROBOT", position_m=(0, 0, 0)),
        rate_hz=1.0,
        parameters=params or {},
    )
    return SensingContext(
        mission_id=ids.new(),
        run_id=ids.new(),
        trace_id=ids.new(),
        sensor=sensor,
        true_pose=pose,
        estimated_pose=None,
        timestamp=ts or twin.now(),
        degradation=degradation or {},
    )


@pytest.fixture
def ctx_for():
    return sensing_ctx
