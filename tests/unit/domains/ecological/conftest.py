from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from conrad.domains.ecological import Model2E, Model2EConfig, baseline_config
from conrad.persistence.db import make_engine, migrate
from conrad.persistence.repository import Repository
from conrad.schemas.frames import SpatialSupport
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Evidence, Modality, QualityContext, SensorHealth
from conrad.schemas.timebase import stamp

CLOCK = "SIM"


class EvidenceMaker:
    def __init__(self, seed: int = 11) -> None:
        self.ids = IdFactory(seed).child("test-evidence")
        self.run_id = self.ids.new()
        self.mission_id = self.ids.new()

    def _ev(self, t: float, pos, meas: dict, units: dict, modality: Modality, sigma: float = 0.3, rng=None):
        return Evidence(
            evidence_id=self.ids.new(),
            source_observation_id=self.ids.new(),
            mission_id=self.mission_id,
            run_id=self.run_id,
            trace_id=self.ids.new(),
            modality=modality,
            timestamp=stamp(t, CLOCK),
            created_time_ns=stamp(t, CLOCK).time_ns + 1000,
            embedding=tuple(float(v) for v in meas.values()),
            spatial_support=SpatialSupport(
                frame_id="WORLD", center_m=tuple(float(v) for v in pos), position_sigma_m=sigma
            ),
            reliability=0.95,
            aleatoric_uncertainty=0.05,
            sensor_context=QualityContext(sensor_health=SensorHealth.OK, range_m=rng),
            measurements=meas,
            measurement_units=units,
            provenance_id=self.ids.new(),
            encoder_version="test",
        )

    def field(self, name: str, value: float, t: float, pos=(0.0, 0.0, -10.0), units: str | None = None):
        u = units or {"temperature": "degC", "turbidity": "NTU", "light": "W m-2"}[name]
        return self._ev(t, pos, {name: value}, {name: u}, Modality.ENVIRONMENTAL)

    def cover(self, value: float, t: float, pos, rng: float = 3.0):
        return self._ev(
            t, pos, {"cover_fraction": value}, {"cover_fraction": "1"}, Modality.STRUCTURED, rng=rng
        )

    def detection(self, t: float, pos):
        return self._ev(t, pos, {"detection": 1.0}, {"detection": "1"}, Modality.STRUCTURED)


@pytest.fixture
def maker() -> EvidenceMaker:
    return EvidenceMaker()


REG_IDS = IdFactory(5).child("registry")
PIPE_ID: UUID = REG_IDS.new()
CORAL_ID: UUID = REG_IDS.new()
FISH_ID: UUID = REG_IDS.new()
PIPE_POS = (-20.0, 0.0, -18.0)
CORAL_POS = (20.0, 10.0, -18.0)
FISH_POS = (0.0, -20.0, -10.0)


def registry() -> list[dict]:
    return [
        {
            "registry_entity_id": PIPE_ID,
            "entity_type": "pipeline_segment",
            "position_m": PIPE_POS,
            "radius_m": 2.0,
        },
        {
            "registry_entity_id": CORAL_ID,
            "entity_type": "coral_patch",
            "position_m": CORAL_POS,
            "radius_m": 2.0,
            "stress_threshold_c": 24.0,
        },
        {"registry_entity_id": FISH_ID, "entity_type": "fish_school", "position_m": FISH_POS},
    ]


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    db = tmp_path / "eco.sqlite"
    migrate(db)
    return Repository(make_engine(db))


@pytest.fixture
def make_model(repo):
    def _make(
        variant: str = "cefd", with_repo: bool = True, seed: int = 3, config: Model2EConfig | None = None
    ):
        m = Model2E(
            baseline_config(variant, config), IdFactory(seed).child("model2e"), repo if with_repo else None
        )
        m.initialize({"asset_registry": registry(), "clock_domain": CLOCK})
        return m

    return _make


@pytest.fixture
def reg() -> dict:
    return {"pipe": (PIPE_ID, PIPE_POS), "coral": (CORAL_ID, CORAL_POS), "fish": (FISH_ID, FISH_POS)}
