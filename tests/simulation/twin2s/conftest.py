from __future__ import annotations

import pytest

from conrad.schemas.ids import IdFactory
from conrad.sim.scenarios.pipeline_inspection import build_pipeline_inspection_scenario


@pytest.fixture(scope="module")
def t2s_scenario():
    return build_pipeline_inspection_scenario(11, IdFactory(11))
